"""
Config Resolution Audit - runs at startup, logs all active parameters.

Zero-risk module: read-only audit, no behavior changes.
Logs conflicts between JSON configs, code defaults, and runtime flags.

Usage:
    from configs.config_resolver import ConfigResolver
    resolver = ConfigResolver(production_config, sota_flags)
    resolver.resolve_and_log()
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class _Conflict:
    parameter: str
    authoritative_value: Any
    authoritative_source: str
    conflicting: List[Tuple[str, Any]]  # [(source, value), ...]


@dataclass
class _Warning:
    message: str


class ConfigResolver:
    """
    Resolve and audit all configuration parameters at startup.

    Reads from:
      - ProductionConfig (the merged JSON + code defaults)
      - SOTAFlags singleton (runtime feature toggles + risk thresholds)
      - JSON config files on disk (for detecting stale values)

    Outputs a human-readable audit log showing:
      - All resolved parameter values
      - Conflicts between sources
      - Warnings about stale/placeholder values
    """

    def __init__(self, config=None, flags=None):
        """
        Args:
            config: ProductionConfig instance (from main.py)
            flags: SOTAFlags instance (from configs.sota_flags)
        """
        self.config = config
        self.flags = flags
        self.resolved: Dict[str, Any] = {}
        self.conflicts: List[_Conflict] = []
        self.warnings: List[_Warning] = []

    def resolve_and_log(self) -> Dict[str, Any]:
        """Run full resolution and log results. Returns resolved params dict."""
        self._resolve_initial_capital()
        self._resolve_drawdown_thresholds()
        self._resolve_kill_switch()
        self._resolve_correlation()
        self._resolve_tranche_sizes()
        self._resolve_fees()
        self._resolve_exposure_caps()
        self._check_stale_configs()

        self._log_audit()
        return self.resolved

    # ------------------------------------------------------------------
    # Resolution methods
    # ------------------------------------------------------------------

    def _resolve_initial_capital(self):
        env_val = os.environ.get("HMATS_INITIAL_CAPITAL")
        if env_val:
            self.resolved["initial_capital"] = float(env_val)
            self.resolved["initial_capital_source"] = "env:HMATS_INITIAL_CAPITAL"
        elif self.config:
            self.resolved["initial_capital"] = self.config.initial_capital
            self.resolved["initial_capital_source"] = "ProductionConfig"
            if self.config.initial_capital >= 100000.0:
                self.warnings.append(_Warning(
                    f"initial_capital=${self.config.initial_capital:,.0f} looks like $100K "
                    f"placeholder. Actual account ~$10K. Set HMATS_INITIAL_CAPITAL=10000."
                ))
        else:
            self.resolved["initial_capital"] = 10000.0
            self.resolved["initial_capital_source"] = "fallback"

        # Check JSON configs for conflicts
        _json_vals = self._scan_json_initial_capital()
        ic = self.resolved["initial_capital"]
        conflicts = [(src, val) for src, val in _json_vals if abs(val - ic) > 1.0]
        if conflicts:
            self.conflicts.append(_Conflict(
                parameter="initial_capital",
                authoritative_value=ic,
                authoritative_source=self.resolved["initial_capital_source"],
                conflicting=conflicts,
            ))

    def _resolve_drawdown_thresholds(self):
        if not self.config:
            return

        self.resolved["dd_halt"] = self.config.hard_drawdown_halt
        self.resolved["dd_reduce"] = self.config.reduce_at_drawdown
        self.resolved["dd_critical"] = self.config.critical_drawdown

        # Check cloud_production.json for stale values
        cloud = self._load_json("configs/cloud_production.json")
        if cloud:
            risk = cloud.get("risk", {})
            cloud_halt = risk.get("hard_drawdown_halt")
            if cloud_halt and abs(cloud_halt - self.config.hard_drawdown_halt) > 0.001:
                self.conflicts.append(_Conflict(
                    parameter="hard_drawdown_halt",
                    authoritative_value=self.config.hard_drawdown_halt,
                    authoritative_source="ProductionConfig (merged)",
                    conflicting=[("cloud_production.json", cloud_halt)],
                ))

    def _resolve_kill_switch(self):
        if not self.flags:
            return

        self.resolved["kill_switch_dd"] = self.flags.RISK_MAX_DRAWDOWN_PCT
        self.resolved["kill_switch_daily"] = self.flags.RISK_DAILY_LOSS_HALT_PCT
        self.resolved["kill_switch_source"] = "sota_flags.py"

    def _resolve_correlation(self):
        if self.config:
            self.resolved["correlation_crisis"] = self.config.correlation_crisis

        if self.flags:
            self.resolved["correlation_collapse_threshold"] = (
                self.flags.RISK_CORRELATION_COLLAPSE_THRESHOLD
            )

        # global_exposure_cap thresholds are hardcoded - just document
        self.resolved["exposure_cap_elevated_corr"] = 0.92
        self.resolved["exposure_cap_high_corr"] = 0.96
        self.resolved["exposure_cap_note"] = "global_exposure_cap.py hardcoded"

    def _resolve_tranche_sizes(self):
        if not self.config:
            return

        tc = self.config.tranche_config
        if tc:
            cum = tc.get("sizes_cumulative", {})
            self.resolved["tranche_source"] = "JSON profile (cumulative)"
            self.resolved["tranche_cumulative"] = cum
            # Convert to individual
            c1 = cum.get("TRANCHE_1", 0.35)
            c2 = cum.get("TRANCHE_2", 0.65)
            c3 = cum.get("TRANCHE_3", 0.90)
            c4 = cum.get("TRANCHE_4", 1.00)
            self.resolved["tranche_individual"] = {
                "T1": round(c1, 4),
                "T2": round(c2 - c1, 4),
                "T3": round(c3 - c2, 4),
                "T4": round(c4 - c3, 4),
            }
        else:
            self.resolved["tranche_source"] = "code defaults (PositionSizingConfig)"
            self.resolved["tranche_individual"] = {
                "T1": 0.35, "T2": 0.30, "T3": 0.25, "T4": 0.10,
            }

    def _resolve_fees(self):
        try:
            from execution.fees import get_fee_calculator

            fee_calc = get_fee_calculator()
            monthly_volume = float(fee_calc.tracker.get_monthly_volume() or 0.0)
            maker_bps = fee_calc.blender.apply(0.0016, monthly_volume) * 10000.0
            taker_bps = fee_calc.blender.apply(0.0026, monthly_volume) * 10000.0

            self.resolved["fees_maker_bps"] = round(maker_bps, 2)
            self.resolved["fees_taker_bps"] = round(taker_bps, 2)
            self.resolved["fees_monthly_volume_usd"] = round(monthly_volume, 2)
            self.resolved["fees_in_free_tier"] = fee_calc.blender.is_in_free_tier(monthly_volume)
            self.resolved["fees_note"] = "Kraken+ effective blended fees from shared volume tracker"
        except Exception:
            # Kraken fee tier: free if <$10K/mo volume, then maker=16bps taker=26bps
            self.resolved["fees_maker_bps"] = 16
            self.resolved["fees_taker_bps"] = 26
            self.resolved["fees_note"] = "Kraken+ free tier <$10K/mo (0 fees)"

    def _resolve_exposure_caps(self):
        if not self.config:
            return
        self.resolved["max_gross_exposure"] = self.config.max_gross_exposure
        self.resolved["max_single_asset_pct"] = self.config.max_single_asset_pct
        self.resolved["max_leverage"] = self.config.max_leverage

    def _check_stale_configs(self):
        # Check if sota_config.json exists and warn
        sota_path = Path("configs/sota_config.json")
        if sota_path.exists():
            sota = self._load_json(str(sota_path))
            if sota and "_DEPRECATED" not in sota:
                self.warnings.append(_Warning(
                    "sota_config.json exists without _DEPRECATED marker. "
                    "This file is NOT loaded at runtime."
                ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_json(self, path: str) -> Optional[Dict]:
        try:
            p = Path(path)
            if p.exists():
                import json
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _scan_json_initial_capital(self) -> List[Tuple[str, float]]:
        """Scan JSON configs for initial_capital values."""
        results = []
        configs = [
            ("cloud_production.json", ["risk", "initial_capital_default"]),
            ("paper_trading.json", ["risk", "initial_capital"]),
            ("default.json", ["initial_capital"]),
            ("backtest.json", ["initial_capital"]),
        ]
        for filename, keys in configs:
            data = self._load_json(f"configs/{filename}")
            if not data:
                continue
            val = data
            for k in keys:
                if isinstance(val, dict):
                    val = val.get(k)
                else:
                    val = None
                    break
            if val is not None:
                results.append((filename, float(val)))
        return results

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_audit(self):
        lines = []
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines.append("=" * 64)
        lines.append(f"HMATS CONFIG RESOLUTION AUDIT - {ts}")
        lines.append("=" * 64)

        # Resolved values
        ic = self.resolved.get("initial_capital", "?")
        ic_src = self.resolved.get("initial_capital_source", "?")
        lines.append(f"[RESOLVED] Initial capital: ${ic:,.0f} (from {ic_src})")

        dd_h = self.resolved.get("dd_halt", "?")
        dd_r = self.resolved.get("dd_reduce", "?")
        dd_c = self.resolved.get("dd_critical", "?")
        if dd_h != "?":
            lines.append(
                f"[RESOLVED] Drawdown: reduce={dd_r:.0%} halt={dd_h:.0%} "
                f"critical={dd_c:.0%}"
            )

        ks_dd = self.resolved.get("kill_switch_dd", "?")
        ks_dl = self.resolved.get("kill_switch_daily", "?")
        if ks_dd != "?":
            lines.append(f"[RESOLVED] Kill-switch: DD={ks_dd:.0%} daily={ks_dl:.0%}")

        cc = self.resolved.get("correlation_crisis", "?")
        if cc != "?":
            lines.append(f"[RESOLVED] Correlation crisis: {cc}")

        ti = self.resolved.get("tranche_individual")
        if ti:
            t_src = self.resolved.get("tranche_source", "?")
            lines.append(
                f"[RESOLVED] Tranche: T1={ti['T1']:.0%} T2={ti['T2']:.0%} "
                f"T3={ti['T3']:.0%} T4={ti['T4']:.0%} ({t_src})"
            )

        mg = self.resolved.get("max_gross_exposure")
        ms = self.resolved.get("max_single_asset_pct")
        ml = self.resolved.get("max_leverage")
        if mg:
            lines.append(
                f"[RESOLVED] Exposure: gross={mg:.0%} single={ms:.0%} leverage={ml}x"
            )

        fm = self.resolved.get("fees_maker_bps", 16)
        ft = self.resolved.get("fees_taker_bps", 26)
        lines.append(f"[RESOLVED] Fees: maker={fm}bps taker={ft}bps (Kraken+: free<$10K)")

        # Conflicts
        for c in self.conflicts:
            conf_str = ", ".join(f"{src}={val}" for src, val in c.conflicting)
            lines.append(
                f"[CONFLICT] {c.parameter}: authoritative={c.authoritative_value} "
                f"({c.authoritative_source}) vs {conf_str}"
            )

        # Warnings
        for w in self.warnings:
            lines.append(f"[WARNING] {w.message}")

        lines.append("=" * 64)

        # Log as single block
        audit_text = "\n".join(lines)
        logger.info(f"\n{audit_text}")


# ======================================================================
# C5: Config Schema Validation
# ======================================================================

# Schemas define required/optional keys and value constraints for JSON configs.
# Unknown keys are flagged as potential typos (unless prefixed with "_").

_RISK_SCHEMA = {
    "required": [],
    "known_keys": {
        "initial_capital", "initial_capital_default", "initial_capital_env",
        "hard_drawdown_halt", "correlation_crisis", "max_position_pct",
        "max_drawdown", "daily_loss_limit", "position_limits",
        "max_gross_exposure", "max_leverage", "max_single_asset_pct",
        "max_asset_exposure", "reduce_at_drawdown", "halt_at_drawdown",
        "critical_drawdown",
    },
    "ranges": {
        "hard_drawdown_halt": (0.01, 0.50),
        "max_drawdown": (0.01, 0.50),
        "daily_loss_limit": (0.01, 0.20),
        "correlation_crisis": (0.80, 1.00),
        "max_position_pct": (0.05, 1.00),
        "max_leverage": (1.0, 10.0),
        "initial_capital": (1000, 10_000_000),
        "initial_capital_default": (1000, 10_000_000),
    },
}

_TRANCHE_SCHEMA = {
    "required": ["sizes_cumulative"],
    "known_keys": {
        "sizes_cumulative", "t2_require_4h_close", "t3_require_regime_aligned",
        "t3_min_profit_bps", "t4_min_profit_bps", "t4_min_time_hours",
        "escalation_min_bars", "entry_volume_collapse_threshold",
        "entry_volume_collapse_threshold_by_asset",
    },
    "ranges": {},
}


def validate_config_section(
    section: Dict, schema: Dict, section_name: str
) -> List[str]:
    """
    Validate a config section against a schema.

    Returns list of error/warning strings. Empty = valid.
    """
    errors: List[str] = []

    # Check required keys
    for key in schema.get("required", []):
        if key not in section:
            errors.append(f"[{section_name}] Missing required key: {key}")

    # Check value ranges
    ranges = schema.get("ranges", {})
    for key, value in section.items():
        if key.startswith("_"):
            continue
        if key in ranges and isinstance(value, (int, float)):
            lo, hi = ranges[key]
            if value < lo or value > hi:
                errors.append(
                    f"[{section_name}] {key}={value} out of range [{lo}, {hi}]"
                )

    # Check for unknown keys (potential typos)
    known = schema.get("known_keys", set())
    if known:
        for key in section:
            if key.startswith("_"):
                continue
            if key not in known:
                errors.append(f"[{section_name}] Unknown key (typo?): {key}")

    return errors


def validate_loaded_config(data: Dict) -> List[str]:
    """Validate a full loaded config dict. Returns list of issues."""
    all_errors: List[str] = []

    risk = data.get("risk", {})
    if risk:
        all_errors.extend(validate_config_section(risk, _RISK_SCHEMA, "risk"))

    tranche = data.get("tranche", {})
    if tranche:
        all_errors.extend(
            validate_config_section(tranche, _TRANCHE_SCHEMA, "tranche")
        )

    return all_errors
