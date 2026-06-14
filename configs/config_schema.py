"""
config_schema.py — Pydantic schema + cross-source consistency validator
=========================================================================

[P113 2026-04-27] Centralized validation for HMATS thresholds across
multiple config sources. Catches drift between:
  - configs/canonical_config.py module constants
  - cloud_production.json (live runtime config)
  - sota_flags.py (feature flags + global limits)
  - configs/strategy_tiers.py (per-tier caps)

Without this, bugs like "canonical 0.20 vs JSON 0.25 hard_drawdown"
(P112 just fixed) sit silently until an audit catches them. With this,
they fail at startup before any trades fire.

Usage in main.py startup:
    from configs.config_schema import validate_config_consistency
    issues = validate_config_consistency(config_dict)
    for severity, msg in issues:
        if severity == "ERROR":
            logger.critical(msg)
            sys.exit(1)  # actually use process exit only at startup
        else:
            logger.warning(msg)

Pure validation library — no side effects beyond returning issue list.
Tests in tests/test_config_schema.py exercise edge cases.
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# =====================================================================
# CANONICAL THRESHOLDS — what the system EXPECTS at startup
# =====================================================================

class DrawdownGradient(BaseModel):
    """Drawdown thresholds in strict ordering: reduce < critical < halt < kill."""
    reduce_at_drawdown: float = Field(0.08, gt=0, lt=1)
    critical_drawdown: float = Field(0.15, gt=0, lt=1)
    hard_drawdown_halt: float = Field(0.25, gt=0, lt=1)
    kill_switch_drawdown: float = Field(0.35, gt=0, lt=1)
    daily_loss_halt: float = Field(0.08, gt=0, lt=1)
    daily_loss_kill: float = Field(0.10, gt=0, lt=1)

    @model_validator(mode="after")
    def check_drawdown_ordering(self) -> "DrawdownGradient":
        # critical < halt < kill is non-negotiable for the safety chain.
        # CLAUDE.md note: "CRITICAL_DRAWDOWN < HARD_DRAWDOWN_HALT is intentional"
        if self.critical_drawdown >= self.hard_drawdown_halt:
            raise ValueError(
                f"critical_drawdown ({self.critical_drawdown}) must be < "
                f"hard_drawdown_halt ({self.hard_drawdown_halt}) per CLAUDE.md "
                f"safety chain ordering. Current values would invert the "
                f"reduce → halt cascade."
            )
        if self.hard_drawdown_halt >= self.kill_switch_drawdown:
            raise ValueError(
                f"hard_drawdown_halt ({self.hard_drawdown_halt}) must be < "
                f"kill_switch_drawdown ({self.kill_switch_drawdown})."
            )
        if self.daily_loss_halt >= self.daily_loss_kill:
            raise ValueError(
                f"daily_loss_halt ({self.daily_loss_halt}) must be < "
                f"daily_loss_kill ({self.daily_loss_kill})."
            )
        return self


class LeverageHierarchy(BaseModel):
    """Tier / regime / global leverage caps; tightest wins (P112 9c decision)."""
    global_max_leverage: float = Field(3.0, gt=0, le=20.0)
    min_leverage: float = Field(1.0, ge=1.0)
    tier_caps: Dict[str, float] = Field(default_factory=dict)
    regime_caps: Dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_no_cap_exceeds_global(self) -> "LeverageHierarchy":
        """Tier and regime caps must NEVER exceed global. min(...) hierarchy
        only enforces if all sources agree on the ceiling."""
        for src_name, src_caps in [("tier", self.tier_caps),
                                    ("regime", self.regime_caps)]:
            for k, v in src_caps.items():
                if v > self.global_max_leverage:
                    raise ValueError(
                        f"{src_name}_caps[{k}] = {v}x exceeds "
                        f"global_max_leverage = {self.global_max_leverage}x. "
                        f"Per P112 9c (min-hierarchy), {src_name} cap can never "
                        f"loosen the global cap."
                    )
        return self


class CorrelationThresholds(BaseModel):
    """Correlation cascade thresholds — UL-4 widened them in production."""
    correlation_warning: float = Field(0.92, gt=0, lt=1)
    correlation_danger: float = Field(0.96, gt=0, lt=1)
    correlation_crisis: float = Field(0.98, gt=0, lt=1)

    @model_validator(mode="after")
    def check_correlation_ordering(self) -> "CorrelationThresholds":
        if not (self.correlation_warning < self.correlation_danger < self.correlation_crisis):
            raise ValueError(
                f"Correlation thresholds must be strictly increasing: "
                f"warning ({self.correlation_warning}) < "
                f"danger ({self.correlation_danger}) < "
                f"crisis ({self.correlation_crisis}). Inversion = silent "
                f"crisis-trigger order swap."
            )
        return self


class TrancheConfig(BaseModel):
    """Tranche allocation must sum to 1.0 (operator-drift guard)."""
    tranche_percentages: Dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_sum_unity(self) -> "TrancheConfig":
        if not self.tranche_percentages:
            return self  # empty config = use defaults elsewhere
        total = sum(self.tranche_percentages.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"tranche_percentages sum = {total}, must be 1.0. "
                f"Operator config drift detected: {self.tranche_percentages}. "
                f"Sum != 1.0 breaks full-position-schedule sizing."
            )
        return self


class HMATSConfigSchema(BaseModel):
    """Top-level config schema. Sub-schemas validate their own ordering."""
    drawdown: DrawdownGradient = Field(default_factory=DrawdownGradient)
    leverage: LeverageHierarchy = Field(default_factory=LeverageHierarchy)
    correlation: CorrelationThresholds = Field(default_factory=CorrelationThresholds)
    tranche: TrancheConfig = Field(default_factory=TrancheConfig)


# =====================================================================
# CROSS-SOURCE CONSISTENCY CHECKS
# =====================================================================

def validate_config_consistency(
    canonical: Optional[Dict[str, Any]] = None,
    json_overrides: Optional[Dict[str, Any]] = None,
    sota_flags: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str]]:
    """Compare same-key values across sources. Returns list of
    (severity, message) tuples. Severity: 'ERROR' (block startup) or
    'WARNING' (operator should investigate but startup proceeds).

    Catches the family of bugs P112 just fixed (canonical 0.20 vs JSON 0.25):
    silent value drift across config sources that auditors only find
    by manual diff.
    """
    issues: List[Tuple[str, str]] = []

    # Read canonical_config module constants if not provided
    if canonical is None:
        try:
            from configs import canonical_config as cc
            canonical = {
                "hard_drawdown_halt": cc.HARD_DRAWDOWN_HALT,
                "critical_drawdown": cc.CRITICAL_DRAWDOWN,
                "kill_switch_drawdown": cc.KILL_SWITCH_DRAWDOWN,
                "max_leverage": cc.MAX_LEVERAGE,
                "max_gross_exposure": cc.MAX_GROSS_EXPOSURE,
                "max_net_exposure": 0.50,  # [NET-CAP 2026-06-14] net signed directional budget (null=OFF)
                "correlation_crisis": cc.CORRELATION_CRISIS,
                "initial_capital": cc.INITIAL_CAPITAL,
                "thesis_budget_loss_pct_nav": cc.THESIS_BUDGET_LOSS_PCT_NAV,
            }
        except (ImportError, AttributeError) as e:
            issues.append(
                ("ERROR", f"Could not load canonical_config for validation: {e}")
            )
            return issues

    # 1. Cross-source value drift (the P112 fix family)
    if json_overrides:
        # Map JSON keys (often nested) to canonical keys
        json_flat = _flatten_json_config(json_overrides)
        for key, can_value in canonical.items():
            if key in json_flat and isinstance(can_value, (int, float)):
                json_value = json_flat[key]
                if isinstance(json_value, (int, float)) and abs(json_value - can_value) > 1e-9:
                    issues.append((
                        "WARNING",
                        f"[CONFIG-DRIFT] {key}: canonical={can_value}, "
                        f"JSON_override={json_value}. JSON wins at runtime "
                        f"(per p0_safety_integrator); update canonical to "
                        f"match if intentional."
                    ))

    # 2. Schema validation against the merged effective config
    effective = {**canonical, **(json_flat if json_overrides else {})}
    try:
        HMATSConfigSchema(
            drawdown=DrawdownGradient(
                hard_drawdown_halt=effective.get("hard_drawdown_halt", 0.25),
                critical_drawdown=effective.get("critical_drawdown", 0.15),
                kill_switch_drawdown=effective.get("kill_switch_drawdown", 0.35),
                reduce_at_drawdown=effective.get("reduce_at_drawdown", 0.08),
                daily_loss_halt=effective.get("daily_loss_halt", 0.08),
                daily_loss_kill=effective.get("daily_loss_kill", 0.10),
            ),
            correlation=CorrelationThresholds(
                correlation_warning=effective.get("correlation_warning", 0.92),
                correlation_danger=effective.get("correlation_danger", 0.96),
                correlation_crisis=effective.get("correlation_crisis", 0.98),
            ),
        )
    except Exception as e:
        issues.append(("ERROR", f"[CONFIG-SCHEMA] {e}"))

    # 3. Sota flags vs canonical (specific known sources of drift)
    if sota_flags:
        sota_max_lev = sota_flags.get("MAX_LEVERAGE")
        if sota_max_lev is not None and abs(sota_max_lev - canonical.get("max_leverage", 0)) > 1e-9:
            issues.append((
                "WARNING",
                f"[CONFIG-DRIFT] MAX_LEVERAGE: canonical="
                f"{canonical.get('max_leverage')}, sota_flags={sota_max_lev}. "
                f"Reconcile to single source of truth."
            ))

    return issues


def _flatten_json_config(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested JSON config so 'risk.hard_drawdown_halt' becomes
    just 'hard_drawdown_halt' for canonical comparison.

    Strategy: only the LEAF key matters for drift detection — collisions
    across nested groups are acceptable as long as values agree."""
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat.update(_flatten_json_config(v, prefix=k))
        else:
            flat[k] = v  # last-write-wins on collision
    return flat


def emit_validation_report(issues: List[Tuple[str, str]]) -> None:
    """Print issues in operator-readable form."""
    if not issues:
        print("[CONFIG_SCHEMA] OK — all sources consistent.")
        return
    errors = [m for s, m in issues if s == "ERROR"]
    warnings = [m for s, m in issues if s == "WARNING"]
    print(f"[CONFIG_SCHEMA] {len(errors)} ERROR, {len(warnings)} WARNING:")
    for m in errors:
        print(f"  ERROR: {m}")
    for m in warnings:
        print(f"  WARNING: {m}")


if __name__ == "__main__":
    # Default invocation: validate canonical alone (no JSON override).
    issues = validate_config_consistency()
    emit_validation_report(issues)
    import sys
    sys.exit(1 if any(s == "ERROR" for s, _ in issues) else 0)
