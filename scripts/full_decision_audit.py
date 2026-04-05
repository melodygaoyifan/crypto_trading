"""
Full module + strategy decision audit for the latest paper diagnostics.

Usage:
    python -X utf8 scripts/full_decision_audit.py
    python -X utf8 scripts/full_decision_audit.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import module_decision_audit as module_audit


ROOT = Path(__file__).resolve().parent.parent
DIAG_DIR = ROOT / "data" / "diagnostics"
PAPER_STDERR = ROOT / "logs" / "paper_run_stderr.log"
LIVE_EXPERIENCES_DIR = ROOT / "data" / "live_experiences"
SHADOW_LEDGER_DIR = ROOT / "data" / "shadow_ledger"
TRADE_ATTRIBUTION_PATH = ROOT / "data" / "trade_attribution.jsonl"
EXECUTION_QUALITY_DIR = ROOT / "logs" / "execution_quality"
ASSETS = module_audit.ASSETS
KNOWN_STRATEGIES = ("momentum", "mean_revert", "volume_breakout", "vrp")
DEFAULT_LOOKBACK_HOURS = 168
DEFAULT_LIMIT_PER_ASSET = 120


@dataclass
class StrategySample:
    asset: str
    tick_time: str
    strategy: str
    quant_dir: float
    quant_conf: float
    alpha_gate_passed: bool
    veto_active: bool
    veto_reason: str
    diag_path: str


@dataclass
class StrategyAudit:
    asset: str
    latest_tick_time: str
    latest_diag_path: str
    latest_strategy: str
    latest_quant_dir: float
    latest_quant_conf: float
    latest_alpha_gate_passed: bool
    latest_veto_reason: str
    recent_total_samples: int
    recent_winner_counts: dict[str, int]
    recent_avg_abs_dir: dict[str, float]
    recent_avg_conf: dict[str, float]
    dominant_strategy: str
    active_strategies: list[str]
    dormant_strategies: list[str]
    latest_strategy_select: dict[str, Any]


@dataclass
class RiskExecutionAudit:
    asset: str
    latest_tick_time: str
    latest_diag_path: str
    final_direction: float
    final_target_exposure: float
    alpha_gate_passed: bool
    final_veto_active: bool
    final_veto_reason: str
    blocked_by: str
    gate_states: dict[str, str]
    veto_chain_log: str
    execution_log: str


@dataclass
class FeedbackAudit:
    asset: str
    latest_tick_time: str
    latest_diag_path: str
    execute_intent_status: str
    pnl_engine_status: str
    shadow_ledger_status: str
    shadow_ledger_latest_type: str
    shadow_ledger_latest_timestamp: str
    shadow_ledger_latest_reason: str
    observation_status: str
    latest_observation_timestamp: str
    decision_status: str
    latest_decision_timestamp: str
    latest_decision_verdict: str
    latest_decision_state_dim: int
    outcome_status: str
    latest_outcome_timestamp: str
    latest_outcome_pnl_bps: float
    trade_attribution_status: str
    trade_attribution_latest_timestamp: str
    execution_quality_status: str
    execution_quality_latest_timestamp: str
    feedback_status: str


@dataclass
class FeedbackSummary:
    latest_live_experiences_file: str
    latest_live_experiences_counts: dict[str, int]
    total_experience_outcomes: int
    trade_attribution_records: int
    trade_attribution_latest_timestamp: str
    execution_quality_latest_file: str
    execution_quality_records: int
    execution_quality_latest_timestamp: str


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_iso8601(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_best_of_n(output: str) -> Optional[dict[str, Any]]:
    try:
        parsed = ast.literal_eval(output)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _parse_output_dict(output: str) -> dict[str, Any]:
    try:
        parsed = ast.literal_eval(output)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return {}
    return {}


def _diag_paths_for_asset(asset: str) -> list[Path]:
    return sorted(
        DIAG_DIR.glob(f"diag_{asset}_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _collect_strategy_samples(
    asset: str,
    *,
    lookback_hours: int,
    limit_per_asset: int,
) -> list[StrategySample]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    samples: list[StrategySample] = []

    for path in _diag_paths_for_asset(asset):
        if len(samples) >= limit_per_asset:
            break
        obj = _load_json(path)
        if not obj:
            continue
        tick_time = str(obj.get("tick_time", "") or "")
        tick_dt = _parse_iso8601(tick_time)
        if tick_dt is not None and tick_dt < cutoff:
            break
        best_of_n = obj.get("components", {}).get("best_of_n")
        if not isinstance(best_of_n, dict):
            continue
        parsed = _parse_best_of_n(str(best_of_n.get("output", "")))
        if not parsed:
            continue
        final_intent = obj.get("final_intent", {}) or {}
        samples.append(
            StrategySample(
                asset=asset,
                tick_time=tick_time,
                strategy=str(parsed.get("strategy", "unknown") or "unknown"),
                quant_dir=_safe_float(parsed.get("quant_dir"), 0.0),
                quant_conf=_safe_float(parsed.get("quant_conf"), 0.0),
                alpha_gate_passed=bool(final_intent.get("alpha_gate_passed", False)),
                veto_active=bool(final_intent.get("veto_active", False)),
                veto_reason=str(final_intent.get("veto_reason", "") or ""),
                diag_path=str(path.relative_to(ROOT)),
            )
        )
    samples.sort(key=lambda row: row.tick_time, reverse=True)
    return samples


_STRATEGY_SELECT_RE = re.compile(
    r"\[STRATEGY_SELECT\]\s+(?P<asset>[A-Z]+):\s+Best=(?P<best>[a-z_]+)\s+"
    r"\(sig=(?P<sig>[-+]?\d+(?:\.\d+)?),\s+str=(?P<strength>[-+]?\d+(?:\.\d+)?)\)\s+\|\s+"
    r"regime=(?P<regime>[A-Z_]+)\s+\|\s+others=\[(?P<others>[^\]]*)\]"
)


def _latest_strategy_select_lines() -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not PAPER_STDERR.exists():
        return latest
    try:
        lines = PAPER_STDERR.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return latest
    for line in reversed(lines):
        match = _STRATEGY_SELECT_RE.search(line)
        if not match:
            continue
        asset = match.group("asset")
        if asset in latest:
            continue
        others_raw = match.group("others").strip()
        others: dict[str, float] = {}
        if others_raw:
            for item in others_raw.split(","):
                key, _, value = item.strip().partition("=")
                if key:
                    others[key.strip()] = _safe_float(value.strip(), 0.0)
        latest[asset] = {
            "best": match.group("best"),
            "sig": _safe_float(match.group("sig"), 0.0),
            "strength": _safe_float(match.group("strength"), 0.0),
            "regime": match.group("regime"),
            "others": others,
        }
        if len(latest) == len(ASSETS):
            break
    return latest


def _latest_log_lines(pattern: re.Pattern[str]) -> dict[str, str]:
    latest: dict[str, str] = {}
    if not PAPER_STDERR.exists():
        return latest
    try:
        lines = PAPER_STDERR.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return latest
    for line in reversed(lines):
        match = pattern.search(line)
        if not match:
            continue
        asset = match.group("asset")
        if asset in latest:
            continue
        latest[asset] = match.group("body").strip()
        if len(latest) == len(ASSETS):
            break
    return latest


_VETO_CHAIN_RE = re.compile(r"\[VETO_CHAIN\]\s+(?P<asset>[A-Z]+):\s+(?P<body>.*)")
_EXECUTION_RE = re.compile(r"\[EXECUTION\]\s+(?P<asset>[A-Z]+):\s+(?P<body>.*)")


def _iter_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    except Exception:
        return []
    return rows


def _latest_jsonl_files(directory: Path, pattern: str) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)


def _pick_latest_by_timestamp(
    latest: dict[str, dict[str, Any]],
    key: str,
    row: dict[str, Any],
    timestamp_field: str = "timestamp",
) -> None:
    timestamp = str(row.get(timestamp_field, "") or "")
    ts_dt = _parse_iso8601(timestamp)
    if ts_dt is None:
        return
    previous = latest.get(key)
    if previous is None:
        latest[key] = row
        return
    prev_dt = _parse_iso8601(str(previous.get(timestamp_field, "") or ""))
    if prev_dt is None or ts_dt > prev_dt:
        latest[key] = row


def _status_from_freshness(timestamp: str, reference_time: str) -> str:
    ts_dt = _parse_iso8601(timestamp)
    ref_dt = _parse_iso8601(reference_time)
    if ts_dt is None:
        return "missing"
    if ref_dt is None:
        return "present"
    if ts_dt >= ref_dt - timedelta(minutes=10):
        return "fresh"
    return "history_only"


def _load_live_experience_state() -> tuple[
    FeedbackSummary,
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    files = _latest_jsonl_files(LIVE_EXPERIENCES_DIR, "experiences_*.jsonl")
    latest_file = files[0] if files else None
    latest_counts: Counter[str] = Counter()
    total_outcomes = 0
    latest_observations: dict[str, dict[str, Any]] = {}
    latest_decisions: dict[str, dict[str, Any]] = {}
    latest_outcomes: dict[str, dict[str, Any]] = {}

    for idx, path in enumerate(files):
        rows = _iter_jsonl_rows(path)
        for row in rows:
            row_type = str(row.get("type", "") or "")
            if idx == 0 and row_type:
                latest_counts[row_type] += 1
            if row_type == "outcome":
                total_outcomes += 1
            asset = str(row.get("asset", "") or "")
            if asset not in ASSETS:
                continue
            if row_type == "observation":
                _pick_latest_by_timestamp(latest_observations, asset, row)
            elif row_type == "decision":
                _pick_latest_by_timestamp(latest_decisions, asset, row)
            elif row_type == "outcome":
                _pick_latest_by_timestamp(latest_outcomes, asset, row)

    summary = FeedbackSummary(
        latest_live_experiences_file=latest_file.name if latest_file is not None else "",
        latest_live_experiences_counts=dict(latest_counts),
        total_experience_outcomes=total_outcomes,
        trade_attribution_records=0,
        trade_attribution_latest_timestamp="",
        execution_quality_latest_file="",
        execution_quality_records=0,
        execution_quality_latest_timestamp="",
    )
    return summary, latest_observations, latest_decisions, latest_outcomes


def _load_shadow_ledger_state() -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    files = _latest_jsonl_files(SHADOW_LEDGER_DIR, "ledger_*.jsonl")
    latest_by_asset: dict[str, dict[str, Any]] = {}
    latest_counts: Counter[str] = Counter()

    for idx, path in enumerate(files):
        rows = _iter_jsonl_rows(path)
        for row in rows:
            entry_type = str(row.get("entry_type", "") or "")
            if idx == 0 and entry_type:
                latest_counts[entry_type] += 1
            asset = str(row.get("asset", "") or "")
            if asset not in ASSETS:
                continue
            _pick_latest_by_timestamp(latest_by_asset, asset, row)
    return latest_by_asset, latest_counts


def _load_trade_attribution_state() -> tuple[int, str, dict[str, dict[str, Any]]]:
    latest_by_asset: dict[str, dict[str, Any]] = {}
    total_records = 0
    latest_timestamp = ""

    if TRADE_ATTRIBUTION_PATH.exists():
        for row in _iter_jsonl_rows(TRADE_ATTRIBUTION_PATH):
            total_records += 1
            asset = str(row.get("asset", "") or "")
            timestamp = str(row.get("exit_time", "") or row.get("entry_time", "") or "")
            if timestamp:
                latest_timestamp = max(
                    latest_timestamp,
                    timestamp,
                    key=lambda value: _parse_iso8601(value) or datetime.min.replace(tzinfo=timezone.utc),
                )
            if asset in ASSETS:
                row = dict(row)
                row["_audit_timestamp"] = timestamp
                _pick_latest_by_timestamp(latest_by_asset, asset, row, timestamp_field="_audit_timestamp")
    return total_records, latest_timestamp, latest_by_asset


def _load_execution_quality_state() -> tuple[str, int, str, dict[str, dict[str, Any]]]:
    files = _latest_jsonl_files(EXECUTION_QUALITY_DIR, "execution_quality_*.jsonl")
    latest_file = files[0] if files else None
    total_records = 0
    latest_timestamp = ""
    latest_by_asset: dict[str, dict[str, Any]] = {}

    for path in files:
        for row in _iter_jsonl_rows(path):
            total_records += 1
            asset = str(row.get("asset", "") or "")
            timestamp = str(row.get("timestamp", "") or "")
            if timestamp:
                latest_timestamp = max(
                    latest_timestamp,
                    timestamp,
                    key=lambda value: _parse_iso8601(value) or datetime.min.replace(tzinfo=timezone.utc),
                )
            if asset in ASSETS:
                _pick_latest_by_timestamp(latest_by_asset, asset, row)
    return latest_file.name if latest_file is not None else "", total_records, latest_timestamp, latest_by_asset


def _component_by_name(audit: module_audit.AssetAudit) -> dict[str, module_audit.ComponentStatus]:
    return {component.name: component for component in audit.components}


def _gate_status(audit: module_audit.AssetAudit, component_name: str) -> str:
    component = _component_by_name(audit).get(component_name)
    if component is None or not component.applicable:
        return "n/a"

    output = _parse_output_dict(component.output)
    if component_name == "confidence_gate":
        if not component.called:
            return "n/a"
        passed = output.get("passed", "N/A")
        if passed == "N/A":
            return "n/a"
        if passed is False:
            return "blocked"
        return "pass"
    if component_name == "opportunity_budget":
        return "blocked" if output.get("veto") else "pass"
    if component_name == "p0_force_flat":
        return "blocked" if output.get("active") else "pass"
    if component_name == "p0_short_block":
        allow_short = bool(output.get("allow_short", True))
        exp_mult = _safe_float(output.get("exp_mult"), 1.0)
        if not allow_short:
            return "blocked"
        if exp_mult < 0.999:
            return "clamped"
        return "pass"
    if component_name in {
        "gambler_entry",
        "v6_short_filter",
        "structure_gate",
        "risk_governor",
        "trade_gate",
        "weekend_manager",
        "sol_toxicity",
    }:
        return "blocked" if output.get("veto") else "pass"
    if component_name == "short_control":
        result = output.get("result")
        if isinstance(result, dict):
            if result.get("override"):
                return "override"
            if result.get("clamped"):
                return "clamped"
        return "pass"
    if component_name == "existence_fuse":
        return "blocked" if output.get("veto") else "pass"
    if component_name == "p0_safety_integrator":
        if not bool(output.get("allow", True)):
            return "blocked"
        result = str(output.get("result", "") or "")
        if result == "CLIPPED":
            return "clamped"
        return "pass"
    if component_name == "thesis_budget":
        if output.get("veto"):
            return "blocked"
        reentry_mult = _safe_float(output.get("reentry_mult"), 1.0)
        if reentry_mult < 0.999:
            return "scaled"
        return "pass"
    if component_name == "regime_power":
        return "applied" if component.consumed else "pass"
    if component_name == "profit_max_adapter":
        if output.get("veto"):
            return "blocked"
        return "applied" if component.consumed else "pass"
    if component_name == "final_size_cap":
        return "applied"
    if component_name == "execute_intent":
        return "executed" if output.get("executed") else "blocked_pre_execution"
    if component_name == "pnl_engine":
        return "recorded" if output.get("recorded") else "idle"
    return "applied" if component.consumed else "pass"


def _build_risk_execution_audit(
    audit: module_audit.AssetAudit,
    *,
    veto_chain_logs: dict[str, str],
    execution_logs: dict[str, str],
) -> RiskExecutionAudit:
    final_intent = audit.final_intent
    gate_states = {
        "confidence_gate": _gate_status(audit, "confidence_gate"),
        "opportunity_budget": _gate_status(audit, "opportunity_budget"),
        "p0_force_flat": _gate_status(audit, "p0_force_flat"),
        "p0_short_block": _gate_status(audit, "p0_short_block"),
        "gambler_entry": _gate_status(audit, "gambler_entry"),
        "v6_short_filter": _gate_status(audit, "v6_short_filter"),
        "short_control": _gate_status(audit, "short_control"),
        "structure_gate": _gate_status(audit, "structure_gate"),
        "risk_governor": _gate_status(audit, "risk_governor"),
        "trade_gate": _gate_status(audit, "trade_gate"),
        "existence_fuse": _gate_status(audit, "existence_fuse"),
        "weekend_manager": _gate_status(audit, "weekend_manager"),
        "sol_toxicity": _gate_status(audit, "sol_toxicity"),
        "p0_safety_integrator": _gate_status(audit, "p0_safety_integrator"),
        "thesis_budget": _gate_status(audit, "thesis_budget"),
        "alpha_gate": "pass" if bool(final_intent.get("alpha_gate_passed", False)) else "blocked",
        "regime_power": _gate_status(audit, "regime_power"),
        "profit_max_adapter": _gate_status(audit, "profit_max_adapter"),
        "final_size_cap": _gate_status(audit, "final_size_cap"),
        "execute_intent": _gate_status(audit, "execute_intent"),
        "pnl_engine": _gate_status(audit, "pnl_engine"),
    }
    veto_reason = str(final_intent.get("veto_reason", "") or "")
    blocked_by = veto_reason or "none"
    return RiskExecutionAudit(
        asset=audit.asset,
        latest_tick_time=audit.tick_time,
        latest_diag_path=audit.diag_path,
        final_direction=_safe_float(final_intent.get("direction"), 0.0),
        final_target_exposure=_safe_float(final_intent.get("target_exposure"), 0.0),
        alpha_gate_passed=bool(final_intent.get("alpha_gate_passed", False)),
        final_veto_active=bool(final_intent.get("veto_active", False)),
        final_veto_reason=veto_reason,
        blocked_by=blocked_by,
        gate_states=gate_states,
        veto_chain_log=veto_chain_logs.get(audit.asset, ""),
        execution_log=execution_logs.get(audit.asset, ""),
    )


def _build_strategy_audit(
    asset: str,
    *,
    lookback_hours: int,
    limit_per_asset: int,
    latest_select_lines: dict[str, dict[str, Any]],
) -> Optional[StrategyAudit]:
    samples = _collect_strategy_samples(
        asset,
        lookback_hours=lookback_hours,
        limit_per_asset=limit_per_asset,
    )
    if not samples:
        return None

    latest = samples[0]
    counts = Counter(sample.strategy for sample in samples)
    abs_dir_sums: defaultdict[str, float] = defaultdict(float)
    conf_sums: defaultdict[str, float] = defaultdict(float)
    strat_counts: defaultdict[str, int] = defaultdict(int)
    for sample in samples:
        abs_dir_sums[sample.strategy] += abs(sample.quant_dir)
        conf_sums[sample.strategy] += sample.quant_conf
        strat_counts[sample.strategy] += 1

    avg_abs_dir = {
        strategy: abs_dir_sums[strategy] / strat_counts[strategy]
        for strategy in sorted(strat_counts)
    }
    avg_conf = {
        strategy: conf_sums[strategy] / strat_counts[strategy]
        for strategy in sorted(strat_counts)
    }
    active = sorted(strategy for strategy, count in counts.items() if count > 0)
    dormant = sorted(strategy for strategy in KNOWN_STRATEGIES if counts.get(strategy, 0) == 0)
    dominant = counts.most_common(1)[0][0] if counts else "n/a"

    return StrategyAudit(
        asset=asset,
        latest_tick_time=latest.tick_time,
        latest_diag_path=latest.diag_path,
        latest_strategy=latest.strategy,
        latest_quant_dir=latest.quant_dir,
        latest_quant_conf=latest.quant_conf,
        latest_alpha_gate_passed=latest.alpha_gate_passed,
        latest_veto_reason=latest.veto_reason,
        recent_total_samples=len(samples),
        recent_winner_counts={strategy: counts.get(strategy, 0) for strategy in KNOWN_STRATEGIES},
        recent_avg_abs_dir=avg_abs_dir,
        recent_avg_conf=avg_conf,
        dominant_strategy=dominant,
        active_strategies=active,
        dormant_strategies=dormant,
        latest_strategy_select=latest_select_lines.get(asset, {}),
    )


def _build_feedback_audit(
    audit: module_audit.AssetAudit,
    *,
    latest_observations: dict[str, dict[str, Any]],
    latest_decisions: dict[str, dict[str, Any]],
    latest_outcomes: dict[str, dict[str, Any]],
    latest_shadow_rows: dict[str, dict[str, Any]],
    latest_trade_rows: dict[str, dict[str, Any]],
    latest_exec_rows: dict[str, dict[str, Any]],
) -> FeedbackAudit:
    execute_status = _gate_status(audit, "execute_intent")
    pnl_engine_status = _gate_status(audit, "pnl_engine")
    latest_tick_time = audit.tick_time

    observation_row = latest_observations.get(audit.asset, {})
    decision_row = latest_decisions.get(audit.asset, {})
    outcome_row = latest_outcomes.get(audit.asset, {})
    shadow_row = latest_shadow_rows.get(audit.asset, {})
    trade_row = latest_trade_rows.get(audit.asset, {})
    exec_row = latest_exec_rows.get(audit.asset, {})

    observation_ts = str(observation_row.get("timestamp", "") or "")
    decision_ts = str(decision_row.get("timestamp", "") or "")
    outcome_ts = str(outcome_row.get("timestamp", "") or "")
    shadow_ts = str(shadow_row.get("timestamp", "") or "")
    trade_ts = str(trade_row.get("_audit_timestamp", "") or "")
    exec_ts = str(exec_row.get("timestamp", "") or "")

    observation_status = _status_from_freshness(observation_ts, latest_tick_time)
    decision_status = _status_from_freshness(decision_ts, latest_tick_time)
    shadow_status = _status_from_freshness(shadow_ts, latest_tick_time)
    trade_status = _status_from_freshness(trade_ts, latest_tick_time)
    exec_status = _status_from_freshness(exec_ts, latest_tick_time)

    if decision_status in {"missing", "history_only"} and observation_status == "fresh":
        decision_status = "pending_flush"
    elif (
        decision_status in {"missing", "history_only"}
        and shadow_status == "fresh"
        and execute_status != "executed"
    ):
        decision_status = "pending_flush"
    if outcome_ts:
        outcome_status = _status_from_freshness(outcome_ts, latest_tick_time)
    elif execute_status != "executed":
        outcome_status = "pending_no_fill"
    else:
        outcome_status = "waiting_for_close"

    if (
        execute_status != "executed"
        and decision_status in {"fresh", "present", "pending_flush"}
        and shadow_status in {"fresh", "present"}
    ):
        feedback_status = "decision_recorded_no_fill_yet"
    elif execute_status == "executed" and outcome_status == "waiting_for_close":
        feedback_status = "fill_recorded_waiting_for_close"
    elif outcome_status == "fresh":
        feedback_status = "close_feedback_recorded"
    else:
        feedback_status = "needs_attention"

    shadow_reason = ""
    if isinstance(shadow_row.get("data"), dict):
        shadow_reason = str(
            shadow_row["data"].get("rejection_reason")
            or shadow_row["data"].get("reason")
            or ""
        )

    decision_state = decision_row.get("state")
    state_dim = len(decision_state) if isinstance(decision_state, list) else 0

    return FeedbackAudit(
        asset=audit.asset,
        latest_tick_time=latest_tick_time,
        latest_diag_path=audit.diag_path,
        execute_intent_status=execute_status,
        pnl_engine_status=pnl_engine_status,
        shadow_ledger_status=shadow_status,
        shadow_ledger_latest_type=str(shadow_row.get("entry_type", "") or ""),
        shadow_ledger_latest_timestamp=shadow_ts,
        shadow_ledger_latest_reason=shadow_reason,
        observation_status=observation_status,
        latest_observation_timestamp=observation_ts,
        decision_status=decision_status,
        latest_decision_timestamp=decision_ts,
        latest_decision_verdict=str(decision_row.get("verdict", "") or ""),
        latest_decision_state_dim=state_dim,
        outcome_status=outcome_status,
        latest_outcome_timestamp=outcome_ts,
        latest_outcome_pnl_bps=_safe_float(outcome_row.get("pnl_bps"), 0.0),
        trade_attribution_status=trade_status,
        trade_attribution_latest_timestamp=trade_ts,
        execution_quality_status=exec_status,
        execution_quality_latest_timestamp=exec_ts,
        feedback_status=feedback_status,
    )


def short_list(items: list[str], limit: int = 10) -> str:
    return module_audit.short_list(items, limit=limit)


def render_strategy_report(audit: StrategyAudit) -> list[str]:
    select = audit.latest_strategy_select or {}
    others = select.get("others", {}) if isinstance(select, dict) else {}
    others_rendered = ", ".join(f"{name}={value:.2f}" for name, value in sorted(others.items())) or "-"
    counts_rendered = ", ".join(
        f"{name}={audit.recent_winner_counts.get(name, 0)}" for name in KNOWN_STRATEGIES
    )
    avg_conf_rendered = ", ".join(
        f"{name}={audit.recent_avg_conf[name]:.3f}" for name in sorted(audit.recent_avg_conf)
    ) or "-"
    avg_abs_dir_rendered = ", ".join(
        f"{name}={audit.recent_avg_abs_dir[name]:.3f}" for name in sorted(audit.recent_avg_abs_dir)
    ) or "-"
    return [
        f"[{audit.asset}] {audit.latest_tick_time}",
        f"  diag: {audit.latest_diag_path}",
        (
            "  latest: "
            f"strategy={audit.latest_strategy} "
            f"dir={audit.latest_quant_dir:+.3f} "
            f"conf={audit.latest_quant_conf:.3f} "
            f"alpha_pass={audit.latest_alpha_gate_passed} "
            f"veto_reason={audit.latest_veto_reason or '-'}"
        ),
        (
            "  latest_select: "
            f"best={select.get('best', audit.latest_strategy)} "
            f"sig={_safe_float(select.get('sig'), audit.latest_quant_dir):+.3f} "
            f"str={_safe_float(select.get('strength'), audit.latest_quant_conf):.3f} "
            f"regime={select.get('regime', 'n/a')}"
        ),
        f"  alternatives: {others_rendered}",
        f"  recent_counts({audit.recent_total_samples}): {counts_rendered}",
        f"  avg_abs_dir: {avg_abs_dir_rendered}",
        f"  avg_conf: {avg_conf_rendered}",
        f"  dominant: {audit.dominant_strategy}",
        f"  active: {short_list(audit.active_strategies, limit=8)}",
        f"  dormant: {short_list(audit.dormant_strategies, limit=8)}",
    ]


def render_risk_execution_report(audit: RiskExecutionAudit) -> list[str]:
    chain_order = [
        "confidence_gate",
        "opportunity_budget",
        "p0_force_flat",
        "p0_short_block",
        "gambler_entry",
        "v6_short_filter",
        "short_control",
        "structure_gate",
        "risk_governor",
        "trade_gate",
        "existence_fuse",
        "weekend_manager",
        "sol_toxicity",
        "p0_safety_integrator",
        "thesis_budget",
        "alpha_gate",
        "regime_power",
        "profit_max_adapter",
        "final_size_cap",
        "execute_intent",
        "pnl_engine",
    ]
    chain_rendered = " -> ".join(
        f"{name}={audit.gate_states.get(name, 'n/a')}" for name in chain_order
    )
    return [
        f"[{audit.asset}] {audit.latest_tick_time}",
        f"  diag: {audit.latest_diag_path}",
        (
            "  final: "
            f"dir={audit.final_direction:+.3f} "
            f"exp={audit.final_target_exposure:.1%} "
            f"alpha_pass={audit.alpha_gate_passed} "
            f"veto={audit.final_veto_active} "
            f"blocked_by={audit.blocked_by or '-'}"
        ),
        f"  gate_chain: {chain_rendered}",
        f"  execution_log: {audit.execution_log or '-'}",
        f"  veto_chain_log: {audit.veto_chain_log or '-'}",
    ]


def render_feedback_report(audit: FeedbackAudit) -> list[str]:
    return [
        f"[{audit.asset}] {audit.latest_tick_time}",
        f"  diag: {audit.latest_diag_path}",
        (
            "  execution_feedback: "
            f"execute_intent={audit.execute_intent_status} "
            f"pnl_engine={audit.pnl_engine_status} "
            f"feedback_status={audit.feedback_status}"
        ),
        (
            "  shadow_ledger: "
            f"status={audit.shadow_ledger_status} "
            f"type={audit.shadow_ledger_latest_type or '-'} "
            f"ts={audit.shadow_ledger_latest_timestamp or '-'}"
        ),
        f"  shadow_reason: {audit.shadow_ledger_latest_reason or '-'}",
        (
            "  observation: "
            f"status={audit.observation_status} "
            f"ts={audit.latest_observation_timestamp or '-'}"
        ),
        (
            "  live_experience: "
            f"decision={audit.decision_status} "
            f"ts={audit.latest_decision_timestamp or '-'} "
            f"verdict={audit.latest_decision_verdict or '-'} "
            f"state_dim={audit.latest_decision_state_dim}"
        ),
        (
            "  outcome: "
            f"status={audit.outcome_status} "
            f"ts={audit.latest_outcome_timestamp or '-'} "
            f"pnl_bps={audit.latest_outcome_pnl_bps:+.1f}"
        ),
        (
            "  attribution: "
            f"status={audit.trade_attribution_status} "
            f"latest={audit.trade_attribution_latest_timestamp or '-'}"
        ),
        (
            "  execution_quality: "
            f"status={audit.execution_quality_status} "
            f"latest={audit.execution_quality_latest_timestamp or '-'}"
        ),
    ]


def build_payload(lookback_hours: int, limit_per_asset: int) -> dict[str, Any]:
    module_audits = [
        audit
        for asset in ASSETS
        if (audit := module_audit.load_asset_audit(asset)) is not None
    ]
    latest_select_lines = _latest_strategy_select_lines()
    veto_chain_logs = _latest_log_lines(_VETO_CHAIN_RE)
    execution_logs = _latest_log_lines(_EXECUTION_RE)
    strategy_audits = [
        audit
        for asset in ASSETS
        if (
            audit := _build_strategy_audit(
                asset,
                lookback_hours=lookback_hours,
                limit_per_asset=limit_per_asset,
                latest_select_lines=latest_select_lines,
            )
        ) is not None
    ]
    risk_execution_audits = [
        _build_risk_execution_audit(
            audit,
            veto_chain_logs=veto_chain_logs,
            execution_logs=execution_logs,
        )
        for audit in module_audits
    ]
    feedback_summary, latest_observations, latest_decisions, latest_outcomes = _load_live_experience_state()
    latest_shadow_rows, latest_shadow_counts = _load_shadow_ledger_state()
    trade_records, trade_latest_ts, latest_trade_rows = _load_trade_attribution_state()
    exec_latest_file, exec_records, exec_latest_ts, latest_exec_rows = _load_execution_quality_state()
    feedback_summary.trade_attribution_records = trade_records
    feedback_summary.trade_attribution_latest_timestamp = trade_latest_ts
    feedback_summary.execution_quality_latest_file = exec_latest_file
    feedback_summary.execution_quality_records = exec_records
    feedback_summary.execution_quality_latest_timestamp = exec_latest_ts
    feedback_audits = [
        _build_feedback_audit(
            audit,
            latest_observations=latest_observations,
            latest_decisions=latest_decisions,
            latest_outcomes=latest_outcomes,
            latest_shadow_rows=latest_shadow_rows,
            latest_trade_rows=latest_trade_rows,
            latest_exec_rows=latest_exec_rows,
        )
        for audit in module_audits
    ]
    return {
        "lookback_hours": lookback_hours,
        "limit_per_asset": limit_per_asset,
        "module": {
            "assets": [
                {
                    "asset": audit.asset,
                    "diag_path": audit.diag_path,
                    "tick_time": audit.tick_time,
                    "final_intent": audit.final_intent,
                    "summary": audit.summary,
                    "components": [asdict(component) for component in audit.components],
                }
                for audit in module_audits
            ],
            "aggregate": module_audit.aggregate(module_audits),
        },
        "strategy": {
            "assets": [asdict(audit) for audit in strategy_audits],
        },
        "risk_execution": {
            "assets": [asdict(audit) for audit in risk_execution_audits],
        },
        "feedback": {
            "summary": asdict(feedback_summary),
            "latest_shadow_ledger_counts": dict(latest_shadow_counts),
            "assets": [asdict(audit) for audit in feedback_audits],
        },
    }


def render_text(payload: dict[str, Any]) -> str:
    module_assets = []
    for raw in payload["module"]["assets"]:
        module_assets.append(
            module_audit.AssetAudit(
                asset=raw["asset"],
                diag_path=raw["diag_path"],
                tick_time=raw["tick_time"],
                final_intent=raw["final_intent"],
                summary=raw["summary"],
                components=[
                    module_audit.ComponentStatus(**component)
                    for component in raw["components"]
                ],
            )
        )

    module_text = module_audit.render_text(module_assets, payload["module"]["aggregate"])

    strategy_lines = [
        "Strategy audit",
        "",
        (
            "Window: "
            f"{payload['lookback_hours']}h, "
            f"max {payload['limit_per_asset']} diagnostics per asset"
        ),
    ]

    for raw in payload["strategy"]["assets"]:
        strategy_lines.extend(["", *render_strategy_report(StrategyAudit(**raw))])

    risk_lines = ["Risk / veto / execution audit"]
    for raw in payload["risk_execution"]["assets"]:
        risk_lines.extend(["", *render_risk_execution_report(RiskExecutionAudit(**raw))])

    feedback_summary = FeedbackSummary(**payload["feedback"]["summary"])
    feedback_lines = [
        "Feedback / learning / attribution audit",
        "",
        (
            "artifacts: "
            f"live_experiences={feedback_summary.latest_live_experiences_file or '-'} "
            f"counts={feedback_summary.latest_live_experiences_counts or {}} "
            f"total_outcomes={feedback_summary.total_experience_outcomes}"
        ),
        (
            "  "
            f"shadow_ledger_latest_counts={payload['feedback']['latest_shadow_ledger_counts']}"
        ),
        (
            "  "
            f"trade_attribution_records={feedback_summary.trade_attribution_records} "
            f"latest={feedback_summary.trade_attribution_latest_timestamp or '-'}"
        ),
        (
            "  "
            f"execution_quality_file={feedback_summary.execution_quality_latest_file or '-'} "
            f"records={feedback_summary.execution_quality_records} "
            f"latest={feedback_summary.execution_quality_latest_timestamp or '-'}"
        ),
    ]
    for raw in payload["feedback"]["assets"]:
        feedback_lines.extend(["", *render_feedback_report(FeedbackAudit(**raw))])

    return (
        module_text
        + "\n\n"
        + "\n".join(strategy_lines)
        + "\n\n"
        + "\n".join(risk_lines)
        + "\n\n"
        + "\n".join(feedback_lines)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Full module + strategy decision audit.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=DEFAULT_LOOKBACK_HOURS,
        help=f"How many recent hours of diagnostics to scan for strategy stats (default: {DEFAULT_LOOKBACK_HOURS}).",
    )
    parser.add_argument(
        "--limit-per-asset",
        type=int,
        default=DEFAULT_LIMIT_PER_ASSET,
        help=f"Maximum diagnostics to scan per asset (default: {DEFAULT_LIMIT_PER_ASSET}).",
    )
    args = parser.parse_args()

    payload = build_payload(args.lookback_hours, args.limit_per_asset)
    if not payload["module"]["assets"]:
        print("No diagnostics found under data/diagnostics.", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
