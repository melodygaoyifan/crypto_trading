from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable


def build_exit_alpha_review_summary(tracker: Any) -> Dict[str, Any]:
    default = {
        "available": False,
        "tracked_trades": 0,
        "avg_retention_pct": 0.0,
        "avg_peak_unrealized_bps": 0.0,
        "avg_realized_bps": 0.0,
        "avg_peak_to_exit_bps": 0.0,
        "total_pnl_usd": 0.0,
        "by_trigger": {},
        "by_asset": {},
        "recommendations": {
            "sample_ready": False,
            "tracked_trades": 0,
            "verdict": "INSUFFICIENT_SAMPLE",
            "recommendations": [],
        },
    }
    if tracker is None:
        return default
    try:
        summary_fn = getattr(tracker, "summary_dict", None)
        rec_fn = getattr(tracker, "recommendation_dict", None)
        if callable(summary_fn):
            summary = dict(summary_fn() or {})
            if callable(rec_fn):
                summary["recommendations"] = dict(rec_fn() or {})
            summary["available"] = True
            return {**default, **summary}
    except Exception:
        pass
    return default


def build_step15_asset_status(
    *,
    assets: Iterable[str],
    asset_data: Dict[str, Dict[str, Any]],
    runtime_state: Dict[str, Dict[str, Any]],
    active_positions: Dict[str, Dict[str, Any]],
    asset_authorities: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    status = {}
    for asset in assets:
        asset_snapshot = dict(asset_data.get(asset, {}) or {})
        asset_runtime = dict(runtime_state.get(asset, {}) or {})
        asset_authority = dict(asset_authorities.get(asset, {}) or {})
        status[asset] = {
            "session_fill_count": int(asset_runtime.get("session_fill_count", 0) or 0),
            "has_position": asset in active_positions,
            "alpha_gate_passed": bool(asset_snapshot.get("alpha_gate_passed", False)),
            "alpha_gate_decision": asset_snapshot.get("alpha_gate_decision", ""),
            "actionable": bool(asset_snapshot.get("actionable", False)),
            "veto_reason": asset_snapshot.get("veto_reason", ""),
            "execution_advisory_guard_open": bool(
                asset_runtime.get("execution_advisory_guard_open", False)
            ),
            "execution_advisory_guard_reason": str(
                asset_runtime.get("execution_advisory_guard_reason", "")
            ),
            "execution_advisory_guard_side": str(
                asset_runtime.get("execution_advisory_guard_side", "flat")
            ),
            "execution_advisory_evidence_quality_open": bool(
                asset_runtime.get("execution_advisory_evidence_quality_open", False)
            ),
            "execution_advisory_evidence_quality_reason": str(
                asset_runtime.get("execution_advisory_evidence_quality_reason", "")
            ),
            "authority_states": dict(asset_authority),
            "btc_short_shadow_relax_candidate": bool(
                asset_runtime.get("btc_short_shadow_relax_candidate", False)
            ),
            "btc_short_shadow_relax_applied": bool(
                asset_runtime.get("btc_short_shadow_relax_applied", False)
            ),
            "btc_short_shadow_relax_reason": str(
                asset_runtime.get("btc_short_shadow_relax_reason", "")
            ),
        }
    return status


def inject_asset_authority_states(
    asset_data: Dict[str, Dict[str, Any]],
    authority_states: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    enriched = {
        asset: dict(data) if isinstance(data, dict) else {}
        for asset, data in (asset_data or {}).items()
    }
    by_asset = dict((authority_states or {}).get("by_asset", {}) or {})
    for asset, data in enriched.items():
        asset_authority = dict(by_asset.get(asset, {}) or {})
        data["authority_states"] = asset_authority
        data["composite_toxicity_authority_state"] = asset_authority.get(
            "composite_toxicity", "DISABLED"
        )
        data["learned_exec_authority_state"] = asset_authority.get(
            "learned_execution_policy", "DISABLED"
        )
        data["g6_authority_state"] = asset_authority.get("g6", "DISABLED")
        data["aggressive_allocator_authority_state"] = asset_authority.get(
            "aggressive_allocator", "DISABLED"
        )
        data["lead_lag_authority_state"] = asset_authority.get(
            "lead_lag", "DISABLED"
        )
        data["model_alpha_authority_state"] = asset_authority.get(
            "model_alpha", "DISABLED"
        )
        data["drl_authority_state"] = asset_authority.get("drl", "DISABLED")
    return enriched
