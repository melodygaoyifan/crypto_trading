from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_execution_advisory_config_snapshot(
    *,
    guard_enabled: bool,
    min_session_fills: int,
    min_asset_fills: int,
    require_verified_fill_telemetry: bool,
    max_latest_slippage_bps: float,
    max_toxicity_rate: float,
    min_side_closed_trades: int,
    min_side_total_realized_pnl_usd: float,
    min_side_avg_realized_pnl_bps: float,
) -> Dict[str, Any]:
    return {
        "guard_enabled": bool(guard_enabled),
        "min_session_fills": int(min_session_fills or 0),
        "min_asset_fills": int(min_asset_fills or 0),
        "require_verified_fill_telemetry": bool(require_verified_fill_telemetry),
        "max_latest_slippage_bps": float(max_latest_slippage_bps or 0.0),
        "max_toxicity_rate": float(max_toxicity_rate or 0.0),
        "min_side_closed_trades": int(min_side_closed_trades or 0),
        "min_side_total_realized_pnl_usd": float(min_side_total_realized_pnl_usd or 0.0),
        "min_side_avg_realized_pnl_bps": float(min_side_avg_realized_pnl_bps or 0.0),
    }


def evaluate_execution_advisory_guard(
    *,
    guard_cfg: Dict[str, Any],
    asset: str,
    trade_side: str,
    fill_summary: Dict[str, Any],
    fill_metrics: Dict[str, Any],
    side_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    counts_by_asset = dict(fill_summary.get("counts_by_asset", {}) or {})
    session_fills = int(fill_summary.get("count", 0) or 0)
    asset_fills = int(counts_by_asset.get(asset, 0) or 0)

    latest_slippage_bps = float(fill_metrics.get("latest_slippage_bps", 0.0) or 0.0)
    toxicity_rate = float(fill_metrics.get("toxicity_rate", 0.0) or 0.0)
    telemetry_fill_count = int(fill_metrics.get("fill_count", 0) or 0)
    telemetry_verified = bool(fill_metrics.get("telemetry_verified", False))

    side_stats = dict(side_stats or {})
    side_closed_trades = int(side_stats.get("count", 0) or 0)
    side_total_pnl_usd = float(side_stats.get("total_pnl_usd", 0.0) or 0.0)
    side_avg_pnl_bps_raw = side_stats.get("avg_pnl_bps")
    side_avg_pnl_bps = (
        float(side_avg_pnl_bps_raw) if side_avg_pnl_bps_raw is not None else None
    )
    pnl_sample_ready = side_closed_trades >= int(
        guard_cfg.get("min_side_closed_trades", 0) or 0
    )

    if not bool(guard_cfg.get("guard_enabled", False)):
        return {
            "enabled": False,
            "open": True,
            "reason": "GUARD_DISABLED",
            "trade_side": trade_side,
            "session_fills": session_fills,
            "asset_fills": asset_fills,
            "fill_telemetry_count": telemetry_fill_count,
            "telemetry_verified": telemetry_verified,
            "latest_slippage_bps": latest_slippage_bps,
            "toxicity_rate": toxicity_rate,
            "side_closed_trades": side_closed_trades,
            "side_total_realized_pnl_usd": side_total_pnl_usd,
            "side_avg_realized_pnl_bps": side_avg_pnl_bps,
            "side_pnl_sample_ready": pnl_sample_ready,
            "fills_ok": True,
            "evidence_quality_open": True,
            "evidence_quality_reason": "GUARD_DISABLED",
            "slippage_ok": True,
            "toxicity_ok": True,
            "pnl_ok": True,
        }

    reasons: List[str] = []
    fills_ok = session_fills >= int(guard_cfg.get("min_session_fills", 0) or 0)
    asset_fills_ok = asset_fills >= int(guard_cfg.get("min_asset_fills", 0) or 0)
    require_verified_fill_telemetry = bool(
        guard_cfg.get("require_verified_fill_telemetry", False)
    )
    evidence_quality_ok = True
    evidence_quality_reason = "READY"
    if not asset_fills_ok:
        evidence_quality_ok = False
        evidence_quality_reason = (
            f"PENDING_ASSET_FILLS_{asset_fills}/{int(guard_cfg.get('min_asset_fills', 0) or 0)}"
        )
    elif require_verified_fill_telemetry:
        evidence_quality_ok = telemetry_verified and telemetry_fill_count >= asset_fills
        evidence_quality_reason = (
            "VERIFIED"
            if evidence_quality_ok
            else f"UNVERIFIED_FILL_TELEMETRY_{telemetry_fill_count}/{asset_fills}"
        )

    slippage_ok = asset_fills_ok and latest_slippage_bps <= float(
        guard_cfg.get("max_latest_slippage_bps", 0.0) or 0.0
    )
    toxicity_ok = asset_fills_ok and toxicity_rate <= float(
        guard_cfg.get("max_toxicity_rate", 0.0) or 0.0
    )
    pnl_ok = True

    if not fills_ok:
        reasons.append(
            f"SESSION_FILLS_{session_fills}/{int(guard_cfg.get('min_session_fills', 0) or 0)}"
        )
    if not asset_fills_ok:
        reasons.append(
            f"ASSET_FILLS_{asset_fills}/{int(guard_cfg.get('min_asset_fills', 0) or 0)}"
        )
    if asset_fills_ok and not evidence_quality_ok:
        reasons.append(f"EVIDENCE_QUALITY_{evidence_quality_reason}")
    if asset_fills_ok and not slippage_ok:
        reasons.append(
            f"SLIPPAGE_{latest_slippage_bps:.1f}>{float(guard_cfg.get('max_latest_slippage_bps', 0.0) or 0.0):.1f}"
        )
    if asset_fills_ok and not toxicity_ok:
        reasons.append(
            f"TOXICITY_{toxicity_rate:.2f}>{float(guard_cfg.get('max_toxicity_rate', 0.0) or 0.0):.2f}"
        )
    if pnl_sample_ready:
        if side_total_pnl_usd < float(
            guard_cfg.get("min_side_total_realized_pnl_usd", 0.0) or 0.0
        ):
            pnl_ok = False
            reasons.append(
                f"SIDE_PNL_{side_total_pnl_usd:.2f}<{float(guard_cfg.get('min_side_total_realized_pnl_usd', 0.0) or 0.0):.2f}"
            )
        if (
            side_avg_pnl_bps is not None
            and side_avg_pnl_bps
            < float(guard_cfg.get("min_side_avg_realized_pnl_bps", 0.0) or 0.0)
        ):
            pnl_ok = False
            reasons.append(
                f"SIDE_AVG_BPS_{side_avg_pnl_bps:.1f}<{float(guard_cfg.get('min_side_avg_realized_pnl_bps', 0.0) or 0.0):.1f}"
            )

    return {
        "enabled": True,
        "open": len(reasons) == 0,
        "reason": "READY" if not reasons else " | ".join(reasons),
        "trade_side": trade_side,
        "session_fills": session_fills,
        "asset_fills": asset_fills,
        "fill_telemetry_count": telemetry_fill_count,
        "telemetry_verified": telemetry_verified,
        "latest_slippage_bps": latest_slippage_bps,
        "toxicity_rate": toxicity_rate,
        "side_closed_trades": side_closed_trades,
        "side_total_realized_pnl_usd": side_total_pnl_usd,
        "side_avg_realized_pnl_bps": side_avg_pnl_bps,
        "side_pnl_sample_ready": pnl_sample_ready,
        "fills_ok": fills_ok and asset_fills_ok,
        "evidence_quality_open": evidence_quality_ok,
        "evidence_quality_reason": evidence_quality_reason,
        "slippage_ok": slippage_ok,
        "toxicity_ok": toxicity_ok,
        "pnl_ok": pnl_ok,
    }
