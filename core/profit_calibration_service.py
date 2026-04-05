from __future__ import annotations

from typing import Any, Dict


def build_profit_calibration_shadow(
    *,
    realized_pnl_summary: Dict[str, Any],
    enabled: bool,
    min_closed_trades_per_side: int,
    tighten_long_avg_bps_below: float,
    tighten_long_total_pnl_usd_below: float,
    keep_long_avg_bps_above: float,
    relax_short_avg_bps_above: float,
    relax_short_total_pnl_usd_above: float,
) -> Dict[str, Any]:
    by_side = dict((realized_pnl_summary or {}).get("by_side", {}) or {})
    min_count = int(min_closed_trades_per_side or 0)

    def _side_payload(side: str) -> Dict[str, Any]:
        stats = dict(by_side.get(side, {}) or {})
        count = int(stats.get("count", 0) or 0)
        total_pnl_usd = float(stats.get("total_pnl_usd", 0.0) or 0.0)
        avg_pnl_bps_raw = stats.get("avg_pnl_bps")
        avg_pnl_bps = float(avg_pnl_bps_raw) if avg_pnl_bps_raw is not None else None
        recommendation = "INSUFFICIENT_SAMPLE"
        reason = f"count={count}<{min_count}"

        if count >= min_count:
            if side == "long":
                if (
                    total_pnl_usd <= float(tighten_long_total_pnl_usd_below or 0.0)
                    or (
                        avg_pnl_bps is not None
                        and avg_pnl_bps <= float(tighten_long_avg_bps_below or 0.0)
                    )
                ):
                    recommendation = "TIGHTEN_QUIET_LONG"
                    reason = (
                        f"long pnl weak: total={total_pnl_usd:+.2f} "
                        f"avg_bps={(avg_pnl_bps if avg_pnl_bps is not None else 0.0):+.1f}"
                    )
                elif avg_pnl_bps is not None and avg_pnl_bps >= float(
                    keep_long_avg_bps_above or 0.0
                ):
                    recommendation = "KEEP_LONG_PATH_OPEN"
                    reason = f"long pnl constructive: avg_bps={avg_pnl_bps:+.1f}"
                else:
                    recommendation = "HOLD"
                    reason = (
                        f"long mixed: total={total_pnl_usd:+.2f} "
                        f"avg_bps={(avg_pnl_bps if avg_pnl_bps is not None else 0.0):+.1f}"
                    )
            else:
                if (
                    total_pnl_usd >= float(relax_short_total_pnl_usd_above or 0.0)
                    and avg_pnl_bps is not None
                    and avg_pnl_bps >= float(relax_short_avg_bps_above or 0.0)
                ):
                    recommendation = "RELAX_SHORT_STRUCTURE_CANDIDATE"
                    reason = (
                        f"short pnl strong: total={total_pnl_usd:+.2f} "
                        f"avg_bps={avg_pnl_bps:+.1f}"
                    )
                else:
                    recommendation = "HOLD"
                    reason = (
                        f"short mixed: total={total_pnl_usd:+.2f} "
                        f"avg_bps={(avg_pnl_bps if avg_pnl_bps is not None else 0.0):+.1f}"
                    )

        return {
            "count": count,
            "total_pnl_usd": total_pnl_usd,
            "avg_pnl_bps": avg_pnl_bps,
            "recommendation": recommendation,
            "reason": reason,
        }

    return {
        "enabled": bool(enabled),
        "mode": "SHADOW",
        "min_closed_trades_per_side": min_count,
        "by_side": {
            "long": _side_payload("long"),
            "short": _side_payload("short"),
        },
    }


def build_profit_calibration_live_adjustments(
    *,
    shadow_summary: Dict[str, Any],
    enabled: bool,
    min_closed_trades_per_side: int,
    short_alpha_bps_relax: float,
    short_structure_gap_bonus_bps: float,
    short_structure_min_edge_relax_bps: float,
    quiet_long_direction_tighten: float,
) -> Dict[str, Any]:
    adjustments = {
        "enabled": bool(enabled),
        "min_closed_trades_per_side": int(min_closed_trades_per_side or 0),
        "short_alpha_bps_delta": 0.0,
        "short_structure_gap_bonus_bps": 0.0,
        "short_structure_min_edge_delta_bps": 0.0,
        "quiet_long_direction_delta": 0.0,
        "live_sample_ready_by_side": {"long": False, "short": False},
        "reasons": [],
    }
    if not adjustments["enabled"]:
        return adjustments

    by_side = dict((shadow_summary or {}).get("by_side", {}) or {})
    long_rec = str((by_side.get("long", {}) or {}).get("recommendation", "") or "")
    short_rec = str((by_side.get("short", {}) or {}).get("recommendation", "") or "")
    live_min_count = int(adjustments["min_closed_trades_per_side"] or 0)
    long_count = int((by_side.get("long", {}) or {}).get("count", 0) or 0)
    short_count = int((by_side.get("short", {}) or {}).get("count", 0) or 0)
    long_live_ready = long_count >= live_min_count
    short_live_ready = short_count >= live_min_count
    adjustments["live_sample_ready_by_side"]["long"] = bool(long_live_ready)
    adjustments["live_sample_ready_by_side"]["short"] = bool(short_live_ready)

    if long_rec == "TIGHTEN_QUIET_LONG":
        if not long_live_ready:
            adjustments["reasons"].append(f"LONG_LIVE_SAMPLE_{long_count}/{live_min_count}")
        else:
            delta = max(0.0, float(quiet_long_direction_tighten or 0.0))
            if delta > 0.0:
                adjustments["quiet_long_direction_delta"] = delta
                adjustments["reasons"].append("TIGHTEN_QUIET_LONG")

    if short_rec == "RELAX_SHORT_STRUCTURE_CANDIDATE":
        if not short_live_ready:
            adjustments["reasons"].append(
                f"SHORT_LIVE_SAMPLE_{short_count}/{live_min_count}"
            )
        else:
            alpha_relax = max(0.0, float(short_alpha_bps_relax or 0.0))
            gap_bonus = max(0.0, float(short_structure_gap_bonus_bps or 0.0))
            min_edge_relax = max(0.0, float(short_structure_min_edge_relax_bps or 0.0))
            adjustments["short_alpha_bps_delta"] = -alpha_relax
            adjustments["short_structure_gap_bonus_bps"] = gap_bonus
            adjustments["short_structure_min_edge_delta_bps"] = -min_edge_relax
            adjustments["reasons"].append("RELAX_SHORT_STRUCTURE_CANDIDATE")

    return adjustments
