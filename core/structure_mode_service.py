"""
Structure mode helpers extracted from main.py (Phase 7).

These are pure/near-pure functions that evaluate structure gate config dicts.
No self.* dependencies except _structure_mode_set and _is_structure_mode_allowed
which are now standalone functions.
"""

from typing import Any, Dict, List, Optional, Set


def structure_mode_set(values: Any) -> Set[str]:
    """Normalize a list of mode/regime values to a set of uppercase strings."""
    return {
        str(value).upper()
        for value in (values or [])
        if str(value).strip()
    }


def is_structure_mode_allowed(
    struct_cfg: Dict[str, Any],
    system_mode: str,
    *,
    is_long: bool,
) -> bool:
    """Check if the current system mode is in the structure gate's allowed modes."""
    base_modes = structure_mode_set(struct_cfg.get("apply_in_system_modes"))
    specific_key = "apply_in_system_modes_long" if is_long else "apply_in_system_modes_short"
    specific_modes = structure_mode_set(struct_cfg.get(specific_key))
    allowed_modes = specific_modes or base_modes
    normalized_mode = str(system_mode or "").upper()
    return (not allowed_modes) or (normalized_mode in allowed_modes)


def get_structure_min_edge_bps(
    struct_cfg: Dict[str, Any],
    asset: str,
    *,
    is_short: bool,
) -> float:
    """Get the minimum net edge in bps for the given asset and side."""
    side_key = "min_net_edge_bps_short" if is_short else "min_net_edge_bps_long"
    by_asset_key = (
        "min_net_edge_bps_short_by_asset" if is_short else "min_net_edge_bps_long_by_asset"
    )
    fallback = float(struct_cfg.get(side_key, struct_cfg.get("min_net_edge_bps", 0.0)) or 0.0)
    by_asset = struct_cfg.get(by_asset_key) or {}
    if not isinstance(by_asset, dict):
        return fallback
    asset_key = str(asset or "").upper().replace("/USD", "").replace("USD", "")
    try:
        return float(by_asset.get(asset_key, fallback) or fallback)
    except Exception:
        return fallback


def get_structure_short_min_edge_bps(
    struct_cfg: Dict[str, Any],
    asset: str,
) -> float:
    """Shorthand for get_structure_min_edge_bps with is_short=True."""
    return get_structure_min_edge_bps(struct_cfg, asset, is_short=True)


def get_effective_structure_short_min_edge_bps(
    base_min_edge_bps: float,
    live_delta_bps: float,
) -> float:
    """Preserve explicit negative asset floors while applying live calibration deltas."""
    try:
        base = float(base_min_edge_bps or 0.0)
    except Exception:
        base = 0.0
    try:
        delta = float(live_delta_bps or 0.0)
    except Exception:
        delta = 0.0
    adjusted = base + delta
    if base >= 0.0:
        return max(0.0, adjusted)
    return adjusted


def get_structure_min_abs_direction(
    struct_cfg: Dict[str, Any],
    asset: str,
    *,
    is_short: bool,
) -> float:
    """Get minimum absolute direction threshold for structure gate."""
    side_key = "min_abs_direction_short" if is_short else "min_abs_direction_long"
    by_asset_key = (
        "min_abs_direction_short_by_asset" if is_short else "min_abs_direction_long_by_asset"
    )
    fallback = float(struct_cfg.get(side_key, struct_cfg.get("min_abs_direction", 0.0)) or 0.0)
    by_asset = struct_cfg.get(by_asset_key) or {}
    if not isinstance(by_asset, dict):
        return fallback
    asset_key = str(asset or "").upper().replace("/USD", "").replace("USD", "")
    try:
        return float(by_asset.get(asset_key, fallback) or fallback)
    except Exception:
        return fallback


def should_pre_structure_hold_short(
    struct_cfg: Dict[str, Any],
    *,
    system_mode: str,
    regime: str,
    strategy_id: str,
    gap_to_support_bps: Optional[float],
    edge_bps: float,
    can_short: bool,
) -> bool:
    """Check if existing short should be held based on structure config."""
    if can_short or not bool(struct_cfg.get("pre_veto_hold_short_enabled", False)):
        return False

    mode_ok = is_structure_mode_allowed(struct_cfg, system_mode, is_long=False)
    if not mode_ok:
        return False

    allowed_regimes = structure_mode_set(
        struct_cfg.get("pre_veto_hold_apply_in_regimes_short")
        or struct_cfg.get("apply_in_regimes_short")
        or struct_cfg.get("apply_in_regimes")
    )
    normalized_regime = str(regime or "").upper()
    if allowed_regimes and normalized_regime not in allowed_regimes:
        return False

    short_mean_revert_strategies = {
        str(value).lower()
        for value in (
            struct_cfg.get("short_mean_revert_strategies")
            or ["mean_revert", "mean_reversion"]
        )
    }
    if str(strategy_id or "").lower() in short_mean_revert_strategies:
        return False

    try:
        max_gap_bps = float(struct_cfg.get("pre_veto_hold_max_support_gap_bps_short", 0.0) or 0.0)
        max_edge_bps = float(struct_cfg.get("pre_veto_hold_max_net_edge_bps_short", 0.0) or 0.0)
    except Exception:
        return False

    if max_gap_bps <= 0.0 or max_edge_bps <= 0.0 or gap_to_support_bps is None:
        return False

    return bool(gap_to_support_bps <= max_gap_bps and edge_bps <= max_edge_bps)


def get_btc_short_shadow_relax_decision(
    struct_cfg: Dict[str, Any],
    *,
    asset: str,
    system_mode: str,
    regime: str,
    strategy_id: str,
    gap_to_support_bps: Optional[float],
    edge_bps: float,
    vpin: Optional[float],
    whale_veto_short: bool,
    can_short: bool,
) -> Dict[str, Any]:
    """Evaluate BTC short shadow relax eligibility."""
    enabled = bool(struct_cfg.get("btc_short_shadow_relax_enabled", False))
    decision: Dict[str, Any] = {
        "enabled": enabled,
        "candidate": False,
        "applied": False,
        "exposure_scale": None,
        "reason": "",
    }

    if str(asset or "").upper() != "BTC" or can_short:
        return decision

    if not is_structure_mode_allowed(struct_cfg, system_mode, is_long=False):
        return decision

    allowed_regimes = structure_mode_set(
        struct_cfg.get("btc_short_shadow_relax_apply_in_regimes")
    )
    normalized_regime = str(regime or "").upper()
    if allowed_regimes and normalized_regime not in allowed_regimes:
        return decision

    short_mean_revert_strategies = {
        str(value).lower()
        for value in (
            struct_cfg.get("short_mean_revert_strategies")
            or ["mean_revert", "mean_reversion"]
        )
    }
    if str(strategy_id or "").lower() in short_mean_revert_strategies:
        return decision

    try:
        max_gap_bps = float(
            struct_cfg.get("btc_short_shadow_relax_max_support_gap_bps", 0.0) or 0.0
        )
        min_edge_bps = float(
            struct_cfg.get("btc_short_shadow_relax_min_net_edge_bps", 0.0) or 0.0
        )
        max_vpin = float(
            struct_cfg.get("btc_short_shadow_relax_max_vpin", 0.0) or 0.0
        )
        exposure_scale = float(
            struct_cfg.get("btc_short_shadow_relax_exposure_scale", 0.0) or 0.0
        )
    except Exception:
        return decision

    if (
        gap_to_support_bps is None
        or max_gap_bps <= 0.0
        or min_edge_bps <= 0.0
        or exposure_scale <= 0.0
        or gap_to_support_bps > max_gap_bps
        or edge_bps < min_edge_bps
    ):
        return decision

    vpin_value = None
    if vpin is not None:
        try:
            vpin_value = float(vpin)
        except Exception:
            vpin_value = None
    if max_vpin > 0.0 and vpin_value is not None and vpin_value > max_vpin:
        return decision

    if bool(struct_cfg.get("btc_short_shadow_relax_block_on_buy_whale", True)) and whale_veto_short:
        return decision

    scale = min(1.0, max(0.0, exposure_scale))
    vpin_part = f"{vpin_value:.2f}" if vpin_value is not None else "n/a"
    decision.update(
        {
            "candidate": True,
            "applied": enabled,
            "exposure_scale": scale,
            "reason": (
                f"BTC short shadow-relax candidate gap={gap_to_support_bps:.1f}bps "
                f"net_edge={edge_bps:.1f}bps vpin={vpin_part} scale={scale:.2f} "
                f"enabled={enabled}"
            ),
        }
    )
    return decision
