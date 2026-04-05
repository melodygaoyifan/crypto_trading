from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .runtime_authority import (
    ALLOWED_RUNTIME_AUTHORITY_STATES,
    build_runtime_authority_snapshot,
)


def build_g6_runtime_settings(
    cfg: Optional[Dict[str, Any]],
    *,
    env_shadow: Optional[str] = None,
) -> Dict[str, Any]:
    enabled = True
    shadow_mode = True
    min_multiplier = 0.80
    max_multiplier = 1.20
    cold_start_prior_enabled = True
    cold_start_prior_floor = 0.10
    cold_start_prior_max_multiplier = 1.40

    if isinstance(cfg, dict):
        enabled = bool(cfg.get("enabled", enabled))
        shadow_mode = bool(cfg.get("shadow_mode", shadow_mode))
        min_multiplier = float(cfg.get("min_multiplier", min_multiplier) or min_multiplier)
        max_multiplier = float(cfg.get("max_multiplier", max_multiplier) or max_multiplier)
        cold_start_prior_enabled = bool(
            cfg.get("cold_start_prior_enabled", cold_start_prior_enabled)
        )
        cold_start_prior_floor = float(
            cfg.get("cold_start_prior_floor", cold_start_prior_floor)
            or cold_start_prior_floor
        )
        cold_start_prior_max_multiplier = float(
            cfg.get(
                "cold_start_prior_max_multiplier",
                cold_start_prior_max_multiplier,
            )
            or cold_start_prior_max_multiplier
        )

    if env_shadow is not None:
        shadow_mode = str(env_shadow).lower() == "true"

    if max_multiplier < min_multiplier:
        max_multiplier = min_multiplier

    return {
        "enabled": bool(enabled),
        "shadow_mode": bool(shadow_mode),
        "min_multiplier": float(min_multiplier),
        "max_multiplier": float(max_multiplier),
        "cold_start_prior_enabled": bool(cold_start_prior_enabled),
        "cold_start_prior_floor": max(0.0, float(cold_start_prior_floor)),
        "cold_start_prior_max_multiplier": max(
            1.0, float(cold_start_prior_max_multiplier)
        ),
    }


def build_aggressive_allocator_runtime_settings(
    cfg: Optional[Dict[str, Any]],
    *,
    env_enabled: Optional[str] = None,
    env_min_fills: Optional[str] = None,
) -> Dict[str, Any]:
    live_enabled = False
    min_session_fills = 3

    if isinstance(cfg, dict):
        live_enabled = bool(cfg.get("live_enabled", live_enabled))
        try:
            min_session_fills = max(
                1,
                int(cfg.get("min_session_fills", min_session_fills) or min_session_fills),
            )
        except Exception:
            min_session_fills = 3

    if env_enabled is not None:
        live_enabled = str(env_enabled).lower() == "true"

    if env_min_fills is not None:
        try:
            min_session_fills = max(1, int(env_min_fills or min_session_fills))
        except Exception:
            pass

    return {
        "live_enabled": bool(live_enabled),
        "min_session_fills": int(min_session_fills),
    }


def build_fast_market_execution_decision(
    intent: Any,
    *,
    enabled: bool,
    modes: Optional[Iterable[str]] = None,
    require_aggressive_mode: bool = True,
    min_short_net_alpha_bps: float = 0.0,
    min_short_abs_direction: float = 0.0,
) -> Dict[str, Any]:
    decision = {"apply": False, "reason": "disabled"}
    if not bool(enabled):
        return decision

    system_mode = str(getattr(intent, "system_mode", "") or "").upper()
    exec_mode = getattr(intent, "execution_mode", None)
    exec_mode_value = (
        exec_mode.value if hasattr(exec_mode, "value") else str(exec_mode or "")
    ).upper()

    if bool(require_aggressive_mode) and exec_mode_value != "AGGRESSIVE":
        return {"apply": False, "reason": f"exec_mode={exec_mode_value or 'UNKNOWN'}"}

    if system_mode in set(modes or {"OPPORTUNITY"}):
        return {"apply": True, "reason": f"mode={system_mode}"}

    direction = float(getattr(intent, "direction", 0.0) or 0.0)
    expected_net_alpha_bps = float(
        getattr(
            intent,
            "expected_net_alpha_bps",
            getattr(intent, "alpha_estimated_bps", 0.0),
        )
        or 0.0
    )
    if (
        direction < 0
        and abs(direction) >= float(min_short_abs_direction or 0.0)
        and expected_net_alpha_bps >= float(min_short_net_alpha_bps or 0.0)
    ):
        return {
            "apply": True,
            "reason": (
                f"short_quality net_alpha={expected_net_alpha_bps:.1f}bps "
                f"dir={abs(direction):.3f}"
            ),
        }

    return {
        "apply": False,
        "reason": (
            f"short_quality_miss net_alpha={expected_net_alpha_bps:.1f}bps "
            f"dir={abs(direction):.3f}"
        ),
    }


def build_asset_runtime_authority_snapshot(
    *,
    asset_runtime: Optional[Dict[str, Any]] = None,
    drl_runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    asset_runtime = dict(asset_runtime or {})
    drl_runtime = dict(drl_runtime or {})
    return build_runtime_authority_snapshot(
        composite_toxicity_authority=asset_runtime.get(
            "composite_toxicity_authority", "DISABLED"
        ),
        learned_exec_authority=asset_runtime.get("learned_exec_authority", "DISABLED"),
        learned_exec_mode=asset_runtime.get("learned_exec_mode", "DISABLED"),
        learned_exec_effective_mode=asset_runtime.get(
            "learned_exec_effective_mode", "DISABLED"
        ),
        g6_authority=asset_runtime.get("g6_authority", "DISABLED"),
        g6_mode=asset_runtime.get("g6_mode", "DISABLED"),
        aggressive_allocator_authority=asset_runtime.get(
            "aggressive_allocator_authority", "DISABLED"
        ),
        lead_lag_authority=asset_runtime.get("lead_lag_authority", "DISABLED"),
        model_alpha_authority=asset_runtime.get("model_alpha_authority", "DISABLED"),
        drl_gate_level=drl_runtime.get("gate_level", "DISABLED"),
        drl_execution_authority=drl_runtime.get("execution_authority", "OFF"),
    )


def build_runtime_authority_matrix(
    *,
    asset_runtime_map: Optional[Dict[str, Dict[str, Any]]] = None,
    drl_runtime: Optional[Dict[str, Any]] = None,
    assets: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    runtime_map = dict(asset_runtime_map or {})
    drl_snapshot = dict(drl_runtime or {})
    by_asset = {}
    for asset in assets or []:
        by_asset[str(asset)] = build_asset_runtime_authority_snapshot(
            asset_runtime=runtime_map.get(asset, {}),
            drl_runtime=drl_snapshot,
        ).get("by_module", {})
    return {
        "allowed_states": list(ALLOWED_RUNTIME_AUTHORITY_STATES),
        "by_asset": by_asset,
    }
