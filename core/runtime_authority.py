from __future__ import annotations

from enum import Enum
from typing import Any, Dict


class RuntimeAuthorityState(str, Enum):
    DISABLED = "DISABLED"
    SHADOW = "SHADOW"
    ADVISORY = "ADVISORY"
    BOUNDED_LIVE = "BOUNDED_LIVE"
    LIVE = "LIVE"


ALLOWED_RUNTIME_AUTHORITY_STATES = tuple(state.value for state in RuntimeAuthorityState)


def _norm(value: Any) -> str:
    return str(value or "").strip().upper()


def normalize_runtime_authority(
    module: str,
    *,
    authority: Any = "",
    mode: Any = "",
    effective_mode: Any = "",
    gate_level: Any = "",
    execution_authority: Any = "",
) -> str:
    module_key = _norm(module)
    authority_key = _norm(authority)
    mode_key = _norm(mode)
    effective_mode_key = _norm(effective_mode)
    gate_level_key = _norm(gate_level)
    execution_authority_key = _norm(execution_authority)

    if module_key == "DRL":
        if gate_level_key in ("", "DISABLED", "OFF", "NONE"):
            return RuntimeAuthorityState.DISABLED.value
        if gate_level_key == "SHADOW":
            return RuntimeAuthorityState.SHADOW.value
        if gate_level_key == "EXIT_ONLY":
            return RuntimeAuthorityState.BOUNDED_LIVE.value
        if gate_level_key in ("ACTIVE", "FULL", "LIVE") or execution_authority_key == "ON":
            return RuntimeAuthorityState.LIVE.value
        return RuntimeAuthorityState.DISABLED.value

    if module_key in ("G6", "AGGRESSIVE_ALLOCATOR"):
        if authority_key in ("SIZING_DEFERRED", "DEFERRED", "SHADOW"):
            return RuntimeAuthorityState.SHADOW.value
        if authority_key in ("SIZING", "BOUNDED_LIVE"):
            return RuntimeAuthorityState.BOUNDED_LIVE.value

    if effective_mode_key.startswith("SHADOW") or mode_key.startswith("SHADOW"):
        return RuntimeAuthorityState.SHADOW.value
    if effective_mode_key.startswith("ADVISORY") or mode_key.startswith("ADVISORY"):
        return RuntimeAuthorityState.ADVISORY.value
    if effective_mode_key.startswith("LIVE") or mode_key == "LIVE":
        return RuntimeAuthorityState.LIVE.value

    if authority_key in ("", "DISABLED", "OFF", "NONE"):
        return RuntimeAuthorityState.DISABLED.value
    if authority_key in ("SHADOW",):
        return RuntimeAuthorityState.SHADOW.value
    if authority_key in ("CONTEXT", "EXECUTION", "EXECUTE", "ADVISORY", "ADVISE"):
        return RuntimeAuthorityState.ADVISORY.value
    if authority_key in ("SIZING_DEFERRED", "DEFERRED", "EXIT_ONLY"):
        return RuntimeAuthorityState.SHADOW.value
    if authority_key in ("SIZING", "BOUNDED_LIVE"):
        return RuntimeAuthorityState.BOUNDED_LIVE.value
    if authority_key in ("LIVE", "FULL", "ACTIVE", "ON"):
        return RuntimeAuthorityState.LIVE.value
    return RuntimeAuthorityState.DISABLED.value


def build_runtime_authority_snapshot(
    *,
    composite_toxicity_authority: Any = "",
    learned_exec_authority: Any = "",
    learned_exec_mode: Any = "",
    learned_exec_effective_mode: Any = "",
    g6_authority: Any = "",
    g6_mode: Any = "",
    aggressive_allocator_authority: Any = "",
    lead_lag_authority: Any = "",
    model_alpha_authority: Any = "",
    drl_gate_level: Any = "",
    drl_execution_authority: Any = "",
) -> Dict[str, Any]:
    states = {
        "composite_toxicity": normalize_runtime_authority(
            "composite_toxicity", authority=composite_toxicity_authority
        ),
        "learned_execution_policy": normalize_runtime_authority(
            "learned_execution_policy",
            authority=learned_exec_authority,
            mode=learned_exec_mode,
            effective_mode=learned_exec_effective_mode,
        ),
        "g6": normalize_runtime_authority("g6", authority=g6_authority, mode=g6_mode),
        "aggressive_allocator": normalize_runtime_authority(
            "aggressive_allocator", authority=aggressive_allocator_authority
        ),
        "lead_lag": normalize_runtime_authority("lead_lag", authority=lead_lag_authority),
        "model_alpha": normalize_runtime_authority(
            "model_alpha", authority=model_alpha_authority
        ),
        "drl": normalize_runtime_authority(
            "drl",
            gate_level=drl_gate_level,
            execution_authority=drl_execution_authority,
        ),
    }
    return {
        "allowed_states": list(ALLOWED_RUNTIME_AUTHORITY_STATES),
        "by_module": states,
    }
