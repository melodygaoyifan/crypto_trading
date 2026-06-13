"""
HMATS v5.1 Phase 2 - Cutover Invariants (Iron Law 1/5/8/9 continuous checks)
============================================================================

Referenced by exchange.adapter and exchange.routing. Pure-logic guards that
gate any Coinbase-migration phase advance. NO live side effects, NO venue
I/O — these are decision functions the operator/orchestration calls before
and during cutover.

The binding invariants during a Coinbase cutover (v5.1 Iron Laws):
  1. obs_dim == 126                       (model input contract unchanged)
  5. DRL authority level == ACTIVE        (DRL preserved through migration)
  8. DRL ACTIVE *continuously* during cutover — advancing a phase while DRL
     has demoted to SHADOW is forbidden; ROLLBACK is always permitted.
  9. maker-first default (post_only)      (fee discipline, defensive)

fail-closed: every function returns an explicit (ok, reason) / result object.
An exception in a caller must be treated as "not safe to advance".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Iron Law 1: the model input contract. Mirrors the architecture rule in
# CLAUDE.md ("DRL state space = 126 dims"). A cutover must never change it.
REQUIRED_OBS_DIM = 126
DRL_ACTIVE = "ACTIVE"


@dataclass
class CutoverInvariantResult:
    """Aggregate result of a cutover invariant sweep. `ok` is True only when
    `violations` is empty."""
    ok: bool
    violations: List[str] = field(default_factory=list)
    checked: List[str] = field(default_factory=list)

    def as_reason(self) -> str:
        if self.ok:
            return f"all {len(self.checked)} cutover invariants hold"
        return "cutover invariant violations: " + "; ".join(self.violations)


def validate_drl_active(drl_authority_level: str) -> Tuple[bool, str]:
    """Iron Law 5/8: DRL authority must be ACTIVE during cutover.

    Returns (ok, reason). Fail-closed: a missing/unknown level is NOT ACTIVE.
    """
    level = str(drl_authority_level or "").upper()
    if level == DRL_ACTIVE:
        return True, "DRL authority ACTIVE"
    return False, (
        f"Iron Law 5/8 violation: DRL authority is '{drl_authority_level}', "
        f"not {DRL_ACTIVE}. Cutover must not proceed while DRL is demoted."
    )


def validate_obs_dim(obs_dim: int) -> Tuple[bool, str]:
    """Iron Law 1: obs_dim must stay 126 across the migration."""
    if int(obs_dim) == REQUIRED_OBS_DIM:
        return True, f"obs_dim == {REQUIRED_OBS_DIM}"
    return False, (
        f"Iron Law 1 violation: obs_dim={obs_dim} != {REQUIRED_OBS_DIM}. "
        f"A Coinbase migration must not restructure the model input."
    )


def validate_maker_first(post_only_default: bool) -> Tuple[bool, str]:
    """Iron Law 9: maker-first default must remain on (post_only)."""
    if bool(post_only_default):
        return True, "maker-first default (post_only) ON"
    return False, "Iron Law 9 violation: maker-first default (post_only) is OFF"


def cutover_invariants(
    drl_authority_level: str,
    obs_dim: int = REQUIRED_OBS_DIM,
    post_only_default: bool = True,
) -> CutoverInvariantResult:
    """Run the full invariant sweep. Call this continuously during a cutover
    (e.g. once per tick / once per phase-advance request).

    Returns a CutoverInvariantResult; `.ok` False => do not advance / consider
    ROLLBACK.
    """
    checks = [
        ("obs_dim", validate_obs_dim(obs_dim)),
        ("drl_active", validate_drl_active(drl_authority_level)),
        ("maker_first", validate_maker_first(post_only_default)),
    ]
    violations = [f"{name}: {reason}" for name, (ok, reason) in checks if not ok]
    return CutoverInvariantResult(
        ok=not violations,
        violations=violations,
        checked=[name for name, _ in checks],
    )


def assert_safe_to_advance(
    routing_policy,
    new_phase,
    drl_authority_level: str,
    obs_dim: int = REQUIRED_OBS_DIM,
    post_only_default: bool = True,
) -> Tuple[bool, str]:
    """Combined gate: invariants MUST hold AND the RoutingPolicy transition
    must be valid. ROLLBACK bypasses the invariant sweep (always permitted)
    because rolling back to Kraken is the safe action.

    This does NOT mutate routing_policy; it only reports whether advancing
    would be permitted. The caller advances via routing_policy.advance_phase().
    """
    # Lazy import to keep this module dependency-free at import time.
    from exchange.routing import CutoverPhase

    if new_phase == CutoverPhase.ROLLBACK:
        return True, "ROLLBACK always permitted (safe action)"

    inv = cutover_invariants(drl_authority_level, obs_dim, post_only_default)
    if not inv.ok:
        return False, inv.as_reason()

    # Dry-run the routing transition validation without committing it.
    allowed_ok, allowed_reason = _routing_transition_allowed(
        routing_policy, new_phase, drl_authority_level
    )
    if not allowed_ok:
        return False, allowed_reason
    return True, f"safe to advance to {new_phase.value}: {inv.as_reason()}"


def _routing_transition_allowed(routing_policy, new_phase, drl_authority_level: str):
    """Mirror RoutingPolicy.advance_phase's validation WITHOUT mutating state."""
    from exchange.routing import CutoverPhase

    if str(drl_authority_level).upper() != DRL_ACTIVE:
        return False, (
            f"Iron Law 8 violation: DRL authority is {drl_authority_level}, "
            f"not {DRL_ACTIVE}."
        )
    valid_progressions = {
        CutoverPhase.PRE_PHASE_2: [CutoverPhase.SHADOW, CutoverPhase.ROLLBACK],
        CutoverPhase.SHADOW: [CutoverPhase.DUAL_VENUE, CutoverPhase.PRE_PHASE_2,
                              CutoverPhase.ROLLBACK],
        CutoverPhase.DUAL_VENUE: [CutoverPhase.COINBASE_PRIMARY, CutoverPhase.SHADOW,
                                  CutoverPhase.ROLLBACK],
        CutoverPhase.COINBASE_PRIMARY: [CutoverPhase.DUAL_VENUE, CutoverPhase.ROLLBACK],
        CutoverPhase.ROLLBACK: [CutoverPhase.PRE_PHASE_2, CutoverPhase.SHADOW],
    }
    allowed = valid_progressions.get(routing_policy.phase, [])
    if new_phase not in allowed:
        return False, (
            f"invalid transition: {routing_policy.phase.value} -> {new_phase.value}. "
            f"Allowed: {[p.value for p in allowed]}"
        )
    return True, "transition allowed"
