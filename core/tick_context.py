"""
================================================================================
TICK CONTEXT - Per-Tick State Container for Pipeline Decomposition
================================================================================
Version: 1.0.0
Purpose: Bundle per-tick state so extracted service functions can receive
         a single context object instead of 20+ individual parameters.

USAGE:
    ctx = TickContext(asset="SOL", market_data=data, tick_id="SOL_20260402_1200")
    enriched = enrich_market_data(ctx, components)

DESIGN PRINCIPLES:
    - Immutable where possible (asset, tick_id)
    - Mutable for accumulated state (agent_signals, position_state)
    - No self.* references - enables extraction from God Object
    - Serializable for diagnostics / audit
================================================================================
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TickContext:
    """Per-tick state container.

    Created at the top of _process_4h_tick_inner and threaded through
    all extracted service functions.
    """

    # ── Immutable per-tick ──
    asset: str = ""
    tick_id: str = ""
    tick_count: int = 0
    tick_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_warmup: bool = False

    # ── Market data (mutable - enriched by pipeline stages) ──
    market_data: Dict[str, Any] = field(default_factory=dict)

    # ── Agent signals (built during pre-decide stage) ──
    agent_signals: Dict[str, Any] = field(default_factory=dict)

    # ── Position state snapshot (for this asset) ──
    position_state: Dict[str, Any] = field(default_factory=dict)

    # ── System state (mode, regime, drawdown) ──
    system_mode: str = "NORMAL"
    regime_state: str = "UNKNOWN"
    regime_confidence: float = 0.0
    regime_power: float = 1.0
    current_drawdown_pct: float = 0.0

    # ── P0 safety pre-tick results ──
    p0_abort_tick: bool = False
    p0_force_flat: bool = False
    p0_exposure_multiplier: float = 1.0
    p0_allow_short: bool = True

    # ── Macro/crowd context ──
    macro_crowd_context: Dict[str, Any] = field(default_factory=dict)

    # ── External feed data (Coinglass, Kraken Futures, OnChain) ──
    external_data: Dict[str, Any] = field(default_factory=dict)

    # ── Diagnostics recorder ──
    diag: Dict[str, Any] = field(default_factory=dict)

    # ── Accumulated modifications for audit trail ──
    modifications: List[str] = field(default_factory=list)

    def record_modification(self, stage: str, detail: str) -> None:
        """Record a pipeline modification for audit."""
        self.modifications.append(f"[{stage}] {detail}")

    def diag_record(
        self,
        name: str,
        called: bool = False,
        output: Any = None,
        consumed: bool = False,
        note: str = "",
        applicable: bool = True,
        expect_consumption: bool = True,
    ) -> None:
        """Record a component diagnostic entry (matches _diag_record signature)."""
        try:
            if "components" not in self.diag:
                self.diag["components"] = {}
            self.diag["components"][name] = {
                "called": bool(called),
                "output": repr(output)[:200] if output is not None else "None",
                "consumed": bool(consumed),
                "note": note,
                "applicable": bool(applicable),
                "expect_consumption": bool(expect_consumption),
            }
        except Exception:
            pass


@dataclass
class IntentModifications:
    """Tracks all modifications made to an intent by post-decide gates.

    Useful for audit trail: shows which gate changed what and why.
    """

    original_direction: float = 0.0
    original_exposure: float = 0.0
    gate_log: List[str] = field(default_factory=list)

    def record(self, gate_name: str, field_name: str, old_value: Any, new_value: Any) -> None:
        """Record a gate modification."""
        if old_value != new_value:
            self.gate_log.append(
                f"{gate_name}: {field_name} {old_value} -> {new_value}"
            )

    def summary(self) -> str:
        """One-line summary of all modifications."""
        if not self.gate_log:
            return "no_modifications"
        return " | ".join(self.gate_log)
