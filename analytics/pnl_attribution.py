"""
================================================================================
HMATS v3.4 - PnL ATTRIBUTION & FEEDBACK
================================================================================
Version: 3.4.0
Type: META-LAYER (not an agent, does not generate signals)
Purpose: Attribute PnL to system components for optimization and diagnostics

FUNCTION:
    - Decompose each trade's PnL into component contributions
    - Track attribution to: Entry, Execution, Exit
    - Persist data for analysis and optimization
    - Provide feedback for system tuning

ATTRIBUTION BREAKDOWN:
    Entry Attribution:
        - Quant signal quality (direction accuracy)
        - Regime timing (phase at entry)
        - Mode effectiveness (OPPORTUNITY vs NORMAL)
    
    Execution Attribution:
        - Slippage cost/savings
        - Execution mode effectiveness
        - Lead-lag edge capture
    
    Exit Attribution:
        - Scale-out timing
        - Runner management
        - Stop efficiency

CRITICAL CONSTRAINTS:
    - Does NOT generate trades or signals
    - Is PASSIVE (only observes and records)
    - Persists data for offline analysis
    - Does not modify real-time behavior

WHY THIS EXISTS:
    Without attribution, you know IF you made money, but not WHERE.
    This layer enables:
        - Identifying which components add/subtract value
        - Tuning parameters based on evidence
        - Detecting regime-specific underperformance
================================================================================
"""

import logging
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EntryAttribution:
    """Attribution for entry decision."""
    
    # Quant contribution
    quant_signal_strength: float = 0.0
    quant_direction_correct: bool = False
    quant_confidence_at_entry: float = 0.0
    
    # Regime contribution
    regime_phase_at_entry: str = ""
    regime_timing_score: float = 0.0  # 0-1, how good was the entry timing
    
    # Mode contribution
    mode_at_entry: str = ""  # OPPORTUNITY or NORMAL
    mode_was_appropriate: bool = False
    
    # Opportunity density
    opportunity_density_at_entry: float = 0.0
    
    # Calculated attribution
    entry_alpha_bps: float = 0.0  # PnL attributable to entry quality
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExecutionAttribution:
    """Attribution for execution."""
    
    # Slippage
    expected_price: float = 0.0
    actual_entry_price: float = 0.0
    slippage_bps: float = 0.0
    
    # Mode
    execution_mode_used: str = ""  # PASSIVE, AGGRESSIVE, etc.
    timing_score_at_execution: float = 0.0
    
    # Lead-lag
    lead_lag_edge_captured: float = 0.0
    lead_lag_was_correct: bool = False
    
    # Time to fill
    bars_to_fill: int = 0
    filled_at_target: bool = False
    
    # Calculated attribution
    execution_alpha_bps: float = 0.0  # PnL from good/bad execution
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExitAttribution:
    """Attribution for exit."""
    
    # Scale-out
    scale_out_triggered: bool = False
    scale_out_trigger_type: str = ""
    scale_out_pnl_bps: float = 0.0
    scale_out_timing_score: float = 0.0  # vs optimal exit
    
    # Runner
    runner_created: bool = False
    runner_pnl_bps: float = 0.0
    runner_bars_held: int = 0
    runner_exit_reason: str = ""
    
    # Stop
    stop_triggered: bool = False
    stop_type: str = ""  # SOFT or HARD
    stop_slippage_bps: float = 0.0
    
    # Comparison
    actual_exit_bar: int = 0
    optimal_exit_bar: int = 0  # Retrospectively calculated
    exit_timing_delta_bps: float = 0.0  # PnL difference from optimal
    
    # Calculated attribution
    exit_alpha_bps: float = 0.0  # PnL from exit quality
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TradeAttribution:
    """Complete attribution for a trade."""
    
    # Trade identity
    trade_id: str = ""
    asset: str = ""
    direction: int = 0  # 1 = long, -1 = short
    
    # Timing — P71: tz-aware default (sibling of P68 exit_alpha.py:135 fix).
    # datetime.utcnow returns naive; mixing with tz-aware comparisons elsewhere
    # raises TypeError. P39/P40 family.
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exit_time: Optional[datetime] = None
    
    # PnL
    gross_pnl_bps: float = 0.0
    fees_bps: float = 0.0
    net_pnl_bps: float = 0.0
    
    # Component attributions
    entry: EntryAttribution = field(default_factory=EntryAttribution)
    execution: ExecutionAttribution = field(default_factory=ExecutionAttribution)
    exit: ExitAttribution = field(default_factory=ExitAttribution)
    
    # Validation
    attribution_complete: bool = False
    attribution_error: str = ""
    
    def validate(self) -> bool:
        """Validate attribution sums correctly."""
        component_sum = (
            self.entry.entry_alpha_bps +
            self.execution.execution_alpha_bps +
            self.exit.exit_alpha_bps
        )
        
        # Allow 5bps tolerance for rounding
        delta = abs(component_sum - self.net_pnl_bps)
        if delta > 5:
            self.attribution_error = f"Component sum {component_sum:.1f} != net PnL {self.net_pnl_bps:.1f}"
            return False
        
        self.attribution_complete = True
        return True
    
    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "asset": self.asset,
            "direction": self.direction,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "gross_pnl_bps": self.gross_pnl_bps,
            "fees_bps": self.fees_bps,
            "net_pnl_bps": self.net_pnl_bps,
            "entry": self.entry.to_dict(),
            "execution": self.execution.to_dict(),
            "exit": self.exit.to_dict(),
            "attribution_complete": self.attribution_complete,
        }


@dataclass
class AttributionSummary:
    """Summary statistics for attribution."""
    
    # Counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # Total PnL
    total_gross_pnl_bps: float = 0.0
    total_fees_bps: float = 0.0
    total_net_pnl_bps: float = 0.0
    
    # Component contributions (sum across all trades)
    total_entry_alpha_bps: float = 0.0
    total_execution_alpha_bps: float = 0.0
    total_exit_alpha_bps: float = 0.0
    
    # Averages
    avg_entry_alpha_bps: float = 0.0
    avg_execution_alpha_bps: float = 0.0
    avg_exit_alpha_bps: float = 0.0
    
    # Win rate by mode
    opportunity_win_rate: float = 0.0
    normal_win_rate: float = 0.0
    
    # Best/worst components
    best_component: str = ""
    worst_component: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PnLAttributionManager:
    """
    Meta-layer for PnL attribution and feedback.
    
    USAGE:
        manager = PnLAttributionManager()
        
        # Start tracking a trade
        manager.start_trade(trade_id, asset, direction, entry_context)
        
        # Record execution details
        manager.record_execution(trade_id, execution_details)
        
        # Complete trade with exit details
        manager.complete_trade(trade_id, exit_context, final_pnl)
        
        # Get attribution
        attr = manager.get_trade_attribution(trade_id)
        
        # Get summary
        summary = manager.get_summary()
        
        # Persist for analysis
        manager.persist_to_file(path)
    
    CRITICAL: This is PASSIVE. It only observes and records.
    It does not modify trading behavior.
    """
    
    def __init__(self, persist_path: Optional[Path] = None):
        self._trades: Dict[str, TradeAttribution] = {}
        self._completed_trades: List[TradeAttribution] = []
        self._persist_path = persist_path
        
        logger.info("PnLAttributionManager initialized")
    
    def start_trade(
        self,
        trade_id: str,
        asset: str,
        direction: int,
        entry_context: Dict,
    ) -> TradeAttribution:
        """
        Start tracking a new trade.
        
        Args:
            trade_id: Unique trade identifier
            asset: Trading asset
            direction: 1 for long, -1 for short
            entry_context: Context at entry including:
                - quant_signal, quant_confidence
                - regime_phase, mode, opportunity_density
                - expected_price
        """
        
        attr = TradeAttribution(
            trade_id=trade_id,
            asset=asset,
            direction=direction,
            entry_time=datetime.now(timezone.utc),
        )
        
        # Record entry attribution
        attr.entry = EntryAttribution(
            quant_signal_strength=entry_context.get("quant_signal", 0.0),
            quant_confidence_at_entry=entry_context.get("quant_confidence", 0.0),
            regime_phase_at_entry=entry_context.get("regime_phase", ""),
            mode_at_entry=entry_context.get("mode", "NORMAL"),
            opportunity_density_at_entry=entry_context.get("opportunity_density", 0.0),
        )
        
        # Record expected price for execution tracking
        attr.execution.expected_price = entry_context.get("expected_price", 0.0)
        
        self._trades[trade_id] = attr
        
        logger.debug(f"Attribution: Started tracking trade {trade_id}")
        
        return attr
    
    def record_execution(
        self,
        trade_id: str,
        execution_details: Dict,
    ):
        """
        Record execution details for a trade.
        
        Args:
            execution_details: Including:
                - actual_entry_price
                - execution_mode
                - timing_score
                - lead_lag_edge, lead_lag_correct
                - bars_to_fill
        """
        
        if trade_id not in self._trades:
            logger.warning(f"Attribution: Unknown trade {trade_id}")
            return
        
        attr = self._trades[trade_id]
        
        attr.execution.actual_entry_price = execution_details.get("actual_entry_price", 0.0)
        attr.execution.execution_mode_used = execution_details.get("execution_mode", "")
        attr.execution.timing_score_at_execution = execution_details.get("timing_score", 0.0)
        attr.execution.lead_lag_edge_captured = execution_details.get("lead_lag_edge", 0.0)
        attr.execution.lead_lag_was_correct = execution_details.get("lead_lag_correct", False)
        attr.execution.bars_to_fill = execution_details.get("bars_to_fill", 0)
        attr.execution.filled_at_target = execution_details.get("filled_at_target", False)
        
        # Calculate slippage
        if attr.execution.expected_price > 0 and attr.execution.actual_entry_price > 0:
            slippage_pct = (
                (attr.execution.actual_entry_price - attr.execution.expected_price) /
                attr.execution.expected_price
            )
            # For long, positive slippage is bad; for short, negative is bad
            attr.execution.slippage_bps = slippage_pct * 10000 * attr.direction
    
    def record_scale_out(
        self,
        trade_id: str,
        scale_out_details: Dict,
    ):
        """Record scale-out details."""
        
        if trade_id not in self._trades:
            return
        
        attr = self._trades[trade_id]
        
        attr.exit.scale_out_triggered = True
        attr.exit.scale_out_trigger_type = scale_out_details.get("trigger_type", "")
        attr.exit.scale_out_pnl_bps = scale_out_details.get("pnl_bps", 0.0)
    
    def record_runner_exit(
        self,
        trade_id: str,
        runner_details: Dict,
    ):
        """Record runner exit details."""
        
        if trade_id not in self._trades:
            return
        
        attr = self._trades[trade_id]
        
        attr.exit.runner_created = True
        attr.exit.runner_pnl_bps = runner_details.get("pnl_bps", 0.0)
        attr.exit.runner_bars_held = runner_details.get("bars_held", 0)
        attr.exit.runner_exit_reason = runner_details.get("exit_reason", "")
    
    def complete_trade(
        self,
        trade_id: str,
        exit_context: Dict,
        gross_pnl_bps: float,
        fees_bps: float,
    ) -> Optional[TradeAttribution]:
        """
        Complete a trade and calculate final attribution.
        
        Args:
            exit_context: Including:
                - exit_bar
                - optimal_exit_bar (if calculated)
                - stop_triggered, stop_type
                - quant_direction_was_correct
                - regime_timing_score
                - mode_was_appropriate
        """
        
        if trade_id not in self._trades:
            logger.warning(f"Attribution: Unknown trade {trade_id}")
            return None
        
        attr = self._trades[trade_id]
        
        # Record exit details
        attr.exit_time = datetime.now(timezone.utc)
        attr.gross_pnl_bps = gross_pnl_bps
        attr.fees_bps = fees_bps
        attr.net_pnl_bps = gross_pnl_bps - fees_bps
        
        attr.exit.actual_exit_bar = exit_context.get("exit_bar", 0)
        attr.exit.optimal_exit_bar = exit_context.get("optimal_exit_bar", attr.exit.actual_exit_bar)
        attr.exit.stop_triggered = exit_context.get("stop_triggered", False)
        attr.exit.stop_type = exit_context.get("stop_type", "")
        
        # Update entry attribution
        attr.entry.quant_direction_correct = exit_context.get("quant_direction_was_correct", False)
        attr.entry.regime_timing_score = exit_context.get("regime_timing_score", 0.5)
        attr.entry.mode_was_appropriate = exit_context.get("mode_was_appropriate", True)
        
        # Calculate component attributions
        self._calculate_attribution(attr)
        
        # Validate
        attr.validate()
        
        # Move to completed
        self._completed_trades.append(attr)
        del self._trades[trade_id]
        
        # Auto-persist if path configured
        if self._persist_path:
            self._persist_trade(attr)
        
        logger.info(
            f"Attribution: Trade {trade_id} completed, "
            f"net={attr.net_pnl_bps:+.0f}bps, "
            f"entry={attr.entry.entry_alpha_bps:+.0f}, "
            f"exec={attr.execution.execution_alpha_bps:+.0f}, "
            f"exit={attr.exit.exit_alpha_bps:+.0f}"
        )
        
        return attr
    
    def _calculate_attribution(self, attr: TradeAttribution):
        """Calculate component attributions."""
        
        net_pnl = attr.net_pnl_bps
        
        # Entry alpha: Based on direction correctness and timing
        # If direction was correct, entry gets positive credit
        if attr.entry.quant_direction_correct:
            entry_base = abs(net_pnl) * 0.4  # 40% base for correct direction
            entry_timing = entry_base * attr.entry.regime_timing_score
            attr.entry.entry_alpha_bps = entry_timing if net_pnl > 0 else -entry_timing * 0.5
        else:
            # Wrong direction - entry takes major blame
            attr.entry.entry_alpha_bps = net_pnl * 0.6 if net_pnl < 0 else 0
        
        # Execution alpha: Based on slippage and mode effectiveness
        # Negative slippage is bad, positive is good
        attr.execution.execution_alpha_bps = -attr.execution.slippage_bps
        
        # Add lead-lag contribution
        if attr.execution.lead_lag_was_correct:
            attr.execution.execution_alpha_bps += attr.execution.lead_lag_edge_captured * 10
        
        # Exit alpha: Remaining attribution
        attr.exit.exit_alpha_bps = (
            net_pnl -
            attr.entry.entry_alpha_bps -
            attr.execution.execution_alpha_bps
        )
        
        # Exit timing adjustment
        if attr.exit.optimal_exit_bar != attr.exit.actual_exit_bar:
            # Estimate PnL lost/gained from exit timing
            # This is approximate - would need actual price data
            timing_delta = attr.exit.optimal_exit_bar - attr.exit.actual_exit_bar
            # Assume 5bps per bar of timing difference
            attr.exit.exit_timing_delta_bps = timing_delta * 5
        
        # Scale-out contribution
        if attr.exit.scale_out_triggered:
            # Scale-out timing score affects attribution
            attr.exit.scale_out_timing_score = min(1.0, attr.exit.scale_out_pnl_bps / 100)
    
    def get_trade_attribution(self, trade_id: str) -> Optional[TradeAttribution]:
        """Get attribution for a specific trade."""
        
        # Check active trades
        if trade_id in self._trades:
            return self._trades[trade_id]
        
        # Check completed trades
        for attr in self._completed_trades:
            if attr.trade_id == trade_id:
                return attr
        
        return None
    
    def get_summary(
        self,
        lookback_days: int = 30,
        asset_filter: Optional[str] = None,
        mode_filter: Optional[str] = None,
    ) -> AttributionSummary:
        """
        Get attribution summary.
        
        WHERE CONSUMED:
            - Performance dashboards
            - Optimization tools
            - Strategy tuning
        """
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        
        trades = [
            t for t in self._completed_trades
            if t.entry_time > cutoff and t.attribution_complete
        ]
        
        if asset_filter:
            trades = [t for t in trades if t.asset == asset_filter]
        
        if mode_filter:
            trades = [t for t in trades if t.entry.mode_at_entry == mode_filter]
        
        if not trades:
            return AttributionSummary()
        
        summary = AttributionSummary(
            total_trades=len(trades),
            winning_trades=sum(1 for t in trades if t.net_pnl_bps > 0),
            losing_trades=sum(1 for t in trades if t.net_pnl_bps < 0),
        )
        
        # Totals
        summary.total_gross_pnl_bps = sum(t.gross_pnl_bps for t in trades)
        summary.total_fees_bps = sum(t.fees_bps for t in trades)
        summary.total_net_pnl_bps = sum(t.net_pnl_bps for t in trades)
        
        summary.total_entry_alpha_bps = sum(t.entry.entry_alpha_bps for t in trades)
        summary.total_execution_alpha_bps = sum(t.execution.execution_alpha_bps for t in trades)
        summary.total_exit_alpha_bps = sum(t.exit.exit_alpha_bps for t in trades)
        
        # Averages
        n = len(trades)
        summary.avg_entry_alpha_bps = summary.total_entry_alpha_bps / n
        summary.avg_execution_alpha_bps = summary.total_execution_alpha_bps / n
        summary.avg_exit_alpha_bps = summary.total_exit_alpha_bps / n
        
        # Win rate by mode
        opp_trades = [t for t in trades if t.entry.mode_at_entry == "OPPORTUNITY"]
        normal_trades = [t for t in trades if t.entry.mode_at_entry == "NORMAL"]
        
        if opp_trades:
            summary.opportunity_win_rate = sum(1 for t in opp_trades if t.net_pnl_bps > 0) / len(opp_trades)
        
        if normal_trades:
            summary.normal_win_rate = sum(1 for t in normal_trades if t.net_pnl_bps > 0) / len(normal_trades)
        
        # Best/worst component
        components = {
            "entry": summary.total_entry_alpha_bps,
            "execution": summary.total_execution_alpha_bps,
            "exit": summary.total_exit_alpha_bps,
        }
        summary.best_component = max(components, key=components.get)
        summary.worst_component = min(components, key=components.get)
        
        return summary
    
    def _persist_trade(self, attr: TradeAttribution):
        """Persist a single trade to file."""
        
        if not self._persist_path:
            return
        
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self._persist_path, "a") as f:
                f.write(json.dumps(attr.to_dict()) + "\n")
                f.flush()  # [P37 2026-04-24] crash-safety — without flush, the
                # last few records can be lost when the engine is killed.
        except Exception as e:
            logger.error(f"Attribution: Failed to persist trade: {e}")
    
    def persist_to_file(self, path: Path):
        """Persist all completed trades to file (atomic write)."""
        try:
            data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trades": [t.to_dict() for t in self._completed_trades],
                "summary": self.get_summary().to_dict(),
            }
            # [P37 2026-04-24] Was open(w)+json.dump → corrupt on crash.
            from core.state_persistence import save_state
            if save_state(path, data):
                logger.info(f"Attribution: Persisted {len(self._completed_trades)} trades to {path}")
            else:
                logger.error(f"Attribution: Failed to persist to {path}")
        except Exception as e:
            logger.error(f"Attribution: Failed to persist: {e}")
    
    def reset(self):
        """Reset all data."""
        self._trades.clear()
        self._completed_trades.clear()


# Singleton
_attribution_manager: Optional[PnLAttributionManager] = None

def get_attribution_manager() -> PnLAttributionManager:
    global _attribution_manager
    if _attribution_manager is None:
        _attribution_manager = PnLAttributionManager()
    return _attribution_manager

def reset_attribution_manager():
    global _attribution_manager
    _attribution_manager = None
