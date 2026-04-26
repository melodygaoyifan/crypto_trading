"""
================================================================================
HMATS v6.7 - Signal Quality Tracker
================================================================================
Purpose: Track per-strategy signal quality - direction accuracy, alpha
calibration, regime performance, and Best-of-N selector effectiveness.

Data flow:
    1. record_candidates() - all 4 strategy scores at decision time
    2. record_signal()     - the selected strategy at entry
    3. record_outcome()    - exit price + PnL for backtest

Reports:
    - Per-strategy: direction accuracy, alpha capture, win rate
    - Per-asset: aggregated signal quality
    - Per-regime: which regime-strategy combos work
    - Alpha calibration: over/under estimation
    - Best-of-N selector: selection distribution + bias detection

PASSIVE: Does not modify trading behavior. Observe-only.
================================================================================
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")


class SignalQualityTracker:
    """Per-strategy signal quality tracking with Best-of-N analysis."""

    PERSIST_FILE = DATA_DIR / "signal_quality.jsonl"

    def __init__(self):
        self.signals: List[Dict] = []       # completed signals with outcomes
        self.by_strategy: Dict[str, List] = defaultdict(list)
        self.by_asset: Dict[str, List] = defaultdict(list)
        self.by_regime: Dict[str, List] = defaultdict(list)
        self.best_of_n_log: List[Dict] = []  # all candidate snapshots
        self._pending: Dict[str, Dict] = {}  # asset -> pending signal
        self._load()

    # ─── Data collection ───

    def record_candidates(self, asset: str, candidates: Dict[str, Dict],
                          selected: str, regime: str, tick: int):
        """Record all 4 strategy candidates + winner at decision time.

        candidates: {strategy_name: {"signal": float, "strength": float}}
        """
        self.best_of_n_log.append({
            "tick": tick,
            "asset": asset,
            "regime": regime,
            "candidates": candidates,
            "selected": selected,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def record_signal(self, asset: str, strategy: str, direction: float,
                      alpha_est_bps: float, confidence: float, regime: str,
                      signal_strength: float, tick: int, price: float):
        """Record the selected signal at entry fill time."""
        self._pending[asset] = {
            "asset": asset,
            "strategy": strategy,
            "direction": direction,
            "alpha_est_bps": alpha_est_bps,
            "confidence": confidence,
            "regime": regime,
            "signal_strength": signal_strength,
            "tick": tick,
            "entry_price": price,
            "entry_time": datetime.now(timezone.utc).isoformat(),
        }

    def record_outcome(self, asset: str, exit_price: float, gross_pnl: float,
                       hold_ticks: int, exit_reason: str = ""):
        """Pair signal with outcome at exit fill time."""
        if asset not in self._pending:
            return
        sig = self._pending.pop(asset)

        # Direction correctness
        if sig["entry_price"] > 0 and exit_price > 0:
            price_moved_up = exit_price > sig["entry_price"]
            was_long = sig["direction"] > 0
            dir_correct = (was_long == price_moved_up) if abs(exit_price - sig["entry_price"]) > 0.01 else True
        else:
            dir_correct = True

        # Realized alpha (bps)
        pct = (exit_price - sig["entry_price"]) / sig["entry_price"] if sig["entry_price"] > 0 else 0
        realized_alpha_bps = sig["direction"] * pct * 10000

        result = {
            **sig,
            "exit_price": exit_price,
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "gross_pnl": gross_pnl,
            "hold_ticks": hold_ticks,
            "exit_reason": exit_reason,
            "direction_correct": dir_correct,
            "realized_alpha_bps": realized_alpha_bps,
            "alpha_error_bps": sig["alpha_est_bps"] - realized_alpha_bps,
        }

        self.signals.append(result)
        self.by_strategy[sig["strategy"]].append(result)
        self.by_asset[sig["asset"]].append(result)
        self.by_regime[sig["regime"]].append(result)
        self._persist(result)

        logger.info(
            f"[SQ] {asset} {sig['strategy']:15s}: "
            f"dir={'OK' if dir_correct else 'WRONG'} "
            f"a_est={sig['alpha_est_bps']:+.0f} a_real={realized_alpha_bps:+.0f} "
            f"pnl=${gross_pnl:+.2f} hold={hold_ticks}t "
            f"exit={exit_reason[:20]}"
        )

    # ─── Persistence ───

    def _persist(self, record: Dict):
        """Append single record to JSONL."""
        try:
            self.PERSIST_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.PERSIST_FILE, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
                f.flush()  # P68 (P37 follow-up): survive Docker SIGKILL
        except Exception as e:
            logger.debug(f"[SQ] Persist failed: {e}")

    def _load(self):
        """Load historical records from JSONL."""
        if not self.PERSIST_FILE.exists():
            return
        try:
            with open(self.PERSIST_FILE) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.signals.append(r)
                        self.by_strategy[r.get("strategy", "unknown")].append(r)
                        self.by_asset[r.get("asset", "")].append(r)
                        self.by_regime[r.get("regime", "")].append(r)
                    except Exception:
                        pass
            if self.signals:
                logger.info(f"[SQ] Loaded {len(self.signals)} historical signal records")
        except Exception as e:
            logger.warning(f"[SQ] Load failed: {e}")

    # ─── Reports ───

    def report(self) -> str:
        """Full signal quality report."""
        if not self.signals:
            return "[SQ] No signals with outcomes recorded yet"
        parts = [
            self._overview(),
            self._by_strategy(),
            self._by_asset(),
            self._by_regime(),
            self._alpha_calibration(),
            self._selector_analysis(),
        ]
        return "\n".join(parts)

    def _overview(self) -> str:
        n = len(self.signals)
        correct = sum(1 for s in self.signals if s.get("direction_correct"))
        avg_est = sum(s.get("alpha_est_bps", 0) for s in self.signals) / n
        avg_real = sum(s.get("realized_alpha_bps", 0) for s in self.signals) / n
        avg_pnl = sum(s.get("gross_pnl", 0) for s in self.signals) / n
        return (
            f"\n{'='*65}\n"
            f"  SIGNAL QUALITY REPORT - {n} signals\n"
            f"{'='*65}\n"
            f"  Direction accuracy:  {correct}/{n} = {correct/n*100:.0f}%\n"
            f"  Avg alpha estimate:  {avg_est:+.0f} bps\n"
            f"  Avg alpha realized:  {avg_real:+.0f} bps\n"
            f"  Alpha capture:       {avg_real/avg_est*100 if avg_est else 0:.0f}%\n"
            f"  Avg gross PnL/trade: ${avg_pnl:+.2f}\n"
        )

    def _by_strategy(self) -> str:
        lines = [
            f"  PER-STRATEGY:",
            f"  {'Strategy':18s} {'N':>4s} {'Dir%':>5s} {'aEst':>6s} "
            f"{'aReal':>6s} {'Capt':>5s} {'AvgPnL':>8s} {'WR':>4s}"
        ]
        lines.append(f"  {'─'*60}")
        for strat in sorted(self.by_strategy.keys()):
            sigs = self.by_strategy[strat]
            n = len(sigs)
            if n == 0:
                continue
            d_acc = sum(1 for s in sigs if s.get("direction_correct")) / n * 100
            a_est = sum(s.get("alpha_est_bps", 0) for s in sigs) / n
            a_real = sum(s.get("realized_alpha_bps", 0) for s in sigs) / n
            capt = a_real / a_est * 100 if a_est else 0
            avg_pnl = sum(s.get("gross_pnl", 0) for s in sigs) / n
            wr = sum(1 for s in sigs if s.get("gross_pnl", 0) > 0) / n * 100
            lines.append(
                f"  {strat:18s} {n:>4d} {d_acc:>4.0f}% {a_est:>+5.0f} "
                f"{a_real:>+5.0f} {capt:>4.0f}% {avg_pnl:>+7.2f} {wr:>3.0f}%"
            )
        return "\n".join(lines)

    def _by_asset(self) -> str:
        lines = [
            f"\n  PER-ASSET:",
            f"  {'Asset':6s} {'N':>4s} {'Dir%':>5s} {'TotalPnL':>9s} {'WR':>4s}"
        ]
        lines.append(f"  {'─'*32}")
        for asset in sorted(self.by_asset.keys()):
            sigs = self.by_asset[asset]
            n = len(sigs)
            if n == 0:
                continue
            d_acc = sum(1 for s in sigs if s.get("direction_correct")) / n * 100
            total = sum(s.get("gross_pnl", 0) for s in sigs)
            wr = sum(1 for s in sigs if s.get("gross_pnl", 0) > 0) / n * 100
            lines.append(f"  {asset:6s} {n:>4d} {d_acc:>4.0f}% {total:>+8.2f} {wr:>3.0f}%")
        return "\n".join(lines)

    def _by_regime(self) -> str:
        lines = [
            f"\n  PER-REGIME:",
            f"  {'Regime':25s} {'N':>4s} {'Dir%':>5s} {'TotalPnL':>9s}"
        ]
        lines.append(f"  {'─'*47}")
        for regime in sorted(self.by_regime.keys(),
                             key=lambda r: -sum(s.get("gross_pnl", 0) for s in self.by_regime[r])):
            sigs = self.by_regime[regime]
            n = len(sigs)
            if n == 0:
                continue
            d_acc = sum(1 for s in sigs if s.get("direction_correct")) / n * 100
            total = sum(s.get("gross_pnl", 0) for s in sigs)
            lines.append(f"  {regime:25s} {n:>4d} {d_acc:>4.0f}% {total:>+8.2f}")
        return "\n".join(lines)

    def _alpha_calibration(self) -> str:
        n = len(self.signals)
        avg_est = sum(s.get("alpha_est_bps", 0) for s in self.signals) / n
        avg_real = sum(s.get("realized_alpha_bps", 0) for s in self.signals) / n
        avg_err = sum(s.get("alpha_error_bps", 0) for s in self.signals) / n
        over = sum(1 for s in self.signals if s.get("alpha_error_bps", 0) > 0)
        bias = "OVERESTIMATES" if avg_err > 0 else "UNDERESTIMATES"
        return (
            f"\n  ALPHA CALIBRATION:\n"
            f"    Avg estimated:  {avg_est:+.0f} bps\n"
            f"    Avg realized:   {avg_real:+.0f} bps\n"
            f"    Avg error:      {avg_err:+.0f} bps ({bias})\n"
            f"    Over-estimated: {over}/{n} ({over/n*100:.0f}%)\n"
        )

    def _selector_analysis(self) -> str:
        if not self.best_of_n_log:
            return "\n  BEST-OF-N: No candidate data recorded yet"
        counts = defaultdict(int)
        for e in self.best_of_n_log:
            counts[e["selected"]] += 1
        total = len(self.best_of_n_log)
        lines = [
            f"\n  BEST-OF-N SELECTOR ({total} decisions):",
            f"    {'Strategy':18s} {'Selected':>8s} {'%':>5s}",
        ]
        for strat in sorted(counts.keys(), key=lambda s: -counts[s]):
            c = counts[strat]
            lines.append(f"    {strat:18s} {c:>8d} {c/total*100:>4.0f}%")
        return "\n".join(lines)


# ─── Singleton ───

_instance: Optional[SignalQualityTracker] = None


def get_signal_quality_tracker() -> SignalQualityTracker:
    """Get singleton SignalQualityTracker."""
    global _instance
    if _instance is None:
        _instance = SignalQualityTracker()
    return _instance