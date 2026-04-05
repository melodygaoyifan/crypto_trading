"""
================================================================================
HMATS v6.7 - Exit Alpha Tracker
================================================================================
Purpose: Track per-trade exit efficiency - profit retention rate, exit trigger
effectiveness, peak-to-close drawdown, and counterfactual "what-if held" analysis.

Data flow:
    1. record_entry()       - position opened: entry price, direction, strategy
    2. update_peak()        - each tick: track unrealized high-water mark
    3. record_exit()        - position closed: exit price, trigger, realized PnL
    4. update_counterfactual() - post-exit: track where price went after close

Reports:
    - Profit retention: realized / peak unrealized for each trade
    - Exit trigger: which trigger types (T1-T7) retain the most alpha
    - Timing analysis: too-early vs too-late exits
    - Counterfactual: $ left on the table (or saved from loss)
    - Per-asset / per-regime breakdown

PASSIVE: Does not modify trading behavior. Observe-only.
================================================================================
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")


class ExitAlphaTracker:
    """Per-trade exit efficiency tracking with counterfactual analysis."""

    PERSIST_FILE = DATA_DIR / "exit_alpha_tracking.jsonl"
    COUNTERFACTUAL_FILE = DATA_DIR / "exit_counterfactual.jsonl"

    # Exit trigger taxonomy
    TRIGGER_NAMES = {
        "T1_TRAILING_STOP": "ATR trailing stop (WIRE-2)",
        "T2_INITIAL_STOP": "Exchange stop loss",
        "T3_PHASE_EXIT": "Phase transition scale-out",
        "T4_MOMENTUM_STALL": "Momentum stall exit",
        "T5_PROFIT_DRAWDOWN": "Peak profit drawdown",
        "T6_DRL_ACTION": "DRL EXIT_ONLY action",
        "T7_MAX_HOLD": "Max hold timeout",
        "T8_BREAKEVEN": "Breakeven stop (R-trigger)",
        "T9_GAMBLER": "Gambler soft exit",
        "T10_SOFT_STOP": "Stop authority check",
        "T11_RUNNER_RELEASE": "Runner trailing released",
        "T12_CRACK_DECAY": "CRACK weight decay",
        "T13_FLIP": "Direction flip (implicit close)",
        "T14_P0_FORCE": "P0 safety force flat",
        "UNKNOWN": "Unknown / untagged exit",
    }

    def __init__(self):
        self._open_trades: Dict[str, Dict] = {}        # asset -> active trade
        self._completed: List[Dict] = []                 # finished trades
        self._counterfactuals: Dict[str, Dict] = {}      # asset -> post-exit tracking
        self.by_trigger: Dict[str, List] = defaultdict(list)
        self.by_asset: Dict[str, List] = defaultdict(list)
        self.by_regime: Dict[str, List] = defaultdict(list)
        self._load()

    # ─── Data collection ───

    def record_entry(self, asset: str, price: float, direction: int,
                     strategy: str, regime: str, tick: int,
                     notional: float = 0.0, mode: str = "NORMAL"):
        """Record position open - starts peak tracking."""
        self._open_trades[asset] = {
            "asset": asset,
            "entry_price": price,
            "direction": direction,
            "strategy": strategy,
            "regime_at_entry": regime,
            "tick_entry": tick,
            "notional": notional,
            "mode": mode,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "peak_price": price,
            "peak_unrealized_bps": 0.0,
            "trough_price": price,
            "trough_unrealized_bps": 0.0,
            "ticks_at_peak": 0,
            "updates": 0,
        }

    def update_peak(self, asset: str, current_price: float, tick: int):
        """Update high-water mark for open position. Call every tick."""
        trade = self._open_trades.get(asset)
        if not trade or trade["entry_price"] <= 0:
            return

        trade["updates"] += 1
        direction = trade["direction"]
        entry = trade["entry_price"]
        unrealized_bps = (current_price - entry) / entry * direction * 10000

        # Track peak
        if unrealized_bps > trade["peak_unrealized_bps"]:
            trade["peak_unrealized_bps"] = unrealized_bps
            trade["peak_price"] = current_price
            trade["ticks_at_peak"] = tick

        # Track trough
        if unrealized_bps < trade["trough_unrealized_bps"]:
            trade["trough_unrealized_bps"] = unrealized_bps
            trade["trough_price"] = current_price

    def record_exit(self, asset: str, exit_price: float, trigger: str,
                    pnl_usd: float, tick: int, exit_type: str = "FULL",
                    regime_at_exit: str = ""):
        """Record position close with exit trigger classification."""
        trade = self._open_trades.pop(asset, None)
        if not trade:
            return

        entry = trade["entry_price"]
        direction = trade["direction"]

        # Realized alpha
        realized_bps = 0.0
        if entry > 0:
            realized_bps = (exit_price - entry) / entry * direction * 10000

        peak_bps = trade["peak_unrealized_bps"]
        trough_bps = trade["trough_unrealized_bps"]

        # Profit retention = realized / peak (how much of the best we kept)
        if peak_bps > 0:
            retention_pct = realized_bps / peak_bps * 100
        elif peak_bps == 0 and realized_bps <= 0:
            retention_pct = 0.0  # never profitable, lost money
        else:
            retention_pct = 0.0

        # Peak-to-exit drawdown (how much we gave back from peak)
        peak_to_exit_bps = peak_bps - realized_bps

        # Timing: ticks from peak to exit
        ticks_held = tick - trade["tick_entry"]
        ticks_from_peak = tick - trade["ticks_at_peak"] if trade["ticks_at_peak"] > 0 else 0

        result = {
            **trade,
            "exit_price": exit_price,
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "exit_trigger": trigger,
            "exit_type": exit_type,
            "regime_at_exit": regime_at_exit or trade["regime_at_entry"],
            "tick_exit": tick,
            "pnl_usd": pnl_usd,
            "realized_bps": realized_bps,
            "peak_unrealized_bps": peak_bps,
            "trough_unrealized_bps": trough_bps,
            "retention_pct": retention_pct,
            "peak_to_exit_bps": peak_to_exit_bps,
            "ticks_held": ticks_held,
            "ticks_from_peak": ticks_from_peak,
        }

        self._completed.append(result)
        self.by_trigger[trigger].append(result)
        self.by_asset[trade["asset"]].append(result)
        self.by_regime[trade["regime_at_entry"]].append(result)
        self._persist(result)

        # Start counterfactual tracking
        self._counterfactuals[asset] = {
            "asset": asset,
            "exit_price": exit_price,
            "exit_trigger": trigger,
            "direction": direction,
            "realized_bps": realized_bps,
            "pnl_usd": pnl_usd,
            "tick_exit": tick,
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "post_exit_peak_bps": 0.0,      # best price AFTER exit
            "post_exit_trough_bps": 0.0,     # worst price AFTER exit
            "post_exit_ticks": 0,
            "post_exit_final_bps": 0.0,      # where price is N ticks later
        }

        logger.info(
            f"[EA] {asset} {trigger:25s}: "
            f"realized={realized_bps:+.0f}bps peak={peak_bps:+.0f}bps "
            f"retention={retention_pct:.0f}% "
            f"gave_back={peak_to_exit_bps:.0f}bps "
            f"held={ticks_held}t peak_ago={ticks_from_peak}t "
            f"pnl=${pnl_usd:+.2f}"
        )

    def update_counterfactual(self, asset: str, current_price: float, tick: int):
        """Track price AFTER exit to measure counterfactual outcome.

        Call every tick for recently-closed positions. Auto-expires after 12 ticks (48h).
        """
        cf = self._counterfactuals.get(asset)
        if not cf:
            return

        ticks_since = tick - cf["tick_exit"]
        if ticks_since > 12:  # 48 hours of tracking
            self._persist_counterfactual(cf)
            self._counterfactuals.pop(asset, None)
            return

        cf["post_exit_ticks"] = ticks_since
        exit_price = cf["exit_price"]
        direction = cf["direction"]

        if exit_price > 0:
            post_bps = (current_price - exit_price) / exit_price * direction * 10000
            cf["post_exit_final_bps"] = post_bps
            if post_bps > cf["post_exit_peak_bps"]:
                cf["post_exit_peak_bps"] = post_bps
            if post_bps < cf["post_exit_trough_bps"]:
                cf["post_exit_trough_bps"] = post_bps

    # ─── Persistence ───

    def _persist(self, record: Dict):
        """Append single trade record to JSONL."""
        try:
            self.PERSIST_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.PERSIST_FILE, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.debug(f"[EA] Persist failed: {e}")

    def _persist_counterfactual(self, record: Dict):
        """Append counterfactual to JSONL."""
        try:
            self.COUNTERFACTUAL_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.COUNTERFACTUAL_FILE, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.debug(f"[EA] Counterfactual persist failed: {e}")

    def _load(self):
        """Load historical trade records."""
        if not self.PERSIST_FILE.exists():
            return
        try:
            with open(self.PERSIST_FILE) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self._completed.append(r)
                        self.by_trigger[r.get("exit_trigger", "UNKNOWN")].append(r)
                        self.by_asset[r.get("asset", "")].append(r)
                        self.by_regime[r.get("regime_at_entry", "")].append(r)
                    except Exception:
                        pass
            if self._completed:
                logger.info(f"[EA] Loaded {len(self._completed)} historical exit records")
        except Exception as e:
            logger.warning(f"[EA] Load failed: {e}")

    # ─── Reports ───

    def report(self) -> str:
        """Full exit alpha report."""
        if not self._completed:
            return "[EA] No completed trades with exit tracking yet"
        parts = [
            self._overview(),
            self._by_trigger_report(),
            self._by_asset_report(),
            self._retention_buckets(),
            self._timing_analysis(),
            self._counterfactual_report(),
        ]
        return "\n".join(parts)

    def summary_dict(self) -> Dict:
        """Compact machine-readable summary for runtime exports and weekly review."""
        trades = list(self._completed)
        return self._build_summary_for_trades(trades)

    def _build_summary_for_trades(self, trades: List[Dict]) -> Dict:
        summary = {
            "tracked_trades": len(trades),
            "avg_retention_pct": 0.0,
            "avg_peak_unrealized_bps": 0.0,
            "avg_realized_bps": 0.0,
            "avg_peak_to_exit_bps": 0.0,
            "total_pnl_usd": 0.0,
            "by_trigger": {},
            "by_asset": {},
        }
        if not trades:
            return summary

        summary.update(
            {
                "avg_retention_pct": _safe_avg([t.get("retention_pct", 0.0) for t in trades]),
                "avg_peak_unrealized_bps": _safe_avg([t.get("peak_unrealized_bps", 0.0) for t in trades]),
                "avg_realized_bps": _safe_avg([t.get("realized_bps", 0.0) for t in trades]),
                "avg_peak_to_exit_bps": _safe_avg([t.get("peak_to_exit_bps", 0.0) for t in trades]),
                "total_pnl_usd": float(sum(t.get("pnl_usd", 0.0) for t in trades)),
            }
        )

        by_trigger: Dict[str, List[Dict]] = defaultdict(list)
        by_asset: Dict[str, List[Dict]] = defaultdict(list)
        for row in trades:
            by_trigger[str(row.get("exit_trigger", "UNKNOWN") or "UNKNOWN")].append(row)
            by_asset[str(row.get("asset", "") or "")].append(row)

        for trigger, rows in by_trigger.items():
            if not rows:
                continue
            summary["by_trigger"][trigger] = {
                "count": len(rows),
                "avg_retention_pct": _safe_avg([t.get("retention_pct", 0.0) for t in rows]),
                "avg_peak_to_exit_bps": _safe_avg([t.get("peak_to_exit_bps", 0.0) for t in rows]),
                "avg_pnl_usd": _safe_avg([t.get("pnl_usd", 0.0) for t in rows]),
            }

        for asset, rows in by_asset.items():
            if not rows:
                continue
            summary["by_asset"][asset] = {
                "count": len(rows),
                "avg_retention_pct": _safe_avg([t.get("retention_pct", 0.0) for t in rows]),
                "avg_peak_to_exit_bps": _safe_avg([t.get("peak_to_exit_bps", 0.0) for t in rows]),
                "total_pnl_usd": float(sum(t.get("pnl_usd", 0.0) for t in rows)),
            }

        return summary

    def _filter_trades(
        self,
        *,
        start_at: Optional[str] = None,
        end_at: Optional[str] = None,
        lookback_days: Optional[int] = None,
    ) -> List[Dict]:
        start_dt = _parse_utc_like(start_at) if start_at else None
        end_dt = _parse_utc_like(end_at) if end_at else None
        if end_dt is None:
            end_dt = datetime.now(timezone.utc)
        if start_dt is None and lookback_days is not None:
            start_dt = end_dt - timedelta(days=max(0, int(lookback_days)))

        filtered: List[Dict] = []
        for trade in self._completed:
            exit_dt = _parse_utc_like(trade.get("exit_time"))
            if exit_dt is None:
                if start_dt is None and end_dt is None:
                    filtered.append(trade)
                continue
            if start_dt is not None and exit_dt < start_dt:
                continue
            if end_dt is not None and exit_dt > end_dt:
                continue
            filtered.append(trade)
        return filtered

    def recommendation_dict(
        self,
        *,
        trades: Optional[List[Dict]] = None,
        min_trades: int = 5,
    ) -> Dict:
        scoped_trades = list(self._completed if trades is None else trades)
        summary = self._build_summary_for_trades(scoped_trades)
        recommendations: List[Dict[str, Any]] = []
        best_trigger = _best_metric_entry(summary.get("by_trigger", {}), "avg_retention_pct")
        worst_trigger = _worst_metric_entry(summary.get("by_trigger", {}), "avg_retention_pct")
        best_asset = _best_metric_entry(summary.get("by_asset", {}), "total_pnl_usd")
        worst_asset = _worst_metric_entry(summary.get("by_asset", {}), "total_pnl_usd")

        result = {
            "sample_ready": len(scoped_trades) >= int(min_trades or 0),
            "min_trades": int(min_trades or 0),
            "tracked_trades": len(scoped_trades),
            "verdict": "INSUFFICIENT_SAMPLE",
            "best_trigger": best_trigger,
            "worst_trigger": worst_trigger,
            "best_asset": best_asset,
            "worst_asset": worst_asset,
            "recommendations": recommendations,
        }
        if not result["sample_ready"]:
            return result

        avg_retention = float(summary.get("avg_retention_pct", 0.0) or 0.0)
        avg_giveback = float(summary.get("avg_peak_to_exit_bps", 0.0) or 0.0)
        if avg_retention < 45.0 or avg_giveback > 70.0:
            delta_bps = max(5.0, min(15.0, round(avg_giveback / 10.0) * 1.0))
            recommendations.append(
                {
                    "parameter": "profit_drawdown_exit_bps",
                    "action": "TIGHTEN",
                    "delta_bps": float(delta_bps),
                    "reason": (
                        f"avg_retention={avg_retention:.1f}% avg_peak_to_exit={avg_giveback:.1f}bps"
                    ),
                }
            )
        if avg_retention > 75.0 and avg_giveback < 30.0:
            recommendations.append(
                {
                    "parameter": "runner_release_bias",
                    "action": "LOOSEN",
                    "delta": 0.05,
                    "reason": (
                        f"avg_retention={avg_retention:.1f}% avg_peak_to_exit={avg_giveback:.1f}bps"
                    ),
                }
            )

        if worst_trigger and int(worst_trigger.get("count", 0) or 0) >= 2 and float(
            worst_trigger.get("avg_retention_pct", 0.0) or 0.0
        ) < 35.0:
            recommendations.append(
                {
                    "parameter": "exit_trigger_review",
                    "action": "REVIEW",
                    "target": worst_trigger.get("name", "UNKNOWN"),
                    "reason": (
                        f"trigger retention weak: {float(worst_trigger.get('avg_retention_pct', 0.0) or 0.0):.1f}%"
                    ),
                }
            )

        if worst_asset and int(worst_asset.get("count", 0) or 0) >= 2 and float(
            worst_asset.get("total_pnl_usd", 0.0) or 0.0
        ) < 0.0:
            recommendations.append(
                {
                    "parameter": "asset_exit_review",
                    "action": "REVIEW",
                    "target": worst_asset.get("name", "UNKNOWN"),
                    "reason": (
                        f"asset exit pnl weak: {float(worst_asset.get('total_pnl_usd', 0.0) or 0.0):+.2f} USD"
                    ),
                }
            )

        result["verdict"] = "ADJUST" if recommendations else "KEEP"
        return result

    def weekly_review_dict(
        self,
        *,
        start_at: Optional[str] = None,
        end_at: Optional[str] = None,
        lookback_days: int = 7,
        min_trades: int = 5,
    ) -> Dict:
        trades = self._filter_trades(
            start_at=start_at,
            end_at=end_at,
            lookback_days=lookback_days,
        )
        end_dt = _parse_utc_like(end_at) if end_at else datetime.now(timezone.utc)
        start_dt = (
            _parse_utc_like(start_at)
            if start_at
            else end_dt - timedelta(days=max(0, int(lookback_days)))
        )
        return {
            "window": {
                "start_at": start_dt.isoformat().replace("+00:00", "Z") if start_dt else None,
                "end_at": end_dt.isoformat().replace("+00:00", "Z") if end_dt else None,
                "lookback_days": int(lookback_days),
            },
            "summary": self._build_summary_for_trades(trades),
            "recommendations": self.recommendation_dict(
                trades=trades,
                min_trades=min_trades,
            ),
        }

    def _overview(self) -> str:
        trades = self._completed
        n = len(trades)
        avg_retention = _safe_avg([t.get("retention_pct", 0) for t in trades])
        avg_peak = _safe_avg([t.get("peak_unrealized_bps", 0) for t in trades])
        avg_realized = _safe_avg([t.get("realized_bps", 0) for t in trades])
        avg_gave_back = _safe_avg([t.get("peak_to_exit_bps", 0) for t in trades])
        total_pnl = sum(t.get("pnl_usd", 0) for t in trades)
        winners = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
        return (
            f"\n{'='*70}\n"
            f"  EXIT ALPHA REPORT - {n} trades\n"
            f"{'='*70}\n"
            f"  Avg profit retention:  {avg_retention:.0f}%\n"
            f"  Avg peak unrealized:   {avg_peak:+.0f} bps\n"
            f"  Avg realized alpha:    {avg_realized:+.0f} bps\n"
            f"  Avg gave back:         {avg_gave_back:.0f} bps\n"
            f"  Total PnL:             ${total_pnl:+.2f}\n"
            f"  Win rate:              {winners}/{n} = {winners/n*100:.0f}%\n"
        )

    def _by_trigger_report(self) -> str:
        lines = [
            f"  PER-TRIGGER EXIT EFFECTIVENESS:",
            f"  {'Trigger':25s} {'N':>4s} {'Ret%':>5s} {'Peak':>6s} "
            f"{'Real':>6s} {'Back':>5s} {'AvgPnL':>8s} {'WR':>4s}"
        ]
        lines.append(f"  {'─'*70}")
        for trigger in sorted(self.by_trigger.keys(),
                               key=lambda t: -_safe_avg(
                                   [x.get("retention_pct", 0) for x in self.by_trigger[t]])):
            trades = self.by_trigger[trigger]
            n = len(trades)
            if n == 0:
                continue
            ret = _safe_avg([t.get("retention_pct", 0) for t in trades])
            peak = _safe_avg([t.get("peak_unrealized_bps", 0) for t in trades])
            real = _safe_avg([t.get("realized_bps", 0) for t in trades])
            back = _safe_avg([t.get("peak_to_exit_bps", 0) for t in trades])
            avg_pnl = _safe_avg([t.get("pnl_usd", 0) for t in trades])
            wr = sum(1 for t in trades if t.get("pnl_usd", 0) > 0) / n * 100
            tag = trigger[:25]
            lines.append(
                f"  {tag:25s} {n:>4d} {ret:>4.0f}% {peak:>+5.0f} "
                f"{real:>+5.0f} {back:>4.0f} {avg_pnl:>+7.2f} {wr:>3.0f}%"
            )
        return "\n".join(lines)

    def _by_asset_report(self) -> str:
        lines = [
            f"\n  PER-ASSET EXIT QUALITY:",
            f"  {'Asset':6s} {'N':>4s} {'Ret%':>5s} {'Peak':>6s} "
            f"{'Real':>6s} {'TotalPnL':>9s}"
        ]
        lines.append(f"  {'─'*42}")
        for asset in sorted(self.by_asset.keys()):
            trades = self.by_asset[asset]
            n = len(trades)
            if n == 0:
                continue
            ret = _safe_avg([t.get("retention_pct", 0) for t in trades])
            peak = _safe_avg([t.get("peak_unrealized_bps", 0) for t in trades])
            real = _safe_avg([t.get("realized_bps", 0) for t in trades])
            total = sum(t.get("pnl_usd", 0) for t in trades)
            lines.append(
                f"  {asset:6s} {n:>4d} {ret:>4.0f}% {peak:>+5.0f} "
                f"{real:>+5.0f} {total:>+8.2f}"
            )
        return "\n".join(lines)

    def _retention_buckets(self) -> str:
        """Bucket trades by retention quality."""
        buckets = {
            "Excellent (>80%)": [],
            "Good (50-80%)": [],
            "Poor (20-50%)": [],
            "Terrible (<20%)": [],
            "Negative (lost $)": [],
        }
        for t in self._completed:
            ret = t.get("retention_pct", 0)
            realized = t.get("realized_bps", 0)
            if realized < 0:
                buckets["Negative (lost $)"].append(t)
            elif ret > 80:
                buckets["Excellent (>80%)"].append(t)
            elif ret > 50:
                buckets["Good (50-80%)"].append(t)
            elif ret > 20:
                buckets["Poor (20-50%)"].append(t)
            else:
                buckets["Terrible (<20%)"].append(t)

        n = len(self._completed)
        lines = [f"\n  RETENTION QUALITY BUCKETS:"]
        for label, trades in buckets.items():
            c = len(trades)
            pct = c / n * 100 if n else 0
            avg_pnl = _safe_avg([t.get("pnl_usd", 0) for t in trades]) if trades else 0
            lines.append(f"    {label:22s} {c:>3d} ({pct:>4.0f}%)  avg_pnl=${avg_pnl:+.2f}")
        return "\n".join(lines)

    def _timing_analysis(self) -> str:
        """Analyze exit timing - how far after peak do we exit?"""
        trades = [t for t in self._completed if t.get("ticks_held", 0) > 0]
        if not trades:
            return "\n  TIMING: No trades with tick data"

        avg_held = _safe_avg([t.get("ticks_held", 0) for t in trades])
        avg_from_peak = _safe_avg([t.get("ticks_from_peak", 0) for t in trades])
        at_peak = sum(1 for t in trades if t.get("ticks_from_peak", 0) == 0)
        late = sum(1 for t in trades if t.get("ticks_from_peak", 0) > 3)

        return (
            f"\n  EXIT TIMING:\n"
            f"    Avg hold duration:     {avg_held:.1f} ticks ({avg_held*4:.0f}h)\n"
            f"    Avg ticks after peak:  {avg_from_peak:.1f} ticks ({avg_from_peak*4:.0f}h)\n"
            f"    Exited AT peak:        {at_peak}/{len(trades)} ({at_peak/len(trades)*100:.0f}%)\n"
            f"    Exited >3 ticks late:  {late}/{len(trades)} ({late/len(trades)*100:.0f}%)\n"
        )

    def _counterfactual_report(self) -> str:
        """Report on post-exit price action."""
        # Load persisted counterfactuals
        cfs = self._load_counterfactuals()
        if not cfs:
            return "\n  COUNTERFACTUAL: No post-exit data yet"

        n = len(cfs)
        # Positive post_exit_peak = price kept moving in our favor (left money)
        # Negative post_exit_trough = price reversed (we were right to exit)
        left_money = sum(1 for c in cfs if c.get("post_exit_peak_bps", 0) > 50)
        saved_money = sum(1 for c in cfs if c.get("post_exit_trough_bps", 0) < -50)
        avg_peak = _safe_avg([c.get("post_exit_peak_bps", 0) for c in cfs])
        avg_trough = _safe_avg([c.get("post_exit_trough_bps", 0) for c in cfs])
        avg_final = _safe_avg([c.get("post_exit_final_bps", 0) for c in cfs])

        return (
            f"\n  COUNTERFACTUAL ({n} post-exit tracked, 48h window):\n"
            f"    Avg post-exit peak:   {avg_peak:+.0f} bps (left on table)\n"
            f"    Avg post-exit trough: {avg_trough:+.0f} bps (avoided loss)\n"
            f"    Avg final (12 ticks): {avg_final:+.0f} bps\n"
            f"    Left >50bps on table: {left_money}/{n} ({left_money/n*100:.0f}%)\n"
            f"    Saved >50bps loss:    {saved_money}/{n} ({saved_money/n*100:.0f}%)\n"
        )

    def _load_counterfactuals(self) -> List[Dict]:
        """Load completed counterfactual records."""
        if not self.COUNTERFACTUAL_FILE.exists():
            return []
        result = []
        try:
            with open(self.COUNTERFACTUAL_FILE) as f:
                for line in f:
                    try:
                        result.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass
        return result


# ─── Helpers ───

def _safe_avg(values: list) -> float:
    """Average that handles empty lists."""
    return sum(values) / len(values) if values else 0.0


def _parse_utc_like(value: Optional[str]) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _best_metric_entry(rows: Dict[str, Dict], metric: str) -> Dict[str, Any]:
    if not rows:
        return {}
    name, payload = max(
        rows.items(),
        key=lambda item: float((item[1] or {}).get(metric, 0.0) or 0.0),
    )
    return {"name": name, **dict(payload or {})}


def _worst_metric_entry(rows: Dict[str, Dict], metric: str) -> Dict[str, Any]:
    if not rows:
        return {}
    name, payload = min(
        rows.items(),
        key=lambda item: float((item[1] or {}).get(metric, 0.0) or 0.0),
    )
    return {"name": name, **dict(payload or {})}


# ─── Singleton ───

_instance: Optional[ExitAlphaTracker] = None


def get_exit_alpha_tracker() -> ExitAlphaTracker:
    """Get singleton ExitAlphaTracker."""
    global _instance
    if _instance is None:
        _instance = ExitAlphaTracker()
    return _instance
