"""
================================================================================
HMATS v6.7 - Trade P&L Attributor
================================================================================
Purpose: Financial P&L waterfall - "where the dollars go"

Unlike pnl_attribution.py (signal quality attribution: entry/execution/exit),
this module tracks the MONEY FLOW:

    Gross Alpha -> Entry Fees -> Exit Fees -> Funding Costs -> Net P&L

Provides multi-dimensional breakdowns:
    - Per-asset (BTC / ETH / SOL)
    - Per-strategy (momentum / mean_revert / volume_breakout / vrp)
    - Per-regime (VOLATILE_CHOP / MOMENTUM_RALLY / etc.)
    - Temporal (daily / weekly / session)
    - Fee efficiency (entry_fee_bps, exit_fee_bps, funding_cost_bps)

PASSIVE: Does not modify trading behavior. Observe-only.
================================================================================
"""

import json
import logging
import threading
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TradeRecord:
    """Complete lifecycle record for a single trade."""

    # Identity
    trade_id: str = ""
    asset: str = ""
    direction: int = 0  # +1 long, -1 short

    # Entry
    entry_time: str = ""
    entry_price: float = 0.0
    entry_fee_usd: float = 0.0
    entry_notional: float = 0.0
    strategy: str = ""
    regime_at_entry: str = ""
    mode_at_entry: str = ""  # OPPORTUNITY / NORMAL

    # Funding (accumulated during hold)
    funding_payments: List[Dict[str, Any]] = field(default_factory=list)
    total_funding_pnl: float = 0.0

    # Exit
    exit_time: str = ""
    exit_price: float = 0.0
    exit_fee_usd: float = 0.0
    exit_notional: float = 0.0
    exit_type: str = ""  # FULL / PARTIAL / FLIP

    # P&L waterfall
    gross_alpha_usd: float = 0.0  # (exit - entry) / entry * dir * notional
    net_pnl_usd: float = 0.0     # gross - entry_fee - exit_fee + funding

    # Status
    is_closed: bool = False

    def compute_waterfall(self) -> Dict[str, float]:
        """Compute the P&L waterfall."""
        return {
            "gross_alpha": self.gross_alpha_usd,
            "entry_fee": -self.entry_fee_usd,
            "exit_fee": -self.exit_fee_usd,
            "funding": self.total_funding_pnl,
            "net_pnl": self.gross_alpha_usd - self.entry_fee_usd - self.exit_fee_usd + self.total_funding_pnl,
        }

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["waterfall"] = self.compute_waterfall()
        return d


# =============================================================================
# TRADE ATTRIBUTOR
# =============================================================================

class TradeAttributor:
    """
    Financial P&L waterfall tracker.

    Usage:
        attr = TradeAttributor()

        # On entry
        attr.record_entry("BTC", price=68000, fee=1.77, notional=1106,
                          direction=-1, strategy="momentum", regime="REGIME_1")

        # During hold (periodic funding)
        attr.record_funding("BTC", funding_rate=0.0001, funding_pnl=-0.55)

        # On exit
        attr.record_exit("BTC", price=64890, fee=1.69, notional=1106,
                         gross_pnl=28.84, exit_type="FULL")

        # Report
        attr.report()
    """

    PERSIST_FILE = "data/trade_attribution.jsonl"

    def __init__(self, persist_path: Optional[str] = None):
        self._open_trades: Dict[str, TradeRecord] = {}  # keyed by asset
        self._closed_trades: List[TradeRecord] = []
        self._lock = threading.Lock()
        self._persist_path = Path(persist_path or self.PERSIST_FILE)
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Load previously persisted trades
        self._load_persisted()

        logger.info(
            f"[TradeAttributor] Initialized: {len(self._closed_trades)} historical trades loaded"
        )

    # =========================================================================
    # RECORDING METHODS
    # =========================================================================

    def record_entry(
        self,
        asset: str,
        price: float,
        fee: float,
        notional: float,
        direction: int,
        strategy: str = "",
        regime: str = "",
        mode: str = "",
    ):
        """Record a new position entry."""
        with self._lock:
            # If there's an existing open trade for this asset, auto-close it
            if asset in self._open_trades:
                logger.warning(
                    f"[TradeAttributor] {asset} has open trade, force-closing before new entry"
                )
                old = self._open_trades.pop(asset)
                old.is_closed = True
                old.exit_type = "FORCE_CLOSE"
                old.exit_time = datetime.now(timezone.utc).isoformat()
                self._closed_trades.append(old)

            trade_id = f"{asset}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            rec = TradeRecord(
                trade_id=trade_id,
                asset=asset,
                direction=direction,
                entry_time=datetime.now(timezone.utc).isoformat(),
                entry_price=price,
                entry_fee_usd=fee,
                entry_notional=notional,
                strategy=strategy,
                regime_at_entry=regime,
                mode_at_entry=mode,
            )
            self._open_trades[asset] = rec

            logger.info(
                f"[TradeAttributor] ENTRY {asset} dir={direction:+d} "
                f"price={price:.2f} notional=${notional:,.0f} "
                f"fee=${fee:.4f} strategy={strategy} regime={regime}"
            )

    def record_funding(
        self,
        asset: str,
        funding_rate: float,
        funding_pnl: float,
    ):
        """Record a funding payment for an open position."""
        with self._lock:
            rec = self._open_trades.get(asset)
            if not rec:
                return  # No open trade for this asset

            rec.funding_payments.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "funding_rate": funding_rate,
                "funding_pnl": funding_pnl,
            })
            rec.total_funding_pnl += funding_pnl

            logger.debug(
                f"[TradeAttributor] FUNDING {asset} rate={funding_rate:.6f} "
                f"pnl=${funding_pnl:+.4f} cumulative=${rec.total_funding_pnl:+.4f}"
            )

    def record_exit(
        self,
        asset: str,
        price: float,
        fee: float,
        notional: float,
        gross_pnl: float,
        exit_type: str = "FULL",
    ):
        """Record a position exit (full or partial)."""
        with self._lock:
            rec = self._open_trades.get(asset)
            if not rec:
                # Create a minimal record for orphan exits
                logger.warning(
                    f"[TradeAttributor] EXIT {asset} with no open trade - creating orphan record"
                )
                rec = TradeRecord(
                    trade_id=f"{asset}_orphan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                    asset=asset,
                )

            rec.exit_time = datetime.now(timezone.utc).isoformat()
            rec.exit_price = price
            rec.exit_fee_usd += fee  # accumulate for partial exits
            rec.exit_notional = notional
            rec.exit_type = exit_type
            rec.gross_alpha_usd = gross_pnl
            rec.net_pnl_usd = gross_pnl - rec.entry_fee_usd - rec.exit_fee_usd + rec.total_funding_pnl
            rec.is_closed = True

            if exit_type == "FULL" or exit_type == "FLIP":
                # Remove from open trades
                self._open_trades.pop(asset, None)
            # PARTIAL stays open

            closed_rec = deepcopy(rec)
            self._closed_trades.append(closed_rec)
            self._persist_trade(closed_rec)

            wf = rec.compute_waterfall()
            logger.info(
                f"[TradeAttributor] EXIT {asset} {exit_type} "
                f"gross=${wf['gross_alpha']:+.2f} "
                f"entry_fee=${wf['entry_fee']:.2f} exit_fee=${wf['exit_fee']:.2f} "
                f"funding=${wf['funding']:+.2f} net=${wf['net_pnl']:+.2f}"
            )

    # =========================================================================
    # REPORTS
    # =========================================================================

    def report(self) -> Dict[str, Any]:
        """
        Generate the full P&L waterfall report.

        Returns a dict with:
        - waterfall: {gross_alpha, entry_fees, exit_fees, funding, net_pnl}
        - per_asset: same waterfall per BTC/ETH/SOL
        - per_strategy: same waterfall per strategy
        - per_regime: same waterfall per regime
        - stats: {trade_count, win_rate, avg_hold_hours, fee_ratio}
        """
        with self._lock:
            trades = [t for t in self._closed_trades if t.is_closed]

        if not trades:
            return {"error": "no closed trades", "waterfall": {}, "stats": {}}

        # Grand waterfall
        waterfall = self._aggregate_waterfall(trades)

        # Per-asset
        per_asset = {}
        for asset in ["BTC", "ETH", "SOL"]:
            subset = [t for t in trades if t.asset == asset]
            if subset:
                per_asset[asset] = self._aggregate_waterfall(subset)
                per_asset[asset]["trade_count"] = len(subset)

        # Per-strategy
        per_strategy = {}
        strategies = set(t.strategy for t in trades if t.strategy)
        for strat in strategies:
            subset = [t for t in trades if t.strategy == strat]
            if subset:
                per_strategy[strat] = self._aggregate_waterfall(subset)
                per_strategy[strat]["trade_count"] = len(subset)

        # Per-regime
        per_regime = {}
        regimes = set(t.regime_at_entry for t in trades if t.regime_at_entry)
        for regime in regimes:
            subset = [t for t in trades if t.regime_at_entry == regime]
            if subset:
                per_regime[regime] = self._aggregate_waterfall(subset)
                per_regime[regime]["trade_count"] = len(subset)

        # Stats
        winners = [t for t in trades if t.net_pnl_usd > 0]
        total_notional = sum(t.entry_notional for t in trades)
        total_fees = waterfall["entry_fees"] + waterfall["exit_fees"]

        stats = {
            "trade_count": len(trades),
            "win_count": len(winners),
            "loss_count": len(trades) - len(winners),
            "win_rate": len(winners) / len(trades) if trades else 0,
            "total_notional": total_notional,
            "fee_ratio_bps": abs(total_fees) / total_notional * 10000 if total_notional > 0 else 0,
            "fee_to_gross_pct": abs(total_fees) / abs(waterfall["gross_alpha"]) * 100 if waterfall["gross_alpha"] != 0 else float("inf"),
        }

        # Compute avg hold time
        hold_hours = []
        for t in trades:
            if t.entry_time and t.exit_time:
                try:
                    entry_dt = datetime.fromisoformat(t.entry_time)
                    exit_dt = datetime.fromisoformat(t.exit_time)
                    hold_hours.append((exit_dt - entry_dt).total_seconds() / 3600)
                except Exception:
                    pass
        stats["avg_hold_hours"] = sum(hold_hours) / len(hold_hours) if hold_hours else 0

        return {
            "waterfall": waterfall,
            "per_asset": per_asset,
            "per_strategy": per_strategy,
            "per_regime": per_regime,
            "stats": stats,
        }

    def report_text(self) -> str:
        """Generate a human-readable P&L attribution report."""
        r = self.report()
        if "error" in r:
            return f"[TradeAttributor] {r['error']}"

        wf = r["waterfall"]
        st = r["stats"]
        lines = [
            "=" * 70,
            "  HMATS P&L ATTRIBUTION REPORT",
            "=" * 70,
            "",
            "--- P&L WATERFALL ---",
            f"  Gross Alpha:      ${wf['gross_alpha']:>+10.2f}",
            f"  Entry Fees:       ${wf['entry_fees']:>+10.2f}",
            f"  Exit Fees:        ${wf['exit_fees']:>+10.2f}",
            f"  Funding P&L:      ${wf['funding']:>+10.2f}",
            f"  ─────────────────────────────",
            f"  Net P&L:          ${wf['net_pnl']:>+10.2f}",
            "",
            "--- STATS ---",
            f"  Trades: {st['trade_count']} ({st['win_count']}W / {st['loss_count']}L)  "
            f"Win rate: {st['win_rate']*100:.0f}%",
            f"  Total notional: ${st['total_notional']:,.0f}",
            f"  Fee ratio: {st['fee_ratio_bps']:.1f} bps of notional",
            f"  Fee/Gross: {st['fee_to_gross_pct']:.0f}%",
            f"  Avg hold: {st['avg_hold_hours']:.1f} hours",
        ]

        # Per-asset
        if r["per_asset"]:
            lines.append("")
            lines.append("--- PER ASSET ---")
            for asset in ["BTC", "ETH", "SOL"]:
                if asset in r["per_asset"]:
                    a = r["per_asset"][asset]
                    lines.append(
                        f"  {asset}: gross=${a['gross_alpha']:+.2f} "
                        f"fees=${a['entry_fees']+a['exit_fees']:.2f} "
                        f"funding=${a['funding']:+.2f} "
                        f"net=${a['net_pnl']:+.2f} "
                        f"({a['trade_count']} trades)"
                    )

        # Per-strategy
        if r["per_strategy"]:
            lines.append("")
            lines.append("--- PER STRATEGY ---")
            for strat, v in sorted(r["per_strategy"].items(), key=lambda x: x[1]["net_pnl"], reverse=True):
                lines.append(
                    f"  {strat}: gross=${v['gross_alpha']:+.2f} "
                    f"fees=${v['entry_fees']+v['exit_fees']:.2f} "
                    f"net=${v['net_pnl']:+.2f} ({v['trade_count']} trades)"
                )

        # Per-regime
        if r["per_regime"]:
            lines.append("")
            lines.append("--- PER REGIME ---")
            for regime, v in sorted(r["per_regime"].items(), key=lambda x: x[1]["net_pnl"], reverse=True):
                lines.append(
                    f"  {regime}: gross=${v['gross_alpha']:+.2f} "
                    f"net=${v['net_pnl']:+.2f} ({v['trade_count']} trades)"
                )

        # Fee efficiency diagnostic
        lines.append("")
        lines.append("--- DIAGNOSTIC ---")
        if st["fee_to_gross_pct"] > 100:
            lines.append(
                f"  !! FEES EXCEED GROSS ALPHA ({st['fee_to_gross_pct']:.0f}%) "
                "-> Optimize execution / reduce churn"
            )
        elif st["fee_to_gross_pct"] > 50:
            lines.append(
                f"  ! High fee drag ({st['fee_to_gross_pct']:.0f}% of gross alpha)"
            )
        else:
            lines.append(
                f"  Fee efficiency OK ({st['fee_to_gross_pct']:.0f}% of gross alpha)"
            )

        lines.append("=" * 70)
        return "\n".join(lines)

    # =========================================================================
    # BACKFILL FROM SHADOW LEDGER
    # =========================================================================

    def backfill_from_shadow_ledger(self, ledger_dir: str = "data/shadow_ledger"):
        """
        Reconstruct trade history from shadow ledger FILL records.

        Pairs entry fills (realized_pnl=0) with exit fills (realized_pnl!=0)
        by asset and chronological order.
        """
        from pathlib import Path
        import json

        ledger_path = Path(ledger_dir)
        if not ledger_path.exists():
            logger.warning(f"[TradeAttributor] Ledger dir not found: {ledger_dir}")
            return

        # Read all FILL records chronologically
        fills = []
        for fpath in sorted(ledger_path.glob("ledger_*.jsonl")):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("entry_type") == "FILL":
                            fills.append(rec)
                    except json.JSONDecodeError:
                        continue

        logger.info(f"[TradeAttributor] Backfilling from {len(fills)} FILL records")

        # Pair entries with exits per asset
        open_entries: Dict[str, TradeRecord] = {}
        closed: List[TradeRecord] = []

        for fill in fills:
            asset = fill.get("asset", "")
            data = fill.get("data", {})
            timestamp = fill.get("timestamp", "")
            pnl = data.get("realized_pnl", 0.0)
            fee = data.get("fee", 0.0)
            price = data.get("price", 0.0)
            notional = data.get("notional", 0.0)
            side = data.get("side", "")
            session = fill.get("session_id", "")

            if pnl == 0.0:
                # Entry fill
                direction = -1 if side.upper() == "SELL" else 1
                trade_id = f"{asset}_{session}_{fill.get('tick_id', 0)}"
                rec = TradeRecord(
                    trade_id=trade_id,
                    asset=asset,
                    direction=direction,
                    entry_time=timestamp,
                    entry_price=price,
                    entry_fee_usd=fee,
                    entry_notional=notional,
                )
                open_entries[asset] = rec
            else:
                # Exit fill - pair with most recent entry for this asset
                entry_rec = open_entries.pop(asset, None)
                if entry_rec is None:
                    # Orphan exit - create minimal record
                    entry_rec = TradeRecord(
                        trade_id=f"{asset}_orphan_{timestamp[:10]}",
                        asset=asset,
                    )

                entry_rec.exit_time = timestamp
                entry_rec.exit_price = price
                entry_rec.exit_fee_usd = fee
                entry_rec.exit_notional = notional
                entry_rec.gross_alpha_usd = pnl
                entry_rec.net_pnl_usd = (
                    pnl - entry_rec.entry_fee_usd - fee + entry_rec.total_funding_pnl
                )
                entry_rec.exit_type = "FULL"
                entry_rec.is_closed = True
                closed.append(entry_rec)

        with self._lock:
            self._closed_trades = closed
            self._open_trades = open_entries

        logger.info(
            f"[TradeAttributor] Backfill complete: "
            f"{len(closed)} closed trades, {len(open_entries)} open"
        )

    def needs_backfill(self) -> bool:
        """Return True when no persisted trade attribution exists yet."""
        if self._closed_trades or self._open_trades:
            return False
        try:
            return (not self._persist_path.exists()) or self._persist_path.stat().st_size == 0
        except Exception:
            return True

    def ensure_backfilled(self, ledger_dir: str = "data/shadow_ledger") -> bool:
        """Hydrate attribution history from the shadow ledger when the file is empty."""
        should_persist = False
        try:
            should_persist = (not self._persist_path.exists()) or self._persist_path.stat().st_size == 0
        except Exception:
            should_persist = True

        if not self.needs_backfill():
            return False

        self.backfill_from_shadow_ledger(ledger_dir=ledger_dir)

        if self._closed_trades and should_persist:
            self.persist_to_file(self._persist_path)
        return bool(self._closed_trades or self._open_trades)

    def persist_to_file(self, path: Optional[str | Path] = None) -> int:
        """Rewrite persisted trade attribution with the current closed-trade set."""
        target = Path(path) if path is not None else self._persist_path
        target.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            closed = [deepcopy(t) for t in self._closed_trades if t.is_closed]

        with open(target, "w", encoding="utf-8") as f:
            for trade in closed:
                f.write(json.dumps(trade.to_dict(), default=str) + "\n")

        logger.info(f"[TradeAttributor] Persisted {len(closed)} trades to {target}")
        return len(closed)

    # =========================================================================
    # TEMPORAL REPORT
    # =========================================================================

    def report_temporal(self, bucket: str = "daily") -> Dict[str, Dict]:
        """
        Break down P&L by time bucket.

        Args:
            bucket: "daily" or "weekly"
        """
        with self._lock:
            trades = [t for t in self._closed_trades if t.is_closed and t.exit_time]

        buckets: Dict[str, List[TradeRecord]] = defaultdict(list)
        for t in trades:
            try:
                dt = datetime.fromisoformat(t.exit_time)
                if bucket == "weekly":
                    key = dt.strftime("%Y-W%W")
                else:
                    key = dt.strftime("%Y-%m-%d")
                buckets[key].append(t)
            except Exception:
                pass

        result = {}
        for key in sorted(buckets.keys()):
            result[key] = self._aggregate_waterfall(buckets[key])
            result[key]["trade_count"] = len(buckets[key])

        return result

    # =========================================================================
    # INTERNAL
    # =========================================================================

    @staticmethod
    def _aggregate_waterfall(trades: List[TradeRecord]) -> Dict[str, float]:
        """Aggregate waterfall across a list of trades."""
        gross = sum(t.gross_alpha_usd for t in trades)
        entry_fees = sum(t.entry_fee_usd for t in trades)
        exit_fees = sum(t.exit_fee_usd for t in trades)
        funding = sum(t.total_funding_pnl for t in trades)
        net = gross - entry_fees - exit_fees + funding

        return {
            "gross_alpha": gross,
            "entry_fees": entry_fees,
            "exit_fees": exit_fees,
            "funding": funding,
            "net_pnl": net,
        }

    def _persist_trade(self, trade: TradeRecord):
        """Append a closed trade to the JSONL persistence file."""
        try:
            with open(self._persist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trade.to_dict(), default=str) + "\n")
                f.flush()  # [P37 2026-04-24] crash-safety
        except Exception as e:
            logger.error(f"[TradeAttributor] Persist failed: {e}")

    def _load_persisted(self):
        """Load previously persisted trades from JSONL."""
        if not self._persist_path.exists():
            return

        loaded = 0
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        rec = TradeRecord(
                            trade_id=d.get("trade_id", ""),
                            asset=d.get("asset", ""),
                            direction=d.get("direction", 0),
                            entry_time=d.get("entry_time", ""),
                            entry_price=d.get("entry_price", 0.0),
                            entry_fee_usd=d.get("entry_fee_usd", 0.0),
                            entry_notional=d.get("entry_notional", 0.0),
                            strategy=d.get("strategy", ""),
                            regime_at_entry=d.get("regime_at_entry", ""),
                            mode_at_entry=d.get("mode_at_entry", ""),
                            total_funding_pnl=d.get("total_funding_pnl", 0.0),
                            exit_time=d.get("exit_time", ""),
                            exit_price=d.get("exit_price", 0.0),
                            exit_fee_usd=d.get("exit_fee_usd", 0.0),
                            exit_notional=d.get("exit_notional", 0.0),
                            exit_type=d.get("exit_type", ""),
                            gross_alpha_usd=d.get("gross_alpha_usd", 0.0),
                            net_pnl_usd=d.get("net_pnl_usd", 0.0),
                            is_closed=d.get("is_closed", True),
                        )
                        self._closed_trades.append(rec)
                        loaded += 1
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"[TradeAttributor] Load failed: {e}")

        if loaded:
            logger.info(f"[TradeAttributor] Loaded {loaded} persisted trades")

    # =========================================================================
    # SINGLETON
    # =========================================================================


_trade_attributor: Optional[TradeAttributor] = None
_trade_attributor_lock = threading.Lock()


def get_trade_attributor() -> TradeAttributor:
    """Get singleton TradeAttributor."""
    global _trade_attributor
    if _trade_attributor is None:
        with _trade_attributor_lock:
            if _trade_attributor is None:
                _trade_attributor = TradeAttributor()
                try:
                    if _trade_attributor.ensure_backfilled():
                        logger.info("[TradeAttributor] Shadow-ledger backfill loaded")
                except Exception as e:
                    logger.warning(f"[TradeAttributor] Backfill skipped: {e}")
    return _trade_attributor
