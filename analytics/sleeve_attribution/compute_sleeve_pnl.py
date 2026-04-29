"""
HMATS v5.1 Phase 6.0 prep - Per-sleeve PnL slicer + realized-vol updater
=========================================================================

Reads shadow_ledger FILL records, attributes each fill's realized_pnl to a
sleeve based on FILL.data.primary_agent, then computes:
  - per-sleeve daily PnL series
  - per-sleeve rolling 30d realized volatility (annualized)
  - per-sleeve Sharpe estimate (annualized, daily rebalance approximation)

Output (advisory):
  analytics/sleeve_attribution/reports/sleeve_pnl_{utc_ts}.json
  + optional callable that feeds SleeveAllocator.update_realized_vol(name, vol)

Iron Laws honored:
  3. training/: untouched (this lives in analytics/, not training/).
  4. fail-closed: malformed FILL line skipped + WARN; missing primary_agent
     -> attributed to "unknown" sleeve.
  7. Phase 6.0 prep: this is the slicer that Phase 10 will use to feed
     realized vol into the live SleeveAllocator. For now we only EMIT
     advisory output; there's no auto-feed back into a running allocator
     instance.

Sleeve mapping rules (codifies Phase 7 sleeve assignments):
  primary_agent in {quant, drl, sentiment, regime, kraken_quant,
                    short_bias, two_stage, llm_sentiment, onchain_*,
                    funding_rate, risk_appetite, model_alpha,
                    microstructure_agent, vol_alpha, cvd, options,
                    soldex, whale, flow, structure}
    -> sleeve "directional_short"  (the existing live system)

  primary_agent prefixed "phase4_" or strategy_name in {ofi, vpin_spike,
                    kyle_lambda}
    -> sleeve "microstructure"     (Phase 4 shadow, still empty in live)

  primary_agent prefixed "phase8_" or strategy_name in {cascade_anticipation,
                    stop_hunt_defense}
    -> sleeve "cascade"            (Phase 8 shadow, still empty in live)

  unknown / missing -> sleeve "unknown"  (debug only, NOT promoted)
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER_DIR = REPO / "data" / "shadow_ledger"
REPORT_DIR = REPO / "analytics" / "sleeve_attribution" / "reports"


# ---------------------------------------------------------------------------
# Sleeve mapping
# ---------------------------------------------------------------------------

DIRECTIONAL_AGENTS = frozenset({
    "quant", "drl", "sentiment", "regime", "kraken_quant",
    "short_bias", "two_stage", "llm_sentiment",
    "funding_rate", "risk_appetite", "model_alpha",
    "microstructure_agent", "vol_alpha", "cvd", "options",
    "soldex", "whale", "flow", "structure",
    # Plus onchain variants
    "onchain", "onchain_sol", "onchain_graph", "onchain_btc", "onchain_eth",
})

MICRO_SHADOW_AGENTS = frozenset({"ofi", "vpin_spike", "kyle_lambda"})
CASCADE_SHADOW_AGENTS = frozenset({"cascade_anticipation", "stop_hunt_defense"})


def map_agent_to_sleeve(primary_agent: Optional[str]) -> str:
    if not primary_agent:
        return "unknown"
    pa = str(primary_agent).strip().lower()
    if pa in DIRECTIONAL_AGENTS:
        return "directional_short"
    if pa in MICRO_SHADOW_AGENTS or pa.startswith("phase4_"):
        return "microstructure"
    if pa in CASCADE_SHADOW_AGENTS or pa.startswith("phase8_"):
        return "cascade"
    return "unknown"


# ---------------------------------------------------------------------------
# FILL loader
# ---------------------------------------------------------------------------

def load_fills(
    ledger_dir: Path,
    since: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Return a list of {ts, asset, sleeve, realized_pnl, fee, notional}.

    Records with non-numeric realized_pnl are kept but coerced to 0.0 so
    fee aggregation still works (entries always carry fee but realized_pnl
    is non-zero only on closes).
    """
    rows: List[Dict[str, Any]] = []
    if not ledger_dir.exists():
        logger.warning(f"[SLEEVE_PNL] ledger dir does not exist: {ledger_dir}")
        return rows
    skipped = 0
    for fp in sorted(ledger_dir.glob("ledger_*.jsonl")):
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:  # noqa: silent-swallow
                        # Counted in `skipped`; batch WARN at end emits total
                        skipped += 1
                        continue
                    if r.get("entry_type") != "FILL":
                        continue
                    ts_str = r.get("timestamp") or r.get("ts")
                    if not ts_str:
                        skipped += 1
                        continue
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except ValueError:  # noqa: silent-swallow
                        # Counted in `skipped`; batch WARN at end emits total
                        skipped += 1
                        continue
                    # Pre-P39 records have naive timestamps (datetime.utcnow()).
                    # Treat naive as UTC per CLAUDE.md datetime-mixing remediation.
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if since is not None and ts < since:
                        continue
                    d = r.get("data") or {}

                    def _f(key: str, default: float = 0.0) -> float:
                        v = d.get(key, default)
                        try:
                            f = float(v)
                            if f != f:
                                return default
                            return f
                        except (TypeError, ValueError):  # noqa: silent-swallow
                            # Per-field type coercion; default value is the
                            # documented fallback for missing/bad numerics.
                            return default

                    sleeve = map_agent_to_sleeve(d.get("primary_agent"))
                    rows.append({
                        "ts": ts,
                        "asset": r.get("asset"),
                        "sleeve": sleeve,
                        "primary_agent": d.get("primary_agent"),
                        "realized_pnl": _f("realized_pnl", 0.0),
                        "fee": _f("fee", 0.0),
                        "trade_fee_usd": _f("trade_fee_usd", 0.0),
                        "notional": _f("notional", 0.0),
                        "side": d.get("side"),
                    })
        except Exception as e:
            logger.warning(f"[SLEEVE_PNL] failed to read {fp}: {type(e).__name__}: {e}")
    if skipped:
        logger.warning(f"[SLEEVE_PNL] skipped {skipped} malformed FILL records")
    return rows


# ---------------------------------------------------------------------------
# Per-sleeve PnL aggregation
# ---------------------------------------------------------------------------

def aggregate_daily_sleeve_pnl(
    fills: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Returns {sleeve_name: {YYYY-MM-DD: total_realized_pnl_minus_fees}}.

    Notes:
      - Realized PnL is recorded only on close events; entries have
        realized_pnl == 0. Fees apply on every fill. Net daily PnL = sum of
        realized_pnl on close fills minus all fees on fills that day.
      - Per CLAUDE.md P64: live FILL records may have realized_pnl=0 due to
        the LIVE-mode feedback gate (P65 fix landed 2026-04-25). Pre-P65
        records will under-report realized PnL for live mode. Phase 6.0 work
        anchors against equity_history for ground truth; this slicer is
        complementary attribution.
    """
    out: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in fills:
        sleeve = r["sleeve"]
        day = r["ts"].astimezone(timezone.utc).strftime("%Y-%m-%d")
        # Aggregate net = realized_pnl − fee. trade_fee_usd duplicates fee
        # in some FILL records; prefer fee, fall back to trade_fee_usd if
        # fee is zero
        fee = r["fee"] or r["trade_fee_usd"]
        out[sleeve][day] += r["realized_pnl"] - fee
    # Convert to plain dicts
    return {s: dict(days) for s, days in out.items()}


# ---------------------------------------------------------------------------
# Vol + Sharpe
# ---------------------------------------------------------------------------

def compute_vol_and_sharpe(
    daily_pnls: Dict[str, float],
    initial_capital: float = 10_000.0,
    trading_days_per_year: int = 365,  # crypto = 24/7
) -> Tuple[float, float, int]:
    """Given {YYYY-MM-DD: pnl_usd}, return (annualized_vol_pct, annual_sharpe, n_days).

    Returns (0.0, 0.0, n) on insufficient data (n < 2).
    """
    if not daily_pnls:
        return 0.0, 0.0, 0
    sorted_days = sorted(daily_pnls.keys())
    rets = []
    cum = 0.0
    for d in sorted_days:
        pnl = daily_pnls[d]
        # Daily return = pnl / (initial_capital + cum_pnl_so_far). Use prior
        # day's running equity to avoid bias.
        denom = initial_capital + cum
        if denom > 0:
            rets.append(pnl / denom)
        cum += pnl

    n = len(rets)
    if n < 2:
        return 0.0, 0.0, n
    mean_r = sum(rets) / n
    var_r = sum((r - mean_r) ** 2 for r in rets) / (n - 1)
    std_r = math.sqrt(var_r)
    if std_r <= 0:
        return 0.0, 0.0, n
    ann_vol = std_r * math.sqrt(trading_days_per_year)
    ann_sharpe = (mean_r / std_r) * math.sqrt(trading_days_per_year)
    return ann_vol, ann_sharpe, n


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(
    ledger_dir: Path,
    window_days: int,
    initial_capital: float = 10_000.0,
) -> Dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(days=window_days)
    fills = load_fills(ledger_dir, since=since)
    by_sleeve = aggregate_daily_sleeve_pnl(fills)

    sleeves: List[Dict[str, Any]] = []
    for sleeve_name, daily in by_sleeve.items():
        ann_vol, ann_sharpe, n = compute_vol_and_sharpe(
            daily, initial_capital=initial_capital
        )
        total_pnl = sum(daily.values())
        sleeves.append({
            "name": sleeve_name,
            "n_days": n,
            "total_net_pnl": total_pnl,
            "annualized_vol": ann_vol,
            "annualized_sharpe": ann_sharpe,
            "n_fills": sum(1 for f in fills if f["sleeve"] == sleeve_name),
        })

    # Sort by total PnL descending for human readability
    sleeves.sort(key=lambda s: s["total_net_pnl"], reverse=True)

    # Coverage: fraction of fills that mapped to "unknown". Phase 10 should
    # gate any allocator update on coverage >= 0.95 (pre-P25 era records do
    # not carry primary_agent and will inflate the unknown bucket).
    unknown_fills = sum(1 for f in fills if f["sleeve"] == "unknown")
    total_fills = len(fills)
    attribution_coverage = (
        (total_fills - unknown_fills) / total_fills if total_fills > 0 else 1.0
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": window_days,
        "since": since.isoformat(),
        "n_total_fills": total_fills,
        "n_unknown_sleeve_fills": unknown_fills,
        "attribution_coverage": attribution_coverage,
        "initial_capital": initial_capital,
        "sleeves": sleeves,
    }


def update_allocator_from_report(allocator: Any, report: Dict[str, Any]) -> int:
    """Phase 10 hook (advisory in Phase 6.0): feed each sleeve's annualized_vol
    into a SleeveAllocator instance. Returns the number of sleeves updated.

    Iron Law 7: caller must ensure the allocator is an advisory instance,
    NOT one wired into UnifiedPositionSizer. Phase 10 promotion will gate
    this call behind operator approval.
    """
    n = 0
    # Inspect the allocator's registered sleeves so we only count updates
    # to known sleeves. Unknown sleeves are silently no-ops in update_realized_vol
    # but we don't want to inflate the success count for them.
    known: set = set()
    try:
        known = set(getattr(allocator, "_sleeves", {}).keys())
    except Exception as e:
        # CLAUDE.md P25/P85: surface the gap. Reaching here means the
        # allocator's _sleeves attribute exists but is non-iterable (weird
        # state, e.g. operator partially-instantiated allocator). Empty
        # known-set is the conservative fallback.
        logger.warning(
            f"[SLEEVE_PNL] allocator._sleeves access raised "
            f"{type(e).__name__}: {e} — feeding all sleeves blind"
        )
        known = set()

    for s in report.get("sleeves", []):
        name = s.get("name")
        ann_vol = s.get("annualized_vol", 0.0)
        if not name or ann_vol <= 0:
            continue
        if known and name not in known:
            logger.debug(f"[SLEEVE_PNL] skipping unknown sleeve in report: {name}")
            continue
        try:
            allocator.update_realized_vol(name, ann_vol)
            n += 1
        except Exception as e:
            logger.warning(
                f"[SLEEVE_PNL] update_realized_vol({name}, {ann_vol}) raised "
                f"{type(e).__name__}: {e}"
            )
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Per-sleeve PnL + realized-vol slicer.")
    p.add_argument("--ledger-dir", default=str(DEFAULT_LEDGER_DIR))
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    report = build_report(
        ledger_dir=Path(args.ledger_dir),
        window_days=args.window_days,
        initial_capital=args.initial_capital,
    )

    # Print human-readable
    print("=" * 80)
    print(f"  SLEEVE PNL REPORT  window={args.window_days}d  fills={report['n_total_fills']}"
          f"  attribution_coverage={report['attribution_coverage']:.1%}")
    print("=" * 80)
    print(f"  {'sleeve':<22} {'days':>5} {'fills':>6} {'net_pnl':>10} {'ann_vol':>10} {'sharpe':>8}")
    print("-" * 80)
    for s in report["sleeves"]:
        print(f"  {s['name']:<22} {s['n_days']:>5} {s['n_fills']:>6} "
              f"{s['total_net_pnl']:>10.2f} {s['annualized_vol']:>9.2%} "
              f"{s['annualized_sharpe']:>8.2f}")
    print("=" * 80)
    if report['attribution_coverage'] < 0.95:
        print(f"  WARNING: attribution coverage {report['attribution_coverage']:.1%} < 95% — "
              f"{report['n_unknown_sleeve_fills']} fills lack primary_agent (likely pre-P25 era).")
        print(f"  Phase 10 promotion gate should require coverage >= 95% before "
              f"feeding allocator.")
        print()

    # Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else (
        REPORT_DIR / f"sleeve_pnl_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    )
    out_path.write_text(json.dumps(report, default=str, indent=2), encoding="utf-8")
    print(f"\nReport saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
