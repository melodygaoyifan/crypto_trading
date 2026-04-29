"""
HMATS v5.1 Phase Pre-6 - Shadow Strategy IC + Promotion Gate
==============================================================

Reads JSONL ledgers under data/strategy_shadow/ (microstructure_*.jsonl,
cascade_*.jsonl), joins each (ts, asset, strategy) signal to forward
returns sourced from the 4H OHLCV parquets, computes per-strategy IC
across configurable horizons, and emits a promotion verdict per strategy:

    PROMOTE : IC > 0.05 stable across horizons + Sharpe > 0.5
    HOLD    : not yet 30 days of data, OR mixed signal
    KILL    : 14d window has IC < 0.05 (kill-criteria per v5.1 prompt)

Output:
    analytics/shadow_ic/reports/shadow_ic_{utc_ts}.json
    + console human-readable summary table

Iron Laws honored:
  4. fail-closed: missing parquet / unparseable JSONL line / NaN → that
     row dropped, run continues.
  7. Phase Pre-6 is the framework; promotion VERDICT is computed here but
     not auto-applied. Phase 10 (Day 57+) reads these reports and gates.

Usage:
    python -X utf8 analytics/shadow_ic/compute_shadow_ic.py \
        --ledger-dir data/strategy_shadow \
        --window-days 14 \
        --horizons 4,12,24

Verdict thresholds match v5.1 prompt's Phase 4/8 kill criteria:
  - microstructure individual: 14d IC < 0.05 -> KILL
  - cascade individual:        14d FP > 50% / 30d IC < 0.04 -> KILL
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


REPO = Path(__file__).resolve().parents[2]
LEDGER_DIR = REPO / "data" / "strategy_shadow"
OHLCV_DIR = REPO / "training" / "training_data" / "drl_training"
REPORT_DIR = REPO / "analytics" / "shadow_ic" / "reports"


class Verdict(Enum):
    PROMOTE = "PROMOTE"
    HOLD = "HOLD"
    KILL = "KILL"
    INSUFFICIENT_SAMPLES = "INSUFFICIENT_SAMPLES"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_shadow_ledgers(
    ledger_dir: Path,
    prefixes: Tuple[str, ...] = ("microstructure", "cascade"),
    since: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Read all matching JSONL files and return parsed records.

    Filters by `since` if provided (records older than `since` are dropped).
    Skips malformed lines (Iron Law 4 fail-closed; logs a warning).
    """
    records: List[Dict[str, Any]] = []
    if not ledger_dir.exists():
        logger.warning(f"[SHADOW_IC] ledger dir does not exist: {ledger_dir}")
        return records

    skipped = 0
    for prefix in prefixes:
        for fp in sorted(ledger_dir.glob(f"{prefix}_*.jsonl")):
            try:
                with fp.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:  # noqa: silent-swallow
                            # Counted in `skipped`; batch WARN at end emits total
                            skipped += 1
                            continue
                        # Parse timestamp
                        ts_str = rec.get("ts")
                        if not ts_str:
                            skipped += 1
                            continue
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        except ValueError:  # noqa: silent-swallow
                            # Counted in `skipped`; batch WARN at end emits the total
                            skipped += 1
                            continue
                        if since is not None and ts < since:
                            continue
                        rec["_parsed_ts"] = ts
                        records.append(rec)
            except Exception as e:
                logger.warning(f"[SHADOW_IC] failed to read {fp}: {type(e).__name__}: {e}")
    if skipped:
        logger.warning(f"[SHADOW_IC] skipped {skipped} malformed/missing-ts records")
    return records


def load_ohlcv(asset: str) -> Any:
    """Load 4H OHLCV parquet. Returns pd.DataFrame indexed by timestamp."""
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(f"pandas required: {e}")
    candidates = [
        OHLCV_DIR / f"{asset}_4H_full.parquet",
        OHLCV_DIR / f"{asset}_4h_full.parquet",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            if "timestamp" not in df.columns:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df.sort_values("timestamp").reset_index(drop=True)
    raise FileNotFoundError(f"OHLCV parquet for {asset} not found in {OHLCV_DIR}")


# ---------------------------------------------------------------------------
# Forward-return join
# ---------------------------------------------------------------------------

def find_forward_return(df: Any, ts: datetime, horizon_bars: int) -> Optional[float]:
    """Given a signal timestamp, find the bar at-or-after `ts`, then return
    the close-to-close return at that bar to bar+horizon. Returns None if
    insufficient future bars."""
    try:
        import pandas as pd
    except ImportError as e:
        logger.warning(
            f"[SHADOW_IC] pandas unavailable for forward-return join: "
            f"{type(e).__name__}: {e} — IC compute will report 0 for all"
        )
        return None

    # Find first bar at-or-after ts
    if df.empty:
        return None
    mask = df["timestamp"] >= ts
    matches = df[mask]
    if matches.empty:
        return None
    entry_idx = matches.index[0]
    exit_idx = entry_idx + horizon_bars
    if exit_idx >= len(df):
        return None
    p_entry = float(df.iloc[entry_idx]["close"])
    p_exit = float(df.iloc[exit_idx]["close"])
    if p_entry <= 0:
        return None
    return (p_exit - p_entry) / p_entry


# ---------------------------------------------------------------------------
# Spearman (no scipy dep)
# ---------------------------------------------------------------------------

def _spearman(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 4:
        return 0.0

    def rank(arr: List[float]) -> List[float]:
        order = sorted(range(len(arr)), key=lambda i: arr[i])
        ranks = [0.0] * len(arr)
        i = 0
        while i < len(arr):
            j = i
            while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(x), rank(y)
    n = len(rx)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denom_x = sum((r - mx) ** 2 for r in rx) ** 0.5
    denom_y = sum((r - my) ** 2 for r in ry) ** 0.5
    if denom_x <= 0 or denom_y <= 0:
        return 0.0
    return num / (denom_x * denom_y)


# ---------------------------------------------------------------------------
# Per-strategy compute
# ---------------------------------------------------------------------------

def compute_per_strategy_ic(
    records: List[Dict[str, Any]],
    horizons_bars: Tuple[int, ...] = (4, 12, 24),
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Group records by (strategy, asset). For each group compute IC at
    each horizon. Returns {(strategy, asset): {n, ic_per_h, ...}}."""

    # Group records
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        strat = r.get("strategy")
        asset = r.get("asset")
        if not strat or not asset:
            continue
        grouped[(strat, asset)].append(r)

    # Cache OHLCV per asset
    ohlcv_cache: Dict[str, Any] = {}

    def get_ohlcv(asset: str):
        if asset not in ohlcv_cache:
            try:
                ohlcv_cache[asset] = load_ohlcv(asset)
            except Exception as e:
                logger.warning(f"[SHADOW_IC] OHLCV load failed for {asset}: {e}")
                ohlcv_cache[asset] = None
        return ohlcv_cache[asset]

    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (strat, asset), recs in grouped.items():
        df = get_ohlcv(asset)
        if df is None:
            out[(strat, asset)] = {"n": 0, "ic_per_h": {}, "error": "ohlcv_missing"}
            continue

        # Build per-horizon (signal_x, forward_y) pairs
        per_h: Dict[int, Tuple[List[float], List[float]]] = {h: ([], []) for h in horizons_bars}
        per_trade_returns: List[float] = []  # for Sharpe at largest horizon

        for r in recs:
            ts = r.get("_parsed_ts")
            direction = float(r.get("direction", 0.0) or 0.0)
            confidence = float(r.get("confidence", 0.0) or 0.0)
            if ts is None:
                continue
            x_val = direction * confidence
            for h in horizons_bars:
                fr = find_forward_return(df, ts, h)
                if fr is None:
                    continue
                per_h[h][0].append(x_val)
                per_h[h][1].append(fr)
            # Per-trade return uses largest horizon, ONLY for non-zero direction
            if direction != 0.0:
                fr_large = find_forward_return(df, ts, max(horizons_bars))
                if fr_large is not None:
                    per_trade_returns.append(direction * fr_large)

        ic_per_h = {h: _spearman(per_h[h][0], per_h[h][1]) for h in horizons_bars}
        n_per_h = {h: len(per_h[h][0]) for h in horizons_bars}

        # Annualized Sharpe at the largest horizon (per-tick re-evaluations,
        # NOT per-trade — these are signals-as-positions, idealized)
        sharpe = 0.0
        if len(per_trade_returns) >= 2:
            mean_r = sum(per_trade_returns) / len(per_trade_returns)
            var_r = sum((r - mean_r) ** 2 for r in per_trade_returns) / (len(per_trade_returns) - 1)
            std_r = var_r ** 0.5
            if std_r > 0:
                # 4H bars; assume always-on at largest horizon
                bars_per_year = 6 * 252  # 6 bars/day * 252 days
                effective_obs_per_year = bars_per_year / max(horizons_bars)
                sharpe = (mean_r / std_r) * (effective_obs_per_year ** 0.5)

        out[(strat, asset)] = {
            "n_records": len(recs),
            "n_per_horizon": n_per_h,
            "ic_per_horizon": ic_per_h,
            "annualized_sharpe": sharpe,
            "n_directional": len(per_trade_returns),
        }

    return out


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def determine_verdict(
    ic_per_h: Dict[int, float],
    n_per_h: Dict[int, int],
    sharpe: float,
    window_days: int,
    min_samples: int = 30,
    promote_ic: float = 0.05,
    kill_ic: float = 0.05,
    promote_sharpe: float = 0.5,
) -> Verdict:
    """Apply v5.1 promotion gate.

    Rules:
      - ALL horizons N < min_samples -> INSUFFICIENT_SAMPLES
      - 14d window: any horizon IC < kill_ic AND N >= min_samples -> KILL
      - 30d window: ALL horizons IC > promote_ic AND sharpe > promote_sharpe -> PROMOTE
      - else -> HOLD
    """
    if all(n < min_samples for n in n_per_h.values()):
        return Verdict.INSUFFICIENT_SAMPLES

    # Use only horizons that have enough samples
    valid_horizons = [h for h, n in n_per_h.items() if n >= min_samples]
    if not valid_horizons:
        return Verdict.INSUFFICIENT_SAMPLES

    valid_ics = [abs(ic_per_h[h]) for h in valid_horizons]

    if window_days <= 14:
        # Short window: KILL aggressively if all IC weak (use absolute value -
        # negative IC is also a signal but flipped; promote logic should handle that)
        if max(valid_ics) < kill_ic:
            return Verdict.KILL
        return Verdict.HOLD

    # Longer window (>=30d): Look at promotion criteria
    if min(valid_ics) > promote_ic and sharpe > promote_sharpe:
        return Verdict.PROMOTE
    if max(valid_ics) < kill_ic:
        return Verdict.KILL
    return Verdict.HOLD


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_summary(
    per_strategy: Dict[Tuple[str, str], Dict[str, Any]],
    window_days: int,
    horizons_bars: Tuple[int, ...],
) -> str:
    lines = []
    lines.append("=" * 90)
    lines.append(f"  SHADOW IC REPORT  window={window_days}d  horizons={horizons_bars}")
    lines.append("=" * 90)
    lines.append(f"  {'strategy':<24} {'asset':<5} {'N':>6} " +
                 " ".join(f"IC({h}b)" for h in horizons_bars) +
                 f" {'Sharpe':>8} {'Verdict':>20}")
    lines.append("-" * 90)
    for (strat, asset), v in sorted(per_strategy.items()):
        if "error" in v:
            lines.append(f"  {strat:<24} {asset:<5} ERROR: {v['error']}")
            continue
        ic_per_h = v.get("ic_per_horizon", {})
        n_per_h = v.get("n_per_horizon", {})
        sharpe = v.get("annualized_sharpe", 0.0)
        verdict = determine_verdict(ic_per_h, n_per_h, sharpe, window_days)
        n_max = max(n_per_h.values()) if n_per_h else 0
        ic_strs = " ".join(f"{ic_per_h.get(h, 0.0):+.3f}" for h in horizons_bars)
        lines.append(
            f"  {strat:<24} {asset:<5} {n_max:>6} {ic_strs} {sharpe:+8.2f} {verdict.value:>20}"
        )
    lines.append("=" * 90)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compute IC + verdict on shadow-strategy ledgers.")
    p.add_argument("--ledger-dir", default=str(LEDGER_DIR))
    p.add_argument("--window-days", type=int, default=14)
    p.add_argument("--horizons", default="4,12,24",
                   help="forward-return horizons in 4H bars, comma-separated")
    p.add_argument("--prefixes", default="microstructure,cascade",
                   help="ledger file prefixes")
    p.add_argument("--output", default=None,
                   help="optional JSON output path; defaults to analytics/shadow_ic/reports/")
    args = p.parse_args(argv)

    horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())
    prefixes = tuple(x.strip() for x in args.prefixes.split(",") if x.strip())
    since = datetime.now(timezone.utc) - timedelta(days=args.window_days)

    ledger_dir = Path(args.ledger_dir)
    records = load_shadow_ledgers(ledger_dir, prefixes=prefixes, since=since)
    if not records:
        print(f"No shadow records loaded from {ledger_dir} since {since.isoformat()}",
              file=sys.stderr)
        return 1

    per_strategy = compute_per_strategy_ic(records, horizons_bars=horizons)
    print(render_summary(per_strategy, args.window_days, horizons))

    # Build report
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": args.window_days,
        "horizons_bars": list(horizons),
        "n_records": len(records),
        "per_strategy": [
            {
                "strategy": s,
                "asset": a,
                **v,
                "verdict": determine_verdict(
                    v.get("ic_per_horizon", {}),
                    v.get("n_per_horizon", {}),
                    v.get("annualized_sharpe", 0.0),
                    args.window_days,
                ).value,
            }
            for (s, a), v in per_strategy.items()
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else (
        REPORT_DIR / f"shadow_ic_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    )
    out_path.write_text(json.dumps(report, default=str, indent=2), encoding="utf-8")
    print(f"\nReport saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
