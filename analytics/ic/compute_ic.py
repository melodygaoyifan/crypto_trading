#!/usr/bin/env python3
"""
Offline IC computation: read ic_signals JSONL + historical OHLCV → IC report.

Usage:
    python analytics/ic/compute_ic.py --asset BTC --horizons 4,12,24
    python analytics/ic/compute_ic.py --all-assets
    python analytics/ic/compute_ic.py --start-date 2026-04-01

IC verdict thresholds (per spec):
  IC ≥ 0.10        STRONG
  IC ≥ 0.05        MODERATE
  IC ≥ 0.02        WEAK
  IC ∈ (-0.02,0.02) NOISE
  IC ≤ -0.05       INVERTED_SIGNAL
  p > 0.05         NOT_SIGNIFICANT

Output:
  analytics/ic/reports/ic_report_{utc_ts}.json
  + console human-readable summary table
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
IC_LOG_DIR = REPO / "logs" / "ic_signals"
OHLCV_DIR = REPO / "training" / "training_data" / "drl_training"
REPORT_DIR = REPO / "analytics" / "ic" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────
# Loaders
# ────────────────────────────────────────────────────────────────
def load_ic_logs(start_date: Optional[str] = None) -> pd.DataFrame:
    """Load all ic_signals JSONL files into a DataFrame."""
    rows: List[Dict] = []
    if not IC_LOG_DIR.exists():
        raise FileNotFoundError(f"IC log dir does not exist: {IC_LOG_DIR}")
    for f in sorted(IC_LOG_DIR.glob("ic_signals_*.jsonl")):
        # Skip files older than start_date (file-name based filter)
        if start_date:
            try:
                date_part = f.stem.replace("ic_signals_", "")
                if not date_part.startswith("BACKFILL_") and date_part < start_date:
                    continue
            except Exception:
                pass
        try:
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[WARN] failed to read {f.name}: {e}", file=sys.stderr)
    if not rows:
        raise FileNotFoundError(f"No IC records loaded from {IC_LOG_DIR}")
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    return df


def load_ohlcv(asset: str) -> pd.DataFrame:
    """Load 4H OHLCV parquet for an asset."""
    candidates = [
        OHLCV_DIR / f"{asset}_4H_full.parquet",
        OHLCV_DIR / f"{asset}_4h_full.parquet",
        REPO / "data" / f"{asset}_4H.parquet",
    ]
    for c in candidates:
        if c.exists():
            df = pd.read_parquet(c)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
            elif df.index.name == "timestamp":
                df = df.reset_index()
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
            return df
    raise FileNotFoundError(f"No OHLCV parquet for {asset} (tried {[str(c) for c in candidates]})")


# ────────────────────────────────────────────────────────────────
# Future-return computation
# ────────────────────────────────────────────────────────────────
def compute_future_returns(
    ic_df: pd.DataFrame, asset: str, horizons: List[int]
) -> pd.DataFrame:
    """
    For each row in ic_df, look up the future N-bar close from OHLCV and
    compute the return.
    """
    sub = ic_df[ic_df["asset"] == asset].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values("timestamp").reset_index(drop=True)

    ohlcv = load_ohlcv(asset).sort_values("timestamp").reset_index(drop=True)

    # Align each IC row to the nearest OHLCV bar
    merged = pd.merge_asof(
        sub,
        ohlcv[["timestamp", "close"]].rename(columns={"close": "price_at_log"}),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("4h"),
    )

    ts_to_close = ohlcv.set_index("timestamp")["close"]
    for h in horizons:
        future_prices: List[float] = []
        for t in merged["timestamp"]:
            target_t = t + timedelta(hours=4 * h)
            try:
                pos = ts_to_close.index.get_indexer([target_t], method="nearest")[0]
                future_t = ts_to_close.index[pos]
                if abs((future_t - target_t).total_seconds()) <= 4 * 3600:
                    future_prices.append(float(ts_to_close.iloc[pos]))
                else:
                    future_prices.append(np.nan)
            except Exception:
                future_prices.append(np.nan)
        merged[f"future_price_{h}b"] = future_prices
        merged[f"future_return_{h}b"] = (
            (merged[f"future_price_{h}b"] - merged["price_at_log"])
            / merged["price_at_log"]
        )
    return merged


# ────────────────────────────────────────────────────────────────
# IC math
# ────────────────────────────────────────────────────────────────
def extract_signal(row: pd.Series, signal_path: str) -> Optional[float]:
    """Extract a numeric value via dotted path 'agent.field' from agent_signals."""
    obj = row.get("agent_signals")
    if not isinstance(obj, dict):
        return None
    for p in signal_path.split("."):
        if not isinstance(obj, dict) or p not in obj:
            return None
        obj = obj[p]
    try:
        v = float(obj)
        if not np.isfinite(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def discover_signal_paths(df: pd.DataFrame, sample_rows: int = 100) -> List[str]:
    """Find every numeric `agent.field` path that appears in any row."""
    paths = set()
    for record in df["agent_signals"].head(sample_rows):
        if not isinstance(record, dict):
            continue
        for agent_name, agent_dict in record.items():
            if not isinstance(agent_dict, dict):
                continue
            for key, value in agent_dict.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    paths.add(f"{agent_name}.{key}")
    return sorted(paths)


def _ic_verdict(ic: Optional[float], pvalue: Optional[float]) -> str:
    if ic is None or pvalue is None or np.isnan(ic):
        return "NO_DATA"
    if pvalue > 0.05:
        return "NOT_SIGNIFICANT"
    if abs(ic) < 0.02:
        return "NOISE"
    if ic <= -0.05:
        return "INVERTED_SIGNAL"
    if ic >= 0.10:
        return "STRONG"
    if ic >= 0.05:
        return "MODERATE"
    if ic >= 0.02:
        return "WEAK"
    return "MARGINAL"


def compute_ic(df: pd.DataFrame, signal_path: str, horizon: int) -> Dict:
    sigs = df.apply(lambda r: extract_signal(r, signal_path), axis=1)
    rets = df.get(f"future_return_{horizon}b", pd.Series([np.nan] * len(df)))
    valid = sigs.notna() & rets.notna()
    n = int(valid.sum())
    if n < 30:
        return {
            "signal": signal_path,
            "horizon_bars": horizon,
            "n_samples": n,
            "ic_spearman": None,
            "ic_pearson": None,
            "ic_pvalue": None,
            "ic_ir": None,
            "verdict": "INSUFFICIENT_SAMPLES",
        }
    s = sigs[valid].astype(float)
    r = rets[valid].astype(float)
    sr, sp = stats.spearmanr(s, r)
    pr, _pp = stats.pearsonr(s, r)

    # IC IR via chunk decomposition
    n_chunks = max(4, n // 200)
    chunk_size = max(20, n // n_chunks)
    ic_per_chunk: List[float] = []
    for i in range(n_chunks):
        a = i * chunk_size
        b = (i + 1) * chunk_size if i < n_chunks - 1 else n
        if b - a >= 20:
            try:
                cic, _ = stats.spearmanr(s.iloc[a:b], r.iloc[a:b])
                if cic is not None and not np.isnan(cic):
                    ic_per_chunk.append(float(cic))
            except Exception:
                pass
    if len(ic_per_chunk) > 1 and float(np.std(ic_per_chunk)) > 1e-12:
        ic_ir = float(np.mean(ic_per_chunk) / np.std(ic_per_chunk))
    else:
        ic_ir = None

    return {
        "signal": signal_path,
        "horizon_bars": horizon,
        "n_samples": n,
        "ic_spearman": float(sr) if not np.isnan(sr) else None,
        "ic_pearson": float(pr) if not np.isnan(pr) else None,
        "ic_pvalue": float(sp) if not np.isnan(sp) else None,
        "ic_ir": ic_ir,
        "verdict": _ic_verdict(float(sr) if not np.isnan(sr) else None, float(sp) if not np.isnan(sp) else None),
    }


# ────────────────────────────────────────────────────────────────
# Report generator
# ────────────────────────────────────────────────────────────────
def generate_report(asset: str, horizons: List[int], start_date: Optional[str]) -> Dict:
    print(f"\n{'=' * 70}")
    print(f"  IC REPORT: {asset}")
    print(f"{'=' * 70}")
    ic_df = load_ic_logs(start_date)
    print(f"  Total IC log records: {len(ic_df)}")
    df = compute_future_returns(ic_df, asset, horizons)
    if df.empty:
        return {"asset": asset, "error": "no records for asset"}
    print(f"  Records for {asset}: {len(df)}")
    paths = discover_signal_paths(df)
    print(f"  Discovered {len(paths)} signal paths")
    results = [compute_ic(df, p, h) for p in paths for h in horizons]
    return {
        "asset": asset,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_records": len(df),
        "n_signals": len(paths),
        "horizons_bars": horizons,
        "signals": results,
    }


def print_summary(report: Dict) -> None:
    if "error" in report:
        print(f"\n[{report.get('asset','?')}] ERROR: {report['error']}")
        return
    print(f"\n{'─' * 90}")
    print(f"  IC SUMMARY: {report['asset']} ({report['n_records']} records, {report['n_signals']} signals)")
    print(f"{'─' * 90}")
    print(f"  {'Signal':<35} {'H':>4} {'N':>6} {'IC':>8} {'p':>7} {'IR':>7}  Verdict")
    print(f"  {'-' * 35} {'-' * 4} {'-' * 6} {'-' * 8} {'-' * 7} {'-' * 7}  {'-' * 25}")
    sigs = sorted(
        report["signals"],
        key=lambda s: abs(s.get("ic_spearman") or 0.0),
        reverse=True,
    )
    for s in sigs:
        ic = s.get("ic_spearman")
        p = s.get("ic_pvalue")
        ir = s.get("ic_ir")
        ic_s = f"{ic:+.3f}" if ic is not None else "  ---"
        p_s = f"{p:.3f}" if p is not None else "  ---"
        ir_s = f"{ir:+.2f}" if ir is not None else " ---"
        print(
            f"  {s['signal']:<35} {s['horizon_bars']:>3}b "
            f"{s['n_samples']:>6} {ic_s:>8} {p_s:>7} {ir_s:>7}  {s['verdict']}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default=None)
    p.add_argument("--all-assets", action="store_true")
    p.add_argument("--horizons", default="4,12,24",
                   help="comma-separated future horizons in 4H bars")
    p.add_argument("--start-date", default=None,
                   help="YYYY-MM-DD: only consider IC log files >= this date")
    args = p.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    assets = ["BTC", "ETH", "SOL"] if args.all_assets else [args.asset or "BTC"]

    all_reports: Dict[str, Dict] = {}
    for asset in assets:
        try:
            rpt = generate_report(asset, horizons, args.start_date)
            print_summary(rpt)
            all_reports[asset] = rpt
        except Exception as e:
            print(f"[{asset}] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            all_reports[asset] = {"asset": asset, "error": str(e)}

    out = REPORT_DIR / f"ic_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    out.write_text(json.dumps(all_reports, indent=2, default=str), encoding="utf-8")
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
