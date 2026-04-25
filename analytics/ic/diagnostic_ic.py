#!/usr/bin/env python3
"""
P0 diagnostic — explain the gap between backfill IC and live alpha.

Three analyses:

1. Conditional IC by regime — answers "RSI 在 WEAK_CONSOLIDATION 下还有 IC 吗?"
   Splits backfill IC by per-bar GMM regime ID and recomputes per-regime IC.
   If a signal's IC is +0.08 in TRENDING regimes but ~0 in CONSOLIDATION,
   that tells us why live (consolidation-heavy) trading sees no alpha.

2. Walk-forward IC — answers "Is the backfill IC stable over time, or
   is it an in-sample artifact?"
   Splits the timeline into 5 chunks (~10mo each) and recomputes IC per chunk.
   High variance across chunks → IC is unstable → backfill IC is unreliable.

3. Live RSI signal trace — pulls the last 24h of QUANT_SIGNAL log lines from
   Hetzner and shows the RSI breakdown that fed each tick's quant direction.

Usage:
    python analytics/ic/diagnostic_ic.py --task 1     # regime-conditional
    python analytics/ic/diagnostic_ic.py --task 2     # walk-forward
    python analytics/ic/diagnostic_ic.py --task all
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
IC_LOG_DIR = REPO / "logs" / "ic_signals"
OHLCV_DIR = REPO / "training" / "training_data" / "drl_training"
REPORT_DIR = REPO / "analytics" / "ic" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Per-asset GMM regime label maps (from models/regime_classifier/*/gmm_config.json
# documented in CLAUDE.md §GMM Per-Asset)
REGIME_NAME_FALLBACK = {
    0: "STEADY_UPTREND",
    1: "WEAK_CONSOLIDATION",
    2: "QUIET_ACCUMULATION",
    3: "TRENDING_BULL",
    4: "TRENDING_BEAR",
    5: "VOLATILE_CHOP",
    6: "REVERSAL",
    7: "DISTRIBUTION",
}


def load_regime_map(asset: str) -> Dict[int, str]:
    cfg = REPO / "models" / "regime_classifier" / asset / "gmm_config.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            names = data.get("regime_names") or data.get("REGIME_NAMES") or {}
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
        except Exception:
            pass
    return REGIME_NAME_FALLBACK


def load_ohlcv_with_regime(asset: str) -> pd.DataFrame:
    p = OHLCV_DIR / f"{asset}_4H_full.parquet"
    df = pd.read_parquet(p)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif df.index.name == "timestamp":
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_backfill(asset: str) -> pd.DataFrame:
    f = IC_LOG_DIR / f"ic_signals_BACKFILL_{asset}.jsonl"
    if not f.exists():
        # Fallback: regenerate via backfill_ic.py
        raise FileNotFoundError(
            f"Backfill missing: {f}. Run analytics/ic/backfill_ic.py --asset {asset} --days 365"
        )
    rows = []
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def merge_with_regime_and_returns(
    ic_df: pd.DataFrame, asset: str, horizons: List[int]
) -> pd.DataFrame:
    """Join backfill records with OHLCV regime + future returns."""
    ohlcv = load_ohlcv_with_regime(asset)
    keep = ["timestamp", "close", "regime"]
    # Rename to avoid collision with IC log's own `regime` dict column
    ohlcv_sub = ohlcv[keep].rename(columns={"close": "price_at_log", "regime": "regime_id"})
    merged = pd.merge_asof(
        ic_df.sort_values("timestamp"),
        ohlcv_sub.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("4h"),
    )
    # Use the parquet's regime_id (integer 0-7) for splits
    merged["regime"] = merged["regime_id"]
    ts_to_close = ohlcv.set_index("timestamp")["close"]
    for h in horizons:
        future_prices = []
        for t in merged["timestamp"]:
            target_t = t + timedelta(hours=4 * h)
            try:
                pos = ts_to_close.index.get_indexer([target_t], method="nearest")[0]
                ft = ts_to_close.index[pos]
                if abs((ft - target_t).total_seconds()) <= 4 * 3600:
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


def extract_signal(row: pd.Series, path: str) -> Optional[float]:
    obj = row.get("agent_signals")
    if not isinstance(obj, dict):
        return None
    for p in path.split("."):
        if not isinstance(obj, dict) or p not in obj:
            return None
        obj = obj[p]
    try:
        v = float(obj)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def discover_paths(df: pd.DataFrame, sample: int = 50) -> List[str]:
    paths = set()
    for rec in df["agent_signals"].head(sample):
        if not isinstance(rec, dict):
            continue
        for k1, sub in rec.items():
            if isinstance(sub, dict):
                for k2, v in sub.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        paths.add(f"{k1}.{k2}")
    return sorted(paths)


def ic_stats(s: pd.Series, r: pd.Series) -> Dict:
    valid = s.notna() & r.notna()
    n = int(valid.sum())
    if n < 30:
        return {"n": n, "ic": None, "p": None}
    sr, sp = stats.spearmanr(s[valid], r[valid])
    return {"n": n, "ic": float(sr) if not np.isnan(sr) else None,
            "p": float(sp) if not np.isnan(sp) else None}


# ──────────────────────────────────────────────────────────────────────
# Task 1: regime-conditional IC
# ──────────────────────────────────────────────────────────────────────
def task1_regime_conditional(asset: str, horizon: int = 12) -> Dict:
    print(f"\n{'='*70}")
    print(f"  TASK 1: regime-conditional IC for {asset} (horizon={horizon}b)")
    print(f"{'='*70}")
    df = load_backfill(asset)
    df = merge_with_regime_and_returns(df, asset, [horizon])
    name_map = load_regime_map(asset)

    print(f"  Total bars: {len(df)} | regime distribution:")
    counts = df["regime"].value_counts().sort_index()
    for rid, cnt in counts.items():
        rname = name_map.get(int(rid), f"REGIME_{int(rid)}")
        pct = 100.0 * cnt / len(df)
        print(f"    {int(rid)} ({rname:<22}): {cnt:>5} bars ({pct:5.1f}%)")

    paths = discover_paths(df)
    results = {}
    print(f"\n  IC by (signal × regime), horizon={horizon}b:")
    print(f"  {'Signal':<35} {'Regime':<24} {'N':>6} {'IC':>8} {'p':>7}")
    print(f"  {'-'*35} {'-'*24} {'-'*6} {'-'*8} {'-'*7}")
    for path in paths:
        s_full = df.apply(lambda r: extract_signal(r, path), axis=1)
        for rid in sorted(df["regime"].dropna().unique()):
            mask = df["regime"] == rid
            sub_s = s_full[mask]
            sub_r = df.loc[mask, f"future_return_{horizon}b"]
            stats_d = ic_stats(sub_s, sub_r)
            rname = name_map.get(int(rid), f"REGIME_{int(rid)}")
            ic = stats_d["ic"]
            p = stats_d["p"]
            ic_s = f"{ic:+.3f}" if ic is not None else "  ---"
            p_s = f"{p:.3f}" if p is not None else "  ---"
            print(f"  {path:<35} {rname:<24} {stats_d['n']:>6} {ic_s:>8} {p_s:>7}")
            results.setdefault(path, {})[rname] = stats_d
    return results


# ──────────────────────────────────────────────────────────────────────
# Task 2: walk-forward IC (5 chunks)
# ──────────────────────────────────────────────────────────────────────
def task2_walk_forward(asset: str, horizon: int = 12, n_chunks: int = 5) -> Dict:
    print(f"\n{'='*70}")
    print(f"  TASK 2: walk-forward IC for {asset} ({n_chunks} chunks, horizon={horizon}b)")
    print(f"{'='*70}")
    df = load_backfill(asset)
    df = merge_with_regime_and_returns(df, asset, [horizon])
    paths = discover_paths(df)

    chunks = np.array_split(df, n_chunks)
    print(f"  Total bars: {len(df)} → {n_chunks} chunks of ~{len(df)//n_chunks} bars")
    for i, c in enumerate(chunks):
        if len(c):
            print(f"    Chunk {i+1}: {len(c)} bars  "
                  f"({pd.to_datetime(c['timestamp'].iloc[0]).strftime('%Y-%m')}"
                  f" → {pd.to_datetime(c['timestamp'].iloc[-1]).strftime('%Y-%m')})")

    print(f"\n  Per-chunk IC (horizon={horizon}b):")
    print(f"  {'Signal':<35} " + "".join(f"{'C'+str(i+1):>9}" for i in range(n_chunks)) + f"{'Mean':>9}{'Std':>8}{'Verdict':>15}")
    print(f"  {'-'*35} " + "  ".join("-"*7 for _ in range(n_chunks)) + "  -------  ------  --------------")

    results = {}
    for path in paths:
        ics: List[Optional[float]] = []
        for chunk in chunks:
            s = chunk.apply(lambda r: extract_signal(r, path), axis=1)
            r_ = chunk[f"future_return_{horizon}b"]
            stats_d = ic_stats(s, r_)
            ics.append(stats_d["ic"])
        valid_ics = [x for x in ics if x is not None]
        if not valid_ics:
            continue
        mean_ic = float(np.mean(valid_ics))
        std_ic = float(np.std(valid_ics))
        # Stability verdict
        if std_ic < 0.02 and abs(mean_ic) > 0.03:
            verdict = "STABLE"
        elif std_ic < 0.05 and abs(mean_ic) > 0.03:
            verdict = "MODERATELY_STABLE"
        elif std_ic >= abs(mean_ic) * 2:
            verdict = "UNSTABLE"
        else:
            verdict = "MIXED"
        chunk_str = "".join(
            f"  {ic:+.3f}" if ic is not None else "  ---  "
            for ic in ics
        )
        print(f"  {path:<35} {chunk_str}  {mean_ic:+.3f}  {std_ic:.3f}  {verdict:>15}")
        results[path] = {"per_chunk": ics, "mean": mean_ic, "std": std_ic, "verdict": verdict}
    return results


# ──────────────────────────────────────────────────────────────────────
# Task 3: live RSI flow trace
# ──────────────────────────────────────────────────────────────────────
def task3_live_rsi_trace(host: str = "hmats", hours: int = 24) -> None:
    """Pull QUANT_SIGNAL log lines from the live engine, decompose RSI flow."""
    import subprocess
    print(f"\n{'='*70}")
    print(f"  TASK 3: live RSI signal trace (last {hours}h on {host})")
    print(f"{'='*70}")
    # Pull QUANT_SIGNAL lines (have RSI breakdown), DECIDE_POOL/CONSENSUS
    cmd = (
        f"docker logs hmats-engine --since {hours}h 2>&1 | "
        f"grep -E 'QUANT_SIGNAL|STRATEGY_SELECT|DECIDE_CONSENSUS|DECIDE_SOLO' | tail -40"
    )
    try:
        out = subprocess.check_output(["ssh", host, cmd], stderr=subprocess.STDOUT, timeout=60)
        text = out.decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  ssh fetch failed: {e}")
        return
    if not text.strip():
        print("  No QUANT_SIGNAL log lines in window")
        return
    print("  Last 40 fusion-relevant lines:\n")
    print("\n".join("  " + line for line in text.splitlines() if line.strip()))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all", choices=["1", "2", "3", "all"])
    p.add_argument("--asset", default="BTC")
    p.add_argument("--all-assets", action="store_true")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--chunks", type=int, default=5)
    p.add_argument("--host", default="hmats")
    args = p.parse_args()

    assets = ["BTC", "ETH", "SOL"] if args.all_assets else [args.asset]
    full_report: Dict = {"generated_at": datetime.now(timezone.utc).isoformat()}

    for asset in assets:
        full_report[asset] = {}
        if args.task in ("1", "all"):
            try:
                full_report[asset]["task1_regime_conditional"] = task1_regime_conditional(asset, args.horizon)
            except Exception as e:
                print(f"  [{asset}] task1 failed: {type(e).__name__}: {e}")
        if args.task in ("2", "all"):
            try:
                full_report[asset]["task2_walk_forward"] = task2_walk_forward(asset, args.horizon, args.chunks)
            except Exception as e:
                print(f"  [{asset}] task2 failed: {type(e).__name__}: {e}")

    if args.task in ("3", "all"):
        task3_live_rsi_trace(args.host, hours=24)

    out = REPORT_DIR / f"diagnostic_ic_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    out.write_text(json.dumps(full_report, indent=2, default=str))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
