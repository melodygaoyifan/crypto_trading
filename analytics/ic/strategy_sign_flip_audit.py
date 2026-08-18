#!/usr/bin/env python3
"""
Strategy sign-flip audit — quantify how many backfilled signals show
regime-conditional sign flips (per P40 finding for RSI).

User prediction: of 12 kraken_quant strategies, 5-6 should have
regime-conditional sign flip not currently handled.

We can't backfill the 12 kraken_quant strategies (cointegration / Hurst /
ETF basis need live cross-asset data), but we DO have backfill data for
4 price-derivable signal families:
  - quant_backfill (RSI-based contrarian + a derived direction)
  - momentum_backfill (24h return z-score)
  - meanrev_backfill (price z-score, contrarian)
  - bb_backfill (Bollinger band position)

If 5+ of THESE 8 signal columns sign-flip, that's evidence the pattern
extends — supporting the hypothesis for the unobservable 12.

Usage:
    python analytics/ic/strategy_sign_flip_audit.py --all-assets --horizon 12
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
IC_LOG_DIR = REPO / "logs" / "ic_signals"
OHLCV_DIR = REPO / "training" / "training_data" / "drl_training"
REPORT_DIR = REPO / "analytics" / "ic" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REGIME_NAMES = {
    0: "STEADY_UPTREND",
    1: "WEAK_CONSOLIDATION",
    2: "QUIET_ACCUMULATION",
    3: "TRENDING_BULL",
    4: "TRENDING_BEAR",
    5: "VOLATILE_CHOP",
    6: "REVERSAL",
    7: "DISTRIBUTION",
}


def _load(asset: str, horizon: int) -> pd.DataFrame:
    f = IC_LOG_DIR / f"ic_signals_BACKFILL_{asset}.jsonl"
    rows = [json.loads(l) for l in f.open() if l.strip()]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    ohlcv = pd.read_parquet(OHLCV_DIR / f"{asset}_4H_full.parquet")
    if "timestamp" in ohlcv.columns:
        ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)

    df = pd.merge_asof(
        df,
        ohlcv[["timestamp", "close", "regime"]].rename(
            columns={"close": "price_at_log", "regime": "regime_id"}
        ),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("4h"),
    )

    # forward returns
    ts_to_close = ohlcv.set_index("timestamp")["close"]
    fp = []
    for t in df["timestamp"]:
        target_t = t + pd.Timedelta(hours=4 * horizon)
        try:
            pos = ts_to_close.index.get_indexer([target_t], method="nearest")[0]
            ft = ts_to_close.index[pos]
            if abs((ft - target_t).total_seconds()) <= 4 * 3600:
                fp.append(float(ts_to_close.iloc[pos]))
            else:
                fp.append(np.nan)
        except Exception:
            fp.append(np.nan)
    df[f"future_price_{horizon}b"] = fp
    df[f"future_return_{horizon}b"] = (
        (df[f"future_price_{horizon}b"] - df["price_at_log"]) / df["price_at_log"]
    )
    return df


def _extract(row: pd.Series, path: str):
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


def _signal_paths(df: pd.DataFrame) -> List[str]:
    seen = set()
    for rec in df["agent_signals"].head(50):
        if not isinstance(rec, dict):
            continue
        for k1, sub in rec.items():
            if isinstance(sub, dict):
                for k2, v in sub.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        seen.add(f"{k1}.{k2}")
    return sorted(seen)


def per_regime_ic(
    df: pd.DataFrame, signal_path: str, horizon: int
) -> Dict[str, Tuple[float, float, int]]:
    """Return {regime_name: (ic, p, n)} for each regime."""
    s_full = df.apply(lambda r: _extract(r, signal_path), axis=1)
    out = {}
    for rid in sorted(df["regime_id"].dropna().unique()):
        mask = df["regime_id"] == rid
        s = s_full[mask]
        r = df.loc[mask, f"future_return_{horizon}b"]
        valid = s.notna() & r.notna()
        n = int(valid.sum())
        if n < 30:
            continue
        ic, p = stats.spearmanr(s[valid], r[valid])
        out[REGIME_NAMES.get(int(rid), f"REGIME_{int(rid)}")] = (
            float(ic) if not np.isnan(ic) else 0.0,
            float(p) if not np.isnan(p) else 1.0,
            n,
        )
    return out


def classify_sign_flip(per_reg: Dict, ic_threshold: float = 0.03, p_threshold: float = 0.10) -> Dict:
    """
    Identify sign flip across regimes.
      pos_regimes: regimes where IC > +threshold AND p < p_threshold
      neg_regimes: regimes where IC < -threshold AND p < p_threshold
      sign_flip: both pos_regimes and neg_regimes are non-empty
    """
    pos = []
    neg = []
    for regime, (ic, p, n) in per_reg.items():
        if ic > ic_threshold and p < p_threshold:
            pos.append((regime, ic, p, n))
        elif ic < -ic_threshold and p < p_threshold:
            neg.append((regime, ic, p, n))
    return {
        "pos_regimes": pos,
        "neg_regimes": neg,
        "has_sign_flip": (len(pos) > 0 and len(neg) > 0),
        "pos_count": len(pos),
        "neg_count": len(neg),
    }


def audit_asset(asset: str, horizon: int) -> Dict:
    df = _load(asset, horizon)
    paths = _signal_paths(df)
    print(f"\n{'='*92}")
    print(f"  ASSET: {asset}  (n_records={len(df)}, horizon={horizon}b, signals={len(paths)})")
    print(f"{'='*92}")
    print(f"  {'Signal':<35} {'+ regimes (IC > +0.03)':<28} {'- regimes (IC < -0.03)':<28} {'flip?':<6}")
    print(f"  {'-'*35} {'-'*28} {'-'*28} {'-'*6}")

    results = {}
    for path in paths:
        per_reg = per_regime_ic(df, path, horizon)
        cls = classify_sign_flip(per_reg)
        # short-form regimes for table
        pos_short = ",".join(r[0][:6] for r in cls["pos_regimes"][:3])
        neg_short = ",".join(r[0][:6] for r in cls["neg_regimes"][:3])
        flip = "YES" if cls["has_sign_flip"] else "no"
        print(f"  {path:<35} {pos_short:<28} {neg_short:<28} {flip:<6}")
        results[path] = {
            "per_regime": {
                rg: {"ic": ic, "p": p, "n": n}
                for rg, (ic, p, n) in per_reg.items()
            },
            "classification": cls,
        }

    n_with_flip = sum(1 for v in results.values() if v["classification"]["has_sign_flip"])
    print(f"\n  Sign-flip count: {n_with_flip} / {len(paths)} signals show regime-conditional sign flip")
    return {
        "asset": asset,
        "n_signals": len(paths),
        "n_with_sign_flip": n_with_flip,
        "signals": results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default=None)
    p.add_argument("--all-assets", action="store_true")
    p.add_argument("--horizon", type=int, default=12)
    args = p.parse_args()

    assets = ["BTC", "ETH", "SOL"] if args.all_assets else [args.asset or "BTC"]
    full: Dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "assets": {}}

    for asset in assets:
        try:
            full["assets"][asset] = audit_asset(asset, args.horizon)
        except Exception as e:
            print(f"  [{asset}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    # Cross-asset summary
    print(f"\n{'='*92}")
    print(f"  CROSS-ASSET SUMMARY (horizon={args.horizon}b)")
    print(f"{'='*92}")
    print(f"  {'Asset':<8} {'Signals':<10} {'With Sign-Flip':<18} {'Pct':<8}")
    print(f"  {'-'*8} {'-'*10} {'-'*18} {'-'*8}")
    total_signals = 0
    total_flips = 0
    for asset, rep in full["assets"].items():
        if "n_signals" in rep:
            n = rep["n_signals"]
            f = rep["n_with_sign_flip"]
            pct = 100.0 * f / n if n else 0.0
            print(f"  {asset:<8} {n:<10} {f:<18} {pct:<7.1f}%")
            total_signals += n
            total_flips += f
    if total_signals:
        print(f"  {'TOTAL':<8} {total_signals:<10} {total_flips:<18} {100.0*total_flips/total_signals:<7.1f}%")

    # Per-signal aggregation: which signals consistently sign-flip across assets
    print()
    print(f"  SIGNALS THAT SIGN-FLIP IN ≥2 OF {len(full['assets'])} ASSETS:")
    per_signal_flips: Dict[str, List[str]] = {}
    for asset, rep in full["assets"].items():
        for sig, info in rep.get("signals", {}).items():
            if info["classification"]["has_sign_flip"]:
                per_signal_flips.setdefault(sig, []).append(asset)
    if not per_signal_flips:
        print("    (none)")
    else:
        for sig, assets_with_flip in sorted(per_signal_flips.items()):
            if len(assets_with_flip) >= 2:
                print(f"    {sig:<35} → {','.join(assets_with_flip)}")

    out = REPORT_DIR / f"strategy_sign_flip_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    out.write_text(json.dumps(full, indent=2, default=str), encoding="utf-8")
    print(f"\n  Report: {out}")


if __name__ == "__main__":
    main()
