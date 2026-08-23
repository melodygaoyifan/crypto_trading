"""[P384] CORRELATION_COLLAPSE on the sleeve: should the book FLATTEN or HOLD?

THE QUESTION. P383 ported the two missing conjuncts into the live
`defense/constitution.py::NoTradeTriggerChecker`, so CORRELATION_COLLAPSE now
fires iff
    correlation_btc_eth_sol >= CORRELATION_COLLAPSE_THRESHOLD (0.92)
    AND all three of BTC/ETH/SOL carry the SAME sign with
        |dir| > CORRELATION_ALIGNMENT_DIRECTION_MIN (0.2)
    AND not has_validated_edge (no producer anywhere -> always False).
In main.py the trigger sits in `_SLEEVE_HOLD_NO_TRADE_TRIGGERS`: when it
fires the Coinbase sleeve HOLDS its position (no new risk, nothing
liquidated). P382 classified it HOLD while the trigger was vacuous (corr
alone, 7.8% of bars). With the conjuncts live the trigger means "crowded,
correlated, same-direction book with no validated edge" -- and whether the
sleeve should FLATTEN on that instead of holding is a live-money semantics
change P383 recorded and did not make. This lab measures it on six years.

WHAT IS REPLICATED, at the call site (P228):
  * correlation: `data_mgmt/market_data_pipeline.py:~1470,~1535-1552` --
    pct returns of the last closes (`np.diff(close)/prev`, prev==0 -> 1.0),
    the LAST 20 returns, mean of the three pairwise `np.corrcoef`, each pair
    falling back to 0.85 (BTC/ETH) / 0.80 (BTC/SOL, ETH/SOL) when either leg
    has zero std, clipped to [-1, 1].
  * direction: the sleeve sizes by the SIGN of `_last_quant_directions`,
    which under `regimebook_mode: enforce` is the regimebook target
    (main.py ~:10880) -- and `cross_asset_directions` is built from exactly
    that dict (main.py ~:13399). So the per-asset direction here is the
    deterministic book target from `training.funding_legs_lab.build_positions`
    (BTC full book incl. funding legs, ETH trend-only, SOL hold-bull).
    The ETH/SOL books never go short, so "all three same-signed" can only be
    "all three LONG". The whale seat (P298) can fill a FLAT book cell live;
    that is not replicable here and is recorded as a caveat.
  * conjunct semantics + constants are IMPORTED from the live checker (the
    lab cannot drift from it): strict `> +m` / `< -m` on every asset.
  * economics: `funding_legs_lab.pnl` -- honest CDE per-leg cost (P315/P334),
    turnover charged on EVERY transition (so a FLATTEN pays the exit AND the
    re-entry), funding carry on every held bar (P245).

THE COUNTERFACTUAL. HOLD = the book as-is (what the sleeve does today).
FLATTEN = the book's target forced to 0 on every fired bar, re-entering at
the book's target the bar the mask clears. Reported as FLATTEN-HOLD in
summed per-bar return units (x100 = percentage points, the repo convention),
overall, per era and per asset, with a block-bootstrap CI90 on the per-bar
difference (block 90 bars, 2000 resamples, seed 7) and a turnover-matched
RANDOM-MASK control: the same number of fired bars arranged as random
non-overlapping contiguous episodes with the SAME length distribution, 200
seeds, giving the distribution of FLATTEN-HOLD under no information.

PRE-COMMITTED VERDICT RULE (fixed BEFORE the first run; the string below is
what the report carries and what tests/test_p384_corr_collapse_lab.py pins):
"""
from __future__ import annotations

VERDICT_RULE = (
    "FLATTEN EARNS iff, for the THREE-ASSET sleeve book summed: "
    "FLATTEN-HOLD net > 0 in the DESIGN era AND FLATTEN-HOLD net > 0 in the "
    "PRE-DESIGN era AND in BOTH eras it exceeds the 90th percentile of the "
    "turnover-matched random-mask control (200 seeds); otherwise HOLD STANDS. "
    "If the mask never fires the trigger is UNREACHABLE and HOLD/FLATTEN is moot."
)

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from defense.constitution import NoTradeTriggerChecker  # noqa: E402
from training.funding_legs_lab import (  # noqa: E402  [P172] single source
    ERAS,
    FUNDING_DIR,
    PRICE_DIR,
    build_positions,
    load_closes,
    load_funding_daily,
    pnl,
)

ASSETS = ("BTC", "ETH", "SOL")
CORR_WINDOW = 20                       # market_data_pipeline: `_btc_c["rets"][-20:]`
CORR_THRESHOLD = float(NoTradeTriggerChecker.CORRELATION_COLLAPSE_THRESHOLD)
DIR_MIN = float(NoTradeTriggerChecker.CORRELATION_ALIGNMENT_DIRECTION_MIN)
# pipeline fallbacks when a leg has zero std in the window (BE / BS / ES)
CORR_FALLBACK = {("BTC", "ETH"): 0.85, ("BTC", "SOL"): 0.80, ("ETH", "SOL"): 0.80}

BOOT_BLOCK = 90
BOOT_N = 2000
BOOT_SEED = 7
CONTROL_SEEDS = 200
REPORT_PATH = REPO / "training" / "reports" / "corr_collapse_lab_p384.json"


# =============================================================================
# PURE PIECES (unit-tested on synthetic frames)
# =============================================================================

def pipeline_corr(closes: pd.DataFrame, window: int = CORR_WINDOW) -> pd.Series:
    """Mean pairwise corr of the LAST `window` pct returns, bar by bar,
    exactly as data_mgmt/market_data_pipeline.py computes
    `correlation_btc_eth_sol`. NaN until `window` returns exist."""
    cols = list(ASSETS)
    c = closes[cols].to_numpy(dtype=float)
    n = len(c)
    out = np.full(n, np.nan)
    for t in range(window, n):
        seg = c[t - window: t + 1]                 # window+1 closes -> window rets
        prev = seg[:-1]
        denom = np.where(prev != 0, prev, 1.0)
        rets = np.diff(seg, axis=0) / denom       # pct returns, pipeline convention
        vals = []
        for (a, b), fb in CORR_FALLBACK.items():
            x = rets[:, cols.index(a)]
            y = rets[:, cols.index(b)]
            if np.std(x) > 0 and np.std(y) > 0:
                vals.append(float(np.corrcoef(x, y)[0, 1]))
            else:
                vals.append(fb)
        xc = float(np.clip(sum(vals) / 3.0, -1.0, 1.0))
        out[t] = xc if np.isfinite(xc) else np.nan
    return pd.Series(out, index=closes.index, name="corr20")


def all_same_direction(dirs: pd.DataFrame, dir_min: float = DIR_MIN) -> pd.Series:
    """The checker's `all_same`: every asset strictly beyond +m, or every
    asset strictly beyond -m. NaN in any asset -> False (absence is not
    evidence of alignment, P2)."""
    d = dirs[list(ASSETS)]
    pos = (d > dir_min).all(axis=1)
    neg = (d < -dir_min).all(axis=1)
    return (pos | neg) & d.notna().all(axis=1)


def fire_mask(corr: pd.Series, dirs: pd.DataFrame,
              threshold: float = CORR_THRESHOLD,
              dir_min: float = DIR_MIN) -> pd.Series:
    """CORRELATION_COLLAPSE as the live checker fires it (has_validated_edge
    has no producer -> always False, so it never exempts)."""
    same = all_same_direction(dirs, dir_min)
    return (corr >= threshold).fillna(False) & same


def episodes(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    """[(start, length)] of consecutive True runs."""
    out: List[Tuple[int, int]] = []
    m = np.asarray(mask, dtype=bool)
    i, n = 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def flatten_positions(book: pd.Series, mask: pd.Series) -> pd.Series:
    """FLATTEN policy: the book's target forced to 0 on fired bars; the bar
    the mask clears the position is the book's target again."""
    m = mask.reindex(book.index).fillna(False).astype(bool)
    return book.where(~m, 0.0)


def random_mask(n: int, lengths: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """A random mask with the SAME episode-length multiset, placed as
    non-overlapping contiguous runs (uniform over arrangements via a random
    composition of the free bars into gaps). Same fired-bar count, same
    number of transitions -> turnover-matched."""
    lengths = [int(x) for x in lengths]
    total = sum(lengths)
    k = len(lengths)
    out = np.zeros(n, dtype=bool)
    if k == 0:
        return out
    free = n - total
    if free < 0:
        raise ValueError("episode lengths exceed the series length")
    order = rng.permutation(k)
    cuts = np.sort(rng.integers(0, free + 1, size=k))     # k cut points in [0, free]
    gaps = np.diff(np.concatenate([[0], cuts]))           # k non-negative gaps
    pos = 0
    for gi, li in zip(gaps, order):
        pos += int(gi)
        L = lengths[li]
        out[pos: pos + L] = True
        pos += L
    return out


def block_bootstrap_sum_ci(d: np.ndarray, block: int = BOOT_BLOCK,
                           n: int = BOOT_N, seed: int = BOOT_SEED) -> Tuple[float, float]:
    """CI90 on the SUM of a per-bar series by circular-free block bootstrap
    (contiguous blocks of `block` bars, resampled with replacement, the
    resampled sum rescaled to the original length)."""
    d = np.asarray(d, dtype=float)
    m = len(d)
    if m < block * 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    nb = m // block
    scale = m / (nb * block)
    out = np.empty(n)
    for i in range(n):
        st = rng.integers(0, m - block + 1, size=nb)
        s = np.concatenate([d[a:a + block] for a in st])
        out[i] = s.sum() * scale
    return (float(np.percentile(out, 5)), float(np.percentile(out, 95)))


def decide_verdict(diff_by_era: Dict[str, float],
                   control_p90_by_era: Dict[str, float],
                   fired_bars: int) -> Tuple[str, List[str]]:
    """The pre-committed rule (VERDICT_RULE), as a function."""
    if fired_bars <= 0:
        return "TRIGGER_UNREACHABLE", ["mask never fires on six years of history"]
    blockers: List[str] = []
    for era in ("design", "pre_design"):
        d = diff_by_era.get(era)
        p90 = control_p90_by_era.get(era)
        if d is None or p90 is None or d != d:
            blockers.append(f"{era}: no measurement")
            continue
        if d <= 0:
            blockers.append(f"{era}: FLATTEN-HOLD {d*100:+.3f}pp <= 0")
        if d <= p90:
            blockers.append(f"{era}: FLATTEN-HOLD {d*100:+.3f}pp does not beat "
                            f"random-mask p90 {p90*100:+.3f}pp")
    return ("FLATTEN_EARNS" if not blockers else "HOLD_STANDS"), blockers


def era_masks(n: int) -> Dict[str, np.ndarray]:
    """funding_legs_lab.ERAS bands applied to the ALIGNED frame's positional
    index (BTC/ETH share the lab's per-asset index exactly; SOL's is ~61 bars
    later — the aligned frame starts at the latest asset's MIN_BARS)."""
    k = np.arange(n)
    out = {}
    for era, (a, b) in ERAS.items():
        m = (k >= a) & ((k < b) if b is not None else np.ones(n, dtype=bool))
        out[era] = m
    return out


# =============================================================================
# THE MEASUREMENT
# =============================================================================

def _bps(x: pd.Series) -> Optional[float]:
    """Mean per-bar return in bps, or None when there are no bars — an era with
    zero fires must read as "no measurement" (null), never as NaN-in-JSON."""
    return float(x.mean() * 1e4) if len(x) else None


def _rd(x: Optional[float], nd: int) -> Optional[float]:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _provenance() -> Dict[str, Any]:
    rev = "unknown"
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                             capture_output=True, text=True, encoding="utf-8",
                             timeout=30).stdout.strip() or "unknown"
    except Exception as e:  # noqa: silent-swallow — provenance only; the lab records "unknown" and says why
        rev = f"unknown ({type(e).__name__})"
    files = {}
    for a in ASSETS:
        for p in (PRICE_DIR / f"{a}_4H_ohlcv.parquet", FUNDING_DIR / f"{a}_funding_1d.parquet"):
            files[str(p.relative_to(REPO))] = (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))
                if p.exists() else "MISSING")
    return {
        "git_rev": rev,
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_file_mtimes": files,
        "corr_window": CORR_WINDOW,
        "corr_threshold": CORR_THRESHOLD,
        "direction_min": DIR_MIN,
        "corr_fallbacks": {f"{a}/{b}": v for (a, b), v in CORR_FALLBACK.items()},
        "bootstrap": {"block": BOOT_BLOCK, "n": BOOT_N, "seed": BOOT_SEED},
        "control_seeds": CONTROL_SEEDS,
        "eras_positional_bands": {k: list(v) for k, v in ERAS.items()},
        "direction_source": ("regimebook target via funding_legs_lab.build_positions "
                             "(= _last_quant_directions under regimebook_mode enforce)"),
        "cost_model": "funding_legs_lab.pnl (honest CDE per-leg cost, carry every bar)",
    }


def run(report_path: Path = REPORT_PATH, control_seeds: int = CONTROL_SEEDS,
        quiet: bool = False) -> Dict[str, Any]:
    P = (lambda *a, **k: None) if quiet else print
    P("=" * 78)
    P("PRE-COMMITTED VERDICT RULE (fixed before this ran):")
    P("  " + VERDICT_RULE)
    P("=" * 78)

    closes: Dict[str, pd.Series] = {}
    funding: Dict[str, pd.Series] = {}
    books: Dict[str, pd.DataFrame] = {}
    for a in ASSETS:
        closes[a] = load_closes(a)
        funding[a] = load_funding_daily(a)
        books[a] = build_positions(a, closes[a], funding[a])
        P(f"{a}: {len(closes[a])} bars {closes[a].index.min().date()} -> "
          f"{closes[a].index.max().date()}; book rows {len(books[a])}")

    # ---- align on a common 4H index -------------------------------------
    common = closes["BTC"].index
    for a in ASSETS:
        common = common.intersection(closes[a].index).intersection(books[a].index)
    common = common.sort_values()
    close_df = pd.DataFrame({a: closes[a].reindex(common) for a in ASSETS})
    dir_df = pd.DataFrame({a: books[a]["book"].reindex(common) for a in ASSETS})
    n = len(common)
    P(f"aligned frame: {n} bars {common.min().date()} -> {common.max().date()}")

    # corr needs the 20 closes BEFORE the aligned start too: compute on the
    # union of closes, then take the aligned index.
    full_idx = closes["BTC"].index
    for a in ASSETS:
        full_idx = full_idx.intersection(closes[a].index)
    full_close = pd.DataFrame({a: closes[a].reindex(full_idx) for a in ASSETS}).sort_index()
    corr = pipeline_corr(full_close).reindex(common)

    same = all_same_direction(dir_df)
    mask = fire_mask(corr, dir_df)
    fired = int(mask.sum())
    eps = episodes(mask.to_numpy())
    lengths = [L for _, L in eps]
    P(f"\ncorr20 >= {CORR_THRESHOLD}: {int((corr >= CORR_THRESHOLD).sum())} bars "
      f"({(corr >= CORR_THRESHOLD).mean()*100:.2f}%); all-same-direction: "
      f"{int(same.sum())} bars ({same.mean()*100:.2f}%)")
    P(f"FIRE MASK (both): {fired} bars ({fired/n*100:.2f}%), "
      f"{len(eps)} episodes, mean length {np.mean(lengths) if lengths else 0:.2f} bars")

    if fired == 0:
        verdict, blockers = decide_verdict({}, {}, 0)
        rep = {"verdict": verdict, "blockers": blockers, "verdict_rule": VERDICT_RULE,
               "aligned_bars": n, "fired_bars": 0, "provenance": _provenance()}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        P("\n!!! THE MASK NEVER FIRES ON SIX YEARS OF HISTORY — the trigger is "
          "UNREACHABLE with the conjuncts live; HOLD vs FLATTEN is moot. !!!")
        return rep

    # ---- HOLD vs FLATTEN, per asset --------------------------------------
    hold_net: Dict[str, pd.Series] = {}
    hold_gross: Dict[str, pd.Series] = {}
    flat_net: Dict[str, pd.Series] = {}
    flat_cost: Dict[str, pd.Series] = {}
    hold_cost: Dict[str, pd.Series] = {}
    pnl_idx = None
    for a in ASSETS:
        book = dir_df[a]
        hp = pnl(a, book, closes[a], funding[a])
        fp = pnl(a, flatten_positions(book, mask), closes[a], funding[a])
        hold_net[a], hold_gross[a], hold_cost[a] = hp["net"], hp["gross"], hp["cost"]
        flat_net[a], flat_cost[a] = fp["net"], fp["cost"]
        pnl_idx = hp.index if pnl_idx is None else pnl_idx.intersection(fp.index)
    pnl_idx = pnl_idx.intersection(common)
    m_al = mask.reindex(pnl_idx).astype(bool).to_numpy()
    N = len(pnl_idx)
    eras = era_masks(N)
    era_dates = {k: ([str(pnl_idx[v].min().date()), str(pnl_idx[v].max().date())]
                     if v.any() else None) for k, v in eras.items()}

    per_asset: Dict[str, Any] = {}
    sleeve_diff = pd.Series(0.0, index=pnl_idx)
    sleeve_hold = pd.Series(0.0, index=pnl_idx)
    for a in ASSETS:
        hn = hold_net[a].reindex(pnl_idx); hg = hold_gross[a].reindex(pnl_idx)
        fn = flat_net[a].reindex(pnl_idx)
        d = fn - hn
        sleeve_diff += d
        sleeve_hold += hn
        bk = dir_df[a].reindex(pnl_idx)
        row: Dict[str, Any] = {
            "hold_net_pct": round(float(hn.sum()) * 100, 3),
            "flatten_net_pct": round(float(fn.sum()) * 100, 3),
            "flatten_minus_hold_pct": round(float(d.sum()) * 100, 3),
            "extra_cost_pct": round(float((flat_cost[a].reindex(pnl_idx)
                                           - hold_cost[a].reindex(pnl_idx)).sum()) * 100, 3),
            "fired_bar_book_net_bps": _rd(_bps(hn[m_al]), 2),
            "fired_bar_book_gross_bps": _rd(_bps(hg[m_al]), 2),
            "unfired_bar_book_net_bps": _rd(_bps(hn[~m_al]), 2),
            "unfired_bar_book_gross_bps": _rd(_bps(hg[~m_al]), 2),
            "fired_bars_held_long": int((bk[m_al] > DIR_MIN).sum()),
            "fired_bars_held_short": int((bk[m_al] < -DIR_MIN).sum()),
            "by_era": {},
        }
        lo, hi = block_bootstrap_sum_ci(d.to_numpy())
        row["flatten_minus_hold_ci90_pct"] = [round(lo * 100, 3), round(hi * 100, 3)]
        for era, em in eras.items():
            if not em.any():
                continue
            sub_m = m_al & em
            elo, ehi = block_bootstrap_sum_ci(d.to_numpy()[em])
            row["by_era"][era] = {
                "bars": int(em.sum()),
                "fired_bars": int(sub_m.sum()),
                "fire_rate_pct": round(float(sub_m.sum() / em.sum()) * 100, 3),
                "hold_net_pct": round(float(hn[em].sum()) * 100, 3),
                "flatten_minus_hold_pct": round(float(d[em].sum()) * 100, 3),
                "flatten_minus_hold_ci90_pct": [round(elo * 100, 3), round(ehi * 100, 3)],
                "fired_bar_book_net_bps": _rd(_bps(hn[sub_m]), 2),
                "fired_bar_book_gross_bps": _rd(_bps(hg[sub_m]), 2),
                "unfired_bar_book_net_bps": _rd(_bps(hn[em & ~m_al]), 2),
            }
        per_asset[a] = row

    # ---- sleeve (three-asset sum) ----------------------------------------
    sleeve: Dict[str, Any] = {
        "hold_net_pct": round(float(sleeve_hold.sum()) * 100, 3),
        "flatten_minus_hold_pct": round(float(sleeve_diff.sum()) * 100, 3),
        "fired_bar_book_net_bps": _rd(_bps(sleeve_hold[m_al]), 2),
        "unfired_bar_book_net_bps": _rd(_bps(sleeve_hold[~m_al]), 2),
        "by_era": {},
    }
    lo, hi = block_bootstrap_sum_ci(sleeve_diff.to_numpy())
    sleeve["flatten_minus_hold_ci90_pct"] = [round(lo * 100, 3), round(hi * 100, 3)]
    diff_by_era: Dict[str, float] = {}
    for era, em in eras.items():
        if not em.any():
            continue
        sub_m = m_al & em
        elo, ehi = block_bootstrap_sum_ci(sleeve_diff.to_numpy()[em])
        diff_by_era[era] = float(sleeve_diff[em].sum())
        sleeve["by_era"][era] = {
            "bars": int(em.sum()),
            "fired_bars": int(sub_m.sum()),
            "fire_rate_pct": round(float(sub_m.sum() / em.sum()) * 100, 3),
            "episodes": len(episodes(sub_m)),
            "hold_net_pct": round(float(sleeve_hold[em].sum()) * 100, 3),
            "flatten_minus_hold_pct": round(diff_by_era[era] * 100, 3),
            "flatten_minus_hold_ci90_pct": [round(elo * 100, 3), round(ehi * 100, 3)],
            "fired_bar_book_net_bps": _rd(_bps(sleeve_hold[sub_m]), 2),
            "unfired_bar_book_net_bps": _rd(_bps(sleeve_hold[em & ~m_al]), 2),
        }

    # ---- turnover-matched random-mask control ----------------------------
    lengths_al = [L for _, L in episodes(m_al)]
    ctrl_overall: List[float] = []
    ctrl_era: Dict[str, List[float]] = {k: [] for k in eras if eras[k].any()}
    ctrl_asset: Dict[str, List[float]] = {a: [] for a in ASSETS}
    for seed in range(control_seeds):
        rng = np.random.default_rng(1000 + seed)
        rm = random_mask(N, lengths_al, rng)
        rm_s = pd.Series(rm, index=pnl_idx)
        tot = pd.Series(0.0, index=pnl_idx)
        for a in ASSETS:
            book = dir_df[a].reindex(pnl_idx)
            fp = pnl(a, flatten_positions(book, rm_s), closes[a], funding[a])
            d = (fp["net"].reindex(pnl_idx) - hold_net[a].reindex(pnl_idx)).fillna(0.0)
            ctrl_asset[a].append(float(d.sum()))
            tot += d
        ctrl_overall.append(float(tot.sum()))
        for era, em in eras.items():
            if em.any():
                ctrl_era[era].append(float(tot[em].sum()))
    co = np.asarray(ctrl_overall)
    if not (co.std() > 0):
        raise SystemExit("[P384] the random-mask control does not move — the "
                         "control is vacuous (P174); refusing to report a verdict.")
    control: Dict[str, Any] = {
        "seeds": control_seeds,
        "overall": {
            "mean_pct": round(float(co.mean()) * 100, 3),
            "p10_pct": round(float(np.percentile(co, 10)) * 100, 3),
            "p50_pct": round(float(np.percentile(co, 50)) * 100, 3),
            "p90_pct": round(float(np.percentile(co, 90)) * 100, 3),
            "observed_percentile": round(float((co < float(sleeve_diff.sum())).mean()) * 100, 1),
        },
        "by_era": {},
        "per_asset_p90_pct": {a: round(float(np.percentile(v, 90)) * 100, 3)
                              for a, v in ctrl_asset.items()},
    }
    p90_by_era: Dict[str, float] = {}
    for era, vals in ctrl_era.items():
        ce = np.asarray(vals)
        p90_by_era[era] = float(np.percentile(ce, 90))
        control["by_era"][era] = {
            "mean_pct": round(float(ce.mean()) * 100, 3),
            "p10_pct": round(float(np.percentile(ce, 10)) * 100, 3),
            "p50_pct": round(float(np.percentile(ce, 50)) * 100, 3),
            "p90_pct": round(p90_by_era[era] * 100, 3),
            "observed_percentile": round(float((ce < diff_by_era[era]).mean()) * 100, 1),
        }

    verdict, blockers = decide_verdict(diff_by_era, p90_by_era, fired)

    # ---- standalone facts (the premise) ----------------------------------
    premise = {
        "fired_bar_sleeve_net_bps": sleeve["fired_bar_book_net_bps"],
        "fired_bar_sleeve_net_negative": (sleeve["fired_bar_book_net_bps"] is not None
                                          and sleeve["fired_bar_book_net_bps"] < 0),
        "per_asset_fired_bar_net_bps": {a: per_asset[a]["fired_bar_book_net_bps"] for a in ASSETS},
        "per_asset_fired_bar_net_negative": {
            a: (per_asset[a]["fired_bar_book_net_bps"] is not None
                and per_asset[a]["fired_bar_book_net_bps"] < 0) for a in ASSETS},
        "all_fired_bars_are_all_long": bool(all(per_asset[a]["fired_bars_held_short"] == 0
                                                for a in ASSETS)),
    }

    # ---- table --------------------------------------------------------------
    P(f"\nfire rate overall: {fired}/{n} = {fired/n*100:.2f}%  "
      f"({len(eps)} episodes, mean {np.mean(lengths):.2f} bars, max {max(lengths)})")
    P(f"{'era':<12}{'bars':>7}{'fired':>7}{'rate%':>7}{'fired net bps':>15}"
      f"{'unfired bps':>13}{'FLAT-HOLD pp':>14}{'CI90':>22}{'ctrl p90':>10}{'pctl':>7}")
    for era, row in sleeve["by_era"].items():
        c = control["by_era"][era]
        ci = f"[{row['flatten_minus_hold_ci90_pct'][0]:+.3f},{row['flatten_minus_hold_ci90_pct'][1]:+.3f}]"
        _fb = row['fired_bar_book_net_bps']
        _fb_s = f"{_fb:>15.2f}" if _fb is not None else f"{'n/a':>15}"
        P(f"{era:<12}{row['bars']:>7}{row['fired_bars']:>7}{row['fire_rate_pct']:>7.2f}"
          f"{_fb_s}{row['unfired_bar_book_net_bps']:>13.2f}"
          f"{row['flatten_minus_hold_pct']:>+14.3f}{ci:>22}{c['p90_pct']:>+10.3f}{c['observed_percentile']:>7.1f}")
    ci = f"[{sleeve['flatten_minus_hold_ci90_pct'][0]:+.3f},{sleeve['flatten_minus_hold_ci90_pct'][1]:+.3f}]"
    P(f"{'OVERALL':<12}{N:>7}{fired:>7}{fired/N*100:>7.2f}{sleeve['fired_bar_book_net_bps']:>15.2f}"
      f"{sleeve['unfired_bar_book_net_bps']:>13.2f}{sleeve['flatten_minus_hold_pct']:>+14.3f}{ci:>22}"
      f"{control['overall']['p90_pct']:>+10.3f}{control['overall']['observed_percentile']:>7.1f}")
    P(f"\n{'asset':<6}{'hold net%':>10}{'FLAT-HOLD pp':>14}{'extra cost pp':>15}"
      f"{'fired bps':>11}{'unfired bps':>13}{'ctrl p90':>10}")
    for a in ASSETS:
        r = per_asset[a]
        P(f"{a:<6}{r['hold_net_pct']:>10.2f}{r['flatten_minus_hold_pct']:>+14.3f}"
          f"{r['extra_cost_pct']:>+15.3f}{r['fired_bar_book_net_bps']:>11.2f}"
          f"{r['unfired_bar_book_net_bps']:>13.2f}{control['per_asset_p90_pct'][a]:>+10.3f}")
    P(f"\nPREMISE (fired-bar sleeve net per bar): {premise['fired_bar_sleeve_net_bps']:+.2f} bps "
      f"-> {'NEGATIVE' if premise['fired_bar_sleeve_net_negative'] else 'NOT negative'}; "
      f"all fired bars all-LONG: {premise['all_fired_bars_are_all_long']}")
    P(f"\nVERDICT: {verdict}")
    for b in blockers:
        P(f"  blocker: {b}")

    rep = {
        "entry": "P384",
        "verdict": verdict,
        "blockers": blockers,
        "verdict_rule": VERDICT_RULE,
        "aligned_bars": n,
        "pnl_bars": N,
        "aligned_span": [str(common.min().date()), str(common.max().date())],
        "era_dates": era_dates,
        "fired_bars": fired,
        "fire_rate_pct": round(fired / n * 100, 3),
        "corr_ge_threshold_pct": round(float((corr >= CORR_THRESHOLD).mean()) * 100, 3),
        "all_same_direction_pct": round(float(same.mean()) * 100, 3),
        "episodes": len(eps),
        "episode_length_mean": round(float(np.mean(lengths)), 3),
        "episode_length_max": int(max(lengths)),
        "fired_episodes": [
            {"start": str(common[s0]), "end": str(common[s0 + L - 1]), "length": int(L),
             "corr20_max": round(float(corr.iloc[s0: s0 + L].max()), 4)}
            for s0, L in eps
        ],
        "premise": premise,
        "sleeve": sleeve,
        "per_asset": per_asset,
        "control": control,
        "provenance": _provenance(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    P(f"\nreport -> {report_path}")
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", default=str(REPORT_PATH))
    ap.add_argument("--control-seeds", type=int, default=CONTROL_SEEDS)
    args = ap.parse_args(argv)
    rep = run(Path(args.output), control_seeds=args.control_seeds)
    return 2 if rep["verdict"] == "TRIGGER_UNREACHABLE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
