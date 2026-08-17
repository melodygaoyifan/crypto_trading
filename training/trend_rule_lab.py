"""[P288-B] Trend-rule family sweep — SMA200 vs its OWN family.

The internal design-axis audit (P288) found the incumbent labeler never
faced its own rule family: it beat fitted forecasters (P250/P281) and the
cheap greedy jump switch (P256), but no EMA ensemble, breakout rule,
adaptive-lookback rule, or vol-managed variant was ever entered — while
the external literature names exactly this family as the plausible
beaters (Baltas-Kosowski; FRL 2025 risk-managed crypto momentum). This
lab runs that competition the house way: pre-registered candidates, NO
searches, design-era selection, pre-design stability, NO validation-era
reads (every series truncated at the design-era end before any signal
math), nothing wired into the runtime.

THE COMPETITION FRAME
  Trend-only book expression (the certified one): long 1.0 in BULL, flat
  otherwise. Challengers replace the BULL LABELER only. The BTC funding
  legs are deliberately untouched (they are a separate, uncertified
  question — P262); the incumbent control here is the SMA200x90d-momentum
  trend-only book, i.e. `regime_labels` bull, which for ETH/SOL IS the
  certified book and for BTC is its certified trend leg.

PRE-REGISTERED CANDIDATES (no knobs beyond what is written here):
  EMA-ENSEMBLE   bull = majority (>=2 of 3) of EMA-cross pairs
                 (30/150), (50/200), (100/400) bars, adjust=False.
  DONCHIAN-100   bull on close > trailing 100-bar close-high (excluding
                 the current bar); bear on close < trailing 100-bar
                 close-low; otherwise KEEP prior state (hysteresis by
                 construction). NOTE: close-based, not high/low-based —
                 the chassis data surface carries closes only; recorded
                 as a deviation from the classic form.
  ADAPTIVE-SMA   lookback 100 or 300 bars selected by the trailing
                 250-bar realized vol vs its EXPANDING (causal) 67th
                 percentile: high-vol regime -> 300, else 100. Bull =
                 close > selected SMA. One fixed mapping.
  VOLMGD-SMA200  the P270 literature contradiction, done the literature's
                 way: incumbent bull labeler x vol-target multiplier
                 clip(sigma*/sigma_t, 0.25, 2.0) with 0.25-wide REBALANCE
                 BANDS (the multiplier only moves when the raw value is
                 >= 0.25 away from the held one — the Baltas-Kosowski
                 turnover fix the P288 sizing lab's per-bar clip lacked),
                 quantized to the expressible contract grid (sizing lab's
                 QUANT: BTC 1/2-steps capped 1.0x, ETH 1/8, SOL 1/4).
                 sigma* = design-era median trailing vol (a constant
                 scale parameter, same convention as the sizing lab /
                 P259 constant-sigma exports; noted, not hidden).
  SMA200         the incumbent (control) — must reproduce the chassis
                 book EXACTLY or the run refuses.

PRE-COMMITTED VERDICT (written before the first run):
  A challenger DETHRONES the incumbent on an asset iff BOTH
    (a) after-cost net (its DECIDING form: quantized for VOLMGD, plain
        for the 0/1 labelers) beats the SMA200 trend-only book in the
        DESIGN era [3000, 9100), AND
    (b) its PRE-DESIGN era [800, 3000) net >= sma_pre - 0.05*|sma_pre|.
  A challenger meeting (a)+(b) while adding > 25% turnover vs the
  incumbent is FLAGGED-NOT-CROWNED (it buys its win at the venue).
  Otherwise: SMA200 STANDS.
  Virgin-era (2017-2020) transfer is NOT re-tested here (those parquets
  are not in the local tree) — the incumbent's certification includes it
  (P262), so even a dethroning challenger would hold LESS total evidence
  than the incumbent until it passes the same virgin-era probe plus a
  forward ledger. Any winner's next step is that probe + a P-entry, not
  a deploy.

Run:  python -X utf8 training/trend_rule_lab.py
Report: training/reports/trend_rule_lab_p288.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
if REPO.name == "training":
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

from training.mechanism_lab import (  # noqa: E402
    book_targets, pnl_after_cost, DS, DE, PRE)
from training.regime_model_lab import _ctx  # noqa: E402
from training.train_supervised_full import COST_BPS  # noqa: E402
# [P172] the Sharpe mirror + quantization grid are single-sourced from the
# sibling sizing lab, not copied a third time.
from training.sizing_overlay_lab import (  # noqa: E402
    evaluate, trailing_vol, quantize_mult)

REPORT = REPO / "training" / "reports" / "trend_rule_lab_p288.json"
ASSETS = ("BTC", "ETH", "SOL")

EMA_PAIRS = ((30, 150), (50, 200), (100, 400))
DON_WIN = 100
ADA_FAST, ADA_SLOW, ADA_VOLWIN = 100, 300, 250
VOLMGD_MMAX, VOLMGD_FLOOR, VOLMGD_BAND = 2.0, 0.25, 0.25


# ---------------------------------------------------------------------------
# challenger labelers (all trailing / recursive -> causal by construction;
# verified by the P164 test below anyway)
# ---------------------------------------------------------------------------

def _sma(close: np.ndarray, win: int) -> np.ndarray:
    return pd.Series(close).rolling(win).mean().to_numpy()


def lab_ema_ensemble(close: np.ndarray) -> np.ndarray:
    votes = np.zeros(len(close))
    for fast, slow in EMA_PAIRS:
        ef = pd.Series(close).ewm(span=fast, adjust=False).mean().to_numpy()
        es = pd.Series(close).ewm(span=slow, adjust=False).mean().to_numpy()
        votes += (ef > es).astype(float)
    # warmup honesty: before the slowest pair has seen a full span, ewm
    # values exist but are seeded — require the bar index past the slow span
    bull = votes >= 2.0
    bull[:max(s for _, s in EMA_PAIRS)] = False
    return bull.astype(float)


def lab_donchian(close: np.ndarray, win: int = DON_WIN) -> np.ndarray:
    s = pd.Series(close)
    hi = s.rolling(win).max().shift(1).to_numpy()
    lo = s.rolling(win).min().shift(1).to_numpy()
    out = np.zeros(len(close))
    state = 0.0
    for i in range(len(close)):
        if np.isnan(hi[i]) or np.isnan(lo[i]):
            state = 0.0
        elif close[i] > hi[i]:
            state = 1.0
        elif close[i] < lo[i]:
            state = 0.0          # bear -> flat in the trend-only expression
        out[i] = state
    return out


def lab_adaptive_sma(close: np.ndarray) -> np.ndarray:
    sig = trailing_vol(close, ADA_VOLWIN)
    fast, slow = _sma(close, ADA_FAST), _sma(close, ADA_SLOW)
    out = np.zeros(len(close))
    for i in range(len(close)):
        if np.isnan(sig[i]):
            continue
        hist = sig[ADA_VOLWIN:i + 1]
        hist = hist[~np.isnan(hist)]
        if len(hist) < 30:
            continue
        thr = float(np.quantile(hist, 2.0 / 3.0))   # expanding -> causal
        ref = slow[i] if sig[i] >= thr else fast[i]
        if not np.isnan(ref) and close[i] > ref:
            out[i] = 1.0
    return out


def volmgd_mult(close: np.ndarray, sigma_star: float) -> np.ndarray:
    sig = trailing_vol(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.clip(sigma_star / sig, VOLMGD_FLOOR, VOLMGD_MMAX)
    raw[~np.isfinite(raw)] = 1.0
    out = np.empty_like(raw)
    held = 1.0
    for i, r in enumerate(raw):
        if abs(r - held) >= VOLMGD_BAND:      # rebalance band
            held = r
        out[i] = held
    return out


CHALLENGERS = (
    ("EMA-ENSEMBLE", lab_ema_ensemble),
    ("DONCHIAN-100", lab_donchian),
    ("ADAPTIVE-SMA", lab_adaptive_sma),
)


def causality_check(close: np.ndarray) -> None:
    t0 = min(2000, len(close) - 100)
    fut = close.copy()
    fut[t0:] *= np.linspace(3.0, 0.1, len(fut) - t0)
    checks = list(CHALLENGERS) + [
        ("VOLMGD-mult", lambda c: volmgd_mult(c, 0.01))]
    for name, fn in checks:
        a, b = fn(close)[:t0], fn(fut)[:t0]
        if not np.array_equal(np.nan_to_num(a), np.nan_to_num(b)):
            raise SystemExit(f"[P288-B] CAUSALITY VIOLATION in {name} — refusing")
    print("[P288-B] causality construction test: PASS")


def n_flips(pos: np.ndarray, lo: int, hi: int) -> int:
    return int(np.sum(np.abs(np.diff(pos[lo:hi])) > 1e-12))


def main() -> int:
    out = {"generated_at": datetime.now(timezone.utc).isoformat(),
           "design_era": [DS, DE], "pre_design_era": list(PRE),
           "verdict_rule": (
               "DETHRONES iff deciding-form net > SMA200 in design AND "
               "pre-design >= sma_pre - 0.05*|sma_pre|; >25% extra turnover "
               "= FLAGGED-not-crowned; else SMA200 STANDS. Virgin-era "
               "transfer NOT re-tested here — incumbent keeps that edge."),
           "assets": {}}
    verdicts = {}
    for a in ASSETS:
        c = _ctx(a)
        # STRUCTURAL guard: nothing past the design-era end is readable.
        close = np.asarray(c["close"], dtype=float)[:DE]
        lab = np.asarray(c["lab"])[:DE]
        n = len(close)
        if a == "BTC":
            causality_check(close)

        base_pos = (lab == 1).astype(float)
        # UNIT-control parity: the chassis book with funding structurally
        # inactive (fz = all-NaN) must equal the trend-only incumbent.
        chassis = book_targets(a, lab, np.full(n, np.nan))
        if not np.array_equal(chassis, base_pos):
            raise SystemExit(f"[P288-B] incumbent parity FAILED for {a} — "
                             f"harness disagrees with chassis, refusing")
        cost = COST_BPS[a]
        base_d = evaluate(close, base_pos, cost, DS, DE)
        base_p = evaluate(close, base_pos, cost, PRE[0], PRE[1])
        base_d["n_flips"] = n_flips(base_pos, DS, DE)
        base_p["n_flips"] = n_flips(base_pos, PRE[0], PRE[1])

        sig = trailing_vol(close)
        sigma_star = float(np.nanmedian(sig[DS:DE]))

        rows, crowned = [], []
        cands = [(name, fn(close), None) for name, fn in CHALLENGERS]
        vm = volmgd_mult(close, sigma_star)
        vm_q = quantize_mult(vm, base_pos, a, VOLMGD_MMAX)
        cands.append(("VOLMGD-SMA200", base_pos * vm_q, base_pos * vm))

        for name, pos, pos_cont in cands:
            d = evaluate(close, pos, cost, DS, DE)
            p = evaluate(close, pos, cost, PRE[0], PRE[1])
            d["n_flips"] = n_flips(pos, DS, DE)
            p["n_flips"] = n_flips(pos, PRE[0], PRE[1])
            row = {"challenger": name, "design": d, "pre_design": p}
            if pos_cont is not None:
                row["continuous_design"] = evaluate(close, pos_cont, cost, DS, DE)
                row["continuous_pre"] = evaluate(close, pos_cont, cost,
                                                 PRE[0], PRE[1])
            wins = (d["net"] > base_d["net"] and
                    p["net"] >= base_p["net"] - 0.05 * abs(base_p["net"]))
            extra_to = (d["turnover_units"] >
                        1.25 * max(base_d["turnover_units"], 1e-9))
            row["dethrones"] = bool(wins and not extra_to)
            row["flagged_turnover"] = bool(wins and extra_to)
            if row["dethrones"]:
                crowned.append(name)
            rows.append(row)

        verdicts[a] = (f"DETHRONED by {', '.join(crowned)}" if crowned
                       else "SMA200 STANDS")
        out["assets"][a] = {"incumbent": {"design": base_d,
                                          "pre_design": base_p},
                            "sigma_star": round(sigma_star, 6),
                            "challengers": rows,
                            "verdict": verdicts[a]}
        print(f"\n[{a}] SMA200 D={base_d['net']:+.4f} (sh {base_d['sharpe']:+.2f}, "
              f"TO {base_d['turnover_units']}, flips {base_d['n_flips']})  "
              f"P={base_p['net']:+.4f}")
        for r in rows:
            d, p = r["design"], r["pre_design"]
            tag = ("DETHRONES" if r["dethrones"]
                   else "FLAGGED(turnover)" if r["flagged_turnover"] else "-")
            print(f"  {r['challenger']:14s} D={d['net']:+.4f} (sh {d['sharpe']:+.2f}, "
                  f"TO {d['turnover_units']:6.1f}, flips {d['n_flips']:3d})  "
                  f"P={p['net']:+.4f} (TO {p['turnover_units']:6.1f})  {tag}")
        print(f"  VERDICT {a}: {verdicts[a]}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nreport: {REPORT}")
    print("verdicts:", verdicts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
