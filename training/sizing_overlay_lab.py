"""[P288] Sizing-overlay lab — does DYNAMIC SIZING on top of the certified
rule books earn after costs?

Operator question (2026-08-16): "shouldn't the model decide the amount of
buy/sell?" This lab answers the sizing half by measurement, the house way:
small pre-registered grid, design-era selection, pre-design era stability,
NO validation-era reads (structurally: every series is truncated at the
design-era end before any overlay is computed), nothing wired into the
runtime. An earning config's next step is a P-entry + forward-ledger
decision, never a deploy.

THE OVERLAY CONTRACT
  pos[t] = book[t] * mult[t],  mult[t] in [0, m_max], causal.
  The overlay multiplies the certified book's position. It never changes
  the book's sign and never creates a position the book doesn't hold —
  sizing only, direction stays the certified mechanism's.

PRE-REGISTERED GRID (6 configs + control; no searches — multiplicity is
the enemy, P260):
  VOLTARGET   mult = clip(sigma_star / sigma_t, 0.25, m_max)
              sigma_t = trailing 20-bar realized vol (causal);
              sigma_star in {design-era median vol, 0.75 * median};
              m_max in {1.5, 2.0}.                      -> 4 configs
  TRENDSTRENGTH
              mult = clip(z_t, 0, m_max)
              z_t = trailing 250-bar z-score of |close/SMA200 - 1|
              (both windows trailing, causal); m_max in {1.5, 2.0}.
                                                        -> 2 configs
  UNIT        mult = 1 exactly. CONTROL: must reproduce the base book to
              the last decimal in both eras (continuous AND quantized) or
              the chassis import is wrong and the run refuses.

QUANTIZATION (the granularity reality — the DECIDING numbers):
  Continuous mult is an upper bound only. The deciding numbers quantize
  mult to the contract grid expressible at current equity (~$10.8k, 0.15
  fractions, P274): BTC steps of 1/2 (2ct book -> {0, 1, 2}ct) capped at
  1.0x book (the venue book IS 2ct); ETH steps of 1/8 (8ct book); SOL
  steps of 1/4 (4ct book). NOTE recorded in the report: mult > 1.0 on
  ETH/SOL is only expressible if the P274 cap==target coupling is raised —
  an operator config decision; BTC upsizing is inexpressible today.

PRE-COMMITTED VERDICT (written before the first run; the rule decides,
not the reader):
  A config EARNS on an asset iff BOTH:
    (a) QUANTIZED after-cost net beats the base book in DESIGN [3000,9100);
    (b) QUANTIZED after-cost net in PRE-DESIGN [800,3000) is
        >= base_pre - 0.05*|base_pre|  (era-stability, P243/P244 lesson).
  Anything else: BASE STANDS. If several configs earn on one asset, the
  smallest-m_max earner is the reported candidate (conservatism tiebreak).
  Turnover is reported for every cell: an overlay that earns by adding
  turnover is buying its gain at the venue.

Chassis is single-sourced from mechanism_lab (P172): book_targets +
pnl_after_cost are imported, never re-implemented; the per-bar series used
for Sharpe is parity-asserted against pnl_after_cost's net on EVERY cell
(mismatch -> hard refusal). Causality of both overlay forms is verified at
startup by a P164-style construction test (violent future perturbation
must not move the past).

Run:  python -X utf8 training/sizing_overlay_lab.py
Report: training/reports/sizing_overlay_lab_p288.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if REPO.name == "training":
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

from training.mechanism_lab import (  # noqa: E402
    book_targets, pnl_after_cost, DS, DE, PRE)
from training.regime_model_lab import _ctx  # noqa: E402
from training.train_supervised_full import COST_BPS  # noqa: E402

REPORT = REPO / "training" / "reports" / "sizing_overlay_lab_p288.json"
ASSETS = ("BTC", "ETH", "SOL")
BARS_PER_DAY_4H = 6
ANN = np.sqrt(365 * BARS_PER_DAY_4H)

# contract grid at current equity (~$10.8k, 0.15 fractions, P274):
#   steps = expressible fractions of the base book; cap = max mult
#   expressible without an operator cap-raise (BTC book already at 2ct).
QUANT = {"BTC": {"steps": 2, "cap": 1.0},
         "ETH": {"steps": 8, "cap": None},   # cap = m_max (needs cap-raise)
         "SOL": {"steps": 4, "cap": None}}

VOL_WIN = 20
Z_WIN = 250
SMA_WIN = 200


# ---------------------------------------------------------------------------
# overlay forms (causal by construction: trailing windows only)
# ---------------------------------------------------------------------------

def trailing_vol(close: np.ndarray, win: int = VOL_WIN) -> np.ndarray:
    """sigma_t from log returns up to and including t. NaN until warm."""
    n = len(close)
    lr = np.full(n, np.nan)
    lr[1:] = np.log(close[1:] / close[:-1])
    out = np.full(n, np.nan)
    for i in range(win, n):
        out[i] = np.nanstd(lr[i - win + 1:i + 1])
    return out


def mult_voltarget(close: np.ndarray, sigma_star: float,
                   m_max: float) -> np.ndarray:
    sig = trailing_vol(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = sigma_star / sig
    m = np.clip(m, 0.25, m_max)
    m[~np.isfinite(m)] = 1.0          # warmup / zero-vol: overlay inactive
    return m


def mult_trendstrength(close: np.ndarray, m_max: float) -> np.ndarray:
    n = len(close)
    sma = np.full(n, np.nan)
    c = np.cumsum(close)
    sma[SMA_WIN - 1:] = (c[SMA_WIN - 1:] -
                         np.concatenate(([0.0], c[:-SMA_WIN]))) / SMA_WIN
    strength = np.abs(close / sma - 1.0)
    out = np.full(n, 1.0)             # warmup: overlay inactive
    for i in range(Z_WIN, n):
        w = strength[i - Z_WIN + 1:i + 1]
        mu, sd = np.nanmean(w), np.nanstd(w)
        if np.isfinite(sd) and sd > 0 and np.isfinite(strength[i]):
            out[i] = float(np.clip((strength[i] - mu) / sd, 0.0, m_max))
    return out


def quantize_mult(mult: np.ndarray, book: np.ndarray, asset: str,
                  m_max: float) -> np.ndarray:
    """Round mult to the expressible contract grid; flat book stays flat."""
    q = QUANT[asset]
    cap = q["cap"] if q["cap"] is not None else m_max
    m = np.round(mult * q["steps"]) / q["steps"]
    m = np.clip(m, 0.0, cap)
    return np.where(book == 0.0, 0.0, m)


# ---------------------------------------------------------------------------
# evaluation (Sharpe via a per-bar mirror, parity-asserted vs the chassis)
# ---------------------------------------------------------------------------

def per_bar_net(close: np.ndarray, pos: np.ndarray, cost_rt_bps: float,
                lo: int, hi: int, allow_validation: bool = False) -> np.ndarray:
    """Per-bar after-cost return mirroring pnl_after_cost bar-for-bar.
    Its sum is asserted equal to the chassis net at every call site —
    the mirror exists only because Sharpe needs a series.

    [P420] Same hard guard as the chassis it mirrors: a read at/after the
    validation-era start REFUSES unless the caller says `allow_validation=True`
    — an explicit, greppable opt-in for the labs that deliberately spend the
    one-shot recent-era read (conviction_sizing_lab, gate_probes_lab), which
    must ALSO ledger it (training/splits.record_window_usage). The unguarded
    mirror was the one door through which a validation read could happen
    without either."""
    assert hi <= DE or allow_validation, (
        "validation-era read attempted through per_bar_net — refused; pass "
        "allow_validation=True AND record_window_usage() if this is the "
        "deliberate one-shot read (P420)")
    r1 = np.zeros_like(close)
    r1[:-1] = close[1:] / close[:-1] - 1.0
    seg = slice(lo, hi - 1)
    gross = pos[seg] * r1[seg]
    dpos = np.abs(np.diff(pos[lo:hi], prepend=pos[lo]))[:hi - 1 - lo]
    cost = dpos * (cost_rt_bps / 2.0) / 1e4
    return np.nan_to_num(gross, nan=0.0) - cost


def evaluate(close, pos, cost, lo, hi) -> dict:
    p = pnl_after_cost(close, pos, cost, lo, hi)
    series = per_bar_net(close, pos, cost, lo, hi)
    # the last-bar trade charge sits outside the series' cost slice;
    # reconcile: chassis charges dpos over [lo,hi), series over [lo,hi-1)
    dpos_all = np.abs(np.diff(pos[lo:hi], prepend=pos[lo]))
    tail_cost = float(dpos_all[hi - 1 - lo:].sum()) * (cost / 2.0) / 1e4
    # tolerance = the chassis's own round(…, 4) granularity, not tighter
    if abs(float(series.sum()) - tail_cost - p["net"]) > 5.1e-5:
        raise SystemExit(
            f"[P288] per-bar mirror diverged from chassis net "
            f"({series.sum() - tail_cost:.6f} vs {p['net']:.6f}) — REFUSING")
    sd = float(np.std(series))
    p["sharpe"] = round(float(np.mean(series)) / sd * ANN, 3) if sd > 0 else 0.0
    return p


# ---------------------------------------------------------------------------
# causality construction test (P164 style)
# ---------------------------------------------------------------------------

def causality_check(close: np.ndarray) -> None:
    t0 = min(2000, len(close) - 100)
    fut = close.copy()
    fut[t0:] = fut[t0:] * np.linspace(3.0, 0.1, len(fut) - t0)  # violent
    for name, fn in (
            ("voltarget", lambda c: mult_voltarget(c, 0.01, 2.0)),
            ("trendstrength", lambda c: mult_trendstrength(c, 2.0))):
        a, b = fn(close)[:t0], fn(fut)[:t0]
        if not np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=0):
            raise SystemExit(f"[P288] CAUSALITY VIOLATION in {name} — refusing")
    print("[P288] causality construction test: PASS (future perturbation "
          "moved nothing in the past)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    out = {"generated_at": datetime.now(timezone.utc).isoformat(),
           "design_era": [DS, DE], "pre_design_era": list(PRE),
           "quantization": {a: dict(QUANT[a]) for a in ASSETS},
           "expressibility_note": (
               "mult > 1.0 on ETH/SOL requires raising the P274 cap==target "
               "coupling (operator config decision); BTC upsizing beyond the "
               "2ct book is inexpressible at current equity — its quantized "
               "grid caps at 1.0x."),
           "verdict_rule": (
               "EARNS iff quantized design net > base AND quantized "
               "pre-design net >= base_pre - 0.05*|base_pre|; smallest "
               "m_max earner reported; else BASE STANDS."),
           "assets": {}}
    verdicts = {}
    for a in ASSETS:
        c = _ctx(a)
        # STRUCTURAL validation-window guard: truncate every series at the
        # design-era end BEFORE any overlay math — no bar >= DE is readable.
        close = np.asarray(c["close"], dtype=float)[:DE]
        lab = np.asarray(c["lab"])[:DE]
        fz = np.asarray(c["fz"], dtype=float)[:DE]
        if a == "BTC":
            causality_check(close)
        book = book_targets(a, lab, fz)
        cost = COST_BPS[a]
        base_d = evaluate(close, book, cost, DS, DE)
        base_p = evaluate(close, book, cost, PRE[0], PRE[1])

        # sigma_star from the DESIGN slice only (selection window)
        sig = trailing_vol(close)
        med = float(np.nanmedian(sig[DS:DE]))

        configs = [("UNIT", "control", {}, np.full(len(close), 1.0))]
        for ss_name, ss in (("median", med), ("0.75med", 0.75 * med)):
            for mm in (1.5, 2.0):
                configs.append((
                    f"VOLTARGET(sigma*={ss_name},mmax={mm})", "voltarget",
                    {"sigma_star": ss, "m_max": mm},
                    mult_voltarget(close, ss, mm)))
        for mm in (1.5, 2.0):
            configs.append((
                f"TRENDSTRENGTH(mmax={mm})", "trendstrength",
                {"m_max": mm}, mult_trendstrength(close, mm)))

        rows, earners = [], []
        for name, kind, params, mult in configs:
            m_max = params.get("m_max", 1.0)
            pos_c = book * mult
            pos_q = book * quantize_mult(mult, book, a, m_max)
            cd = evaluate(close, pos_c, cost, DS, DE)
            cp = evaluate(close, pos_c, cost, PRE[0], PRE[1])
            qd = evaluate(close, pos_q, cost, DS, DE)
            qp = evaluate(close, pos_q, cost, PRE[0], PRE[1])
            if kind == "control":
                for got, want, tag in ((cd, base_d, "cont/design"),
                                       (cp, base_p, "cont/pre"),
                                       (qd, base_d, "quant/design"),
                                       (qp, base_p, "quant/pre")):
                    if abs(got["net"] - want["net"]) > 1e-9:
                        raise SystemExit(
                            f"[P288] UNIT control parity FAILED ({a} {tag}: "
                            f"{got['net']} != {want['net']}) — chassis "
                            f"import is wrong, refusing to report")
            row = {"config": name, "kind": kind, "params": params,
                   "continuous": {"design": cd, "pre_design": cp},
                   "quantized": {"design": qd, "pre_design": qp},
                   "turnover_delta_design_quant":
                       round(qd["turnover_units"] -
                             base_d["turnover_units"], 1)}
            earns = (kind != "control"
                     and qd["net"] > base_d["net"]
                     and qp["net"] >= base_p["net"] - 0.05 * abs(base_p["net"]))
            row["earns_quantized"] = bool(earns)
            if earns:
                earners.append((m_max, name, qd["net"]))
            rows.append(row)

        if earners:
            earners.sort()
            verdicts[a] = f"EARNS: {earners[0][1]}"
        else:
            verdicts[a] = "BASE STANDS"
        out["assets"][a] = {
            "base": {"design": base_d, "pre_design": base_p},
            "sigma_star_median": round(med, 6),
            "configs": rows, "verdict": verdicts[a]}
        print(f"\n[{a}] base design net={base_d['net']:+.4f} "
              f"(sh {base_d['sharpe']:+.2f})  pre net={base_p['net']:+.4f}")
        for r in rows:
            if r["kind"] == "control":
                continue
            qd, qp = r["quantized"]["design"], r["quantized"]["pre_design"]
            cd = r["continuous"]["design"]
            print(f"  {r['config']:34s} quant D={qd['net']:+.4f} "
                  f"P={qp['net']:+.4f} (cont D={cd['net']:+.4f}) "
                  f"dTO={r['turnover_delta_design_quant']:+.1f} "
                  f"{'EARNS' if r['earns_quantized'] else '-'}")
        print(f"  VERDICT {a}: {verdicts[a]}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nreport: {REPORT}")
    print("verdicts:", verdicts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
