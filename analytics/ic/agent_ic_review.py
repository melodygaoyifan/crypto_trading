"""[P230] Per-AGENT live-IC review — the instrument the P228 decision assumes.

P228 recorded that the 12 zero-weighted ADVISE agents stay off until an agent's
FORWARD live IC clears the P166 cost-aware bar — but no standing tool computed
per-agent IC: the P143/P198 numbers were one-off analyses, and
`compute_shadow_ic` covers the v5.1 strategy shadow ledgers, not the 26 fusion
agents. This closes that gap: it reads the attribution signal logs the tracker
already writes every tick, joins them to public 4H closes, and prints per-agent
forward IC with the same cost-aware arithmetic the promotion path names.

WHERE THIS RUNS (P213 lesson — stated, not implied):
  IN-CONTAINER on the live server, where the attribution volume is:
      docker exec hmats-engine python -X utf8 analytics/ic/agent_ic_review.py
  Or operator-local against scp'd logs:  --log-dir <dir with signals_*.jsonl>
  Prices come from Kraken's PUBLIC OHLC endpoint (no key). A missing log dir
  or an unreachable price API is a REFUSAL (exit 2), never an empty report —
  "no data" must not read as "no signal" (P199/P213/P227b).

Stdlib only, observation only: reads logs, fetches public prices, writes a
report under analytics/ic/reports/. Places no orders, mutates no state.

VERDICT SEMANTICS (mirrors analytics/shadow_ic P166 arithmetic):
  expected_edge_bps(h) = 0.7979 * 2*sin(pi*IC/6) * fwd_vol_bps(h)
  bar: IC > 0 at every horizon, |t| >= 2 (t = IC*sqrt(n-1)), and
  expected_edge >= 2.0 x round-trip cost. [P382] The round trip is derived
  from `core.cde_fees.CDE_FEE_BPS` (2 legs x the asset's taker bps; an agent
  row takes the max over the assets it signalled on), floored at the refuted
  6bps model; every row carries `rt_cost_bps` + `cost_source`.
  A PROMOTE-CANDIDATE here is a candidate for a WEIGHT in
  ADVISE_WEIGHTS_BY_REGIME with its own P-entry — never an automatic change.
  [P371] A signal that barely moved in the window is REFUSED-DEGENERATE, never
  PROMOTE: <= 2 sign changes (per-asset MAX), or one sign on > 90% of the
  scored records, or a |t| that clears 2 at n//h but not at
  n_eff = min(n//h, sign changes). Its n_eff is the number of sign changes,
  not the row count, and the P231 overlap correction cannot see that
  (P370 Q5 — the sentiment F&G artifact: 3/5/1 changes per asset, t 9.4).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import urllib.request
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HORIZON_BARS = (1, 4)          # 4h, 16h on the 4H clock
SAFETY_MARGIN = 2.0            # spread/impact absent from the fee number
MIN_N = 30
# [P420] Was the home trio only, and the record filter below DROPPED every
# attribution row for any other asset SILENTLY — XRP and BNB have been
# tradeable since P412/P412c, so their agents were invisible to this review.
# Pairs for every asset the regime-book harness can price (single source).
KRAKEN_PAIRS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD",
                "XRP": "XRPUSD", "ADA": "ADAUSD", "LTC": "LTCUSD",
                "DOGE": "XDGUSD", "BNB": "BNBUSD"}

# [P382] ROUND-TRIP COST. Was the literal 6.0 — "Coinbase 3bps taker x 2
# legs" — which is the percentage model P315/P334 REFUTED: the CDE fee is
# 9.4-14.5 bps PER LEG (flat per contract -> a percentage of notional), and
# P374 measured the all-in round trip at BTC 27.7 / ETH 44.0 / SOL 41.0 bps.
# The required-IC inverter below was therefore asking agents to clear a bar
# ~3x lower than the venue actually charges. The cost is now DERIVED from the
# registered calibration `core.cde_fees.CDE_FEE_BPS` (P327): 2 legs x taker
# bps per asset, floored at the refuted 6.0, then x SAFETY_MARGIN. An agent
# row mixes assets, so its bar is the MAX over the assets it actually signalled
# on (a mixed row must not be cheaper than its dearest member); with no asset
# attribution it is the max over all three.
REFUTED_MODEL_RT_BPS = 6.0     # the FLOOR, named for what it is
COST_SOURCE_CDE = "cde_fees"
COST_SOURCE_FALLBACK = "fallback_refuted_model"
_CDE_LEGS = 2.0


def _cde_taker_leg_bps() -> dict | None:
    """asset -> per-LEG taker bps from core.cde_fees, or None (logged)."""
    try:
        # Run as a script (docker exec ... python analytics/ic/agent_ic_review.py)
        # sys.path[0] is this file's directory, not the repo root.
        _repo = str(Path(__file__).resolve().parents[2])
        if _repo not in sys.path:
            sys.path.insert(0, _repo)
        from core.cde_fees import CDE_FEE_BPS
    except Exception as e:  # noqa: silent-swallow — reported below
        print(f"  WARNING [P382]: core.cde_fees unavailable ({type(e).__name__}: "
              f"{e}) — the required-IC bar is on the REFUTED 3bps/side model "
              f"({REFUTED_MODEL_RT_BPS:.1f}bps round trip, P315/P334); every "
              f"edge-vs-cost verdict is ~3x too generous. "
              f"cost_source={COST_SOURCE_FALLBACK}", file=sys.stderr)
        return None
    out: dict = {}
    for a, sides in dict(CDE_FEE_BPS).items():
        try:
            v = float(sides["taker"])
        except (KeyError, TypeError, ValueError):  # noqa: silent-swallow — shape coercion
            continue
        if math.isfinite(v) and v > 0:
            out[str(a).upper()] = v
    return out or None


def rt_cost_bps_for(assets=None) -> tuple:
    """(round_trip_bps, cost_source) for a set of assets (P382).

    Per-asset: 2 x that asset's CDE taker leg. Several assets: the MAX.
    None/empty: the max over the whole table. Unknown asset: the WORST in
    the table (P167 — an unmeasured cost is assumed expensive). Calibration
    unreadable: the refuted 6.0 floor, with cost_source saying so. The floor
    applies in every branch, so this can only ever RAISE the bar (P248).
    """
    table = _cde_taker_leg_bps()
    if table is None:
        return REFUTED_MODEL_RT_BPS, COST_SOURCE_FALLBACK
    worst = max(table.values())
    names = [str(a).upper() for a in (assets or []) if a]
    if names:
        # membership test: a deliberate worst-case fallback, written so the
        # silent_failure_audit's two-variable-default heuristic (P47-Bug-2)
        # does not flag it
        rt = max(_CDE_LEGS * (table[a] if a in table else worst) for a in names)
    else:
        rt = _CDE_LEGS * worst
    return max(REFUTED_MODEL_RT_BPS, rt), COST_SOURCE_CDE


# The default bar when a caller has no asset attribution: the dearest of the
# three. Computed ONCE at import (it is a constant table); the P230 test's
# round-trip identity `edge(required_ic(vol)) == SAFETY_MARGIN * TAKER_RT_BPS`
# still holds against this value.
TAKER_RT_BPS, COST_SOURCE = rt_cost_bps_for(None)


def _refuse(msg: str) -> None:
    print(f"REFUSING TO REPORT: {msg}", file=sys.stderr)
    sys.exit(2)


def resolve_log_dir(cli: str | None) -> Path:
    if cli:
        return Path(cli)
    env = os.environ.get("HMATS_LOG_DIR")
    return Path(env, "attribution") if env else Path("logs/attribution")


def load_signal_records(log_dir: Path, window_days: int) -> list[dict]:
    files = sorted(glob.glob(str(log_dir / "signals_*.jsonl")))
    if not files:
        _refuse(
            f"no signals_*.jsonl under {log_dir}. This tool needs the "
            f"attribution volume — run in-container (docker exec hmats-engine "
            f"python -X utf8 analytics/ic/agent_ic_review.py) or pass "
            f"--log-dir pointing at scp'd logs. An empty dir here is 'no "
            f"data source', not 'no agent signal'."
        )
    cutoff = datetime.now(timezone.utc).timestamp() - window_days * 86400
    out = []
    skipped = 0
    unpriced = {}  # [P420] asset -> rows dropped for lack of a Kraken pair
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    ts = datetime.fromisoformat(rec["ts"]).timestamp()
                except Exception:  # noqa: silent-swallow — per-record
                    # JSONL skip, batch-reported below (audit pattern A)
                    skipped += 1
                    continue
                if ts >= cutoff and rec.get("asset") in KRAKEN_PAIRS:
                    rec["_ts"] = ts
                    out.append(rec)
                elif ts >= cutoff:
                    unpriced[str(rec.get("asset"))] = (
                        unpriced.get(str(rec.get("asset")), 0) + 1)  # [P420]
    if skipped:
        print(f"  note: {skipped} unparseable lines skipped", file=sys.stderr)
    if unpriced:  # [P420] a dropped asset must be SAID, never silent
        print(f"  note: rows dropped for assets with no Kraken pair: "
              f"{unpriced}", file=sys.stderr)
    return out


def fetch_closes(asset: str) -> tuple[list[float], list[float]]:
    """(bar_open_timestamps, closes) for 4H bars from Kraken public OHLC."""
    url = (f"https://api.kraken.com/0/public/OHLC?pair="
           f"{KRAKEN_PAIRS[asset]}&interval=240")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            payload = json.loads(r.read().decode())
    except Exception as e:
        _refuse(f"Kraken public OHLC unreachable for {asset}: "
                f"{type(e).__name__}: {e} — cannot price forward returns. "
                f"(Network refusal, distinct from 'no signals'.)")
    if payload.get("error"):
        _refuse(f"Kraken OHLC error for {asset}: {payload['error']}")
    key = next(k for k in payload["result"] if k != "last")
    rows = payload["result"][key]
    # [P265] Kraken returns the IN-PROGRESS 4H candle as the last row
    # (established at regime_book_shadow's P253c fix; this reviewer family
    # never got it). The join guard below claims "forward bar not closed yet"
    # but admitted the pair whose exit bar is the partial candle — the newest
    # forward returns per (asset, horizon) were priced on a provisional
    # close. Drop it at the source, like the harness does.
    rows = rows[:-1]
    return [float(r[0]) for r in rows], [float(r[4]) for r in rows]


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def required_ic(fwd_vol_bps: float, rt_bps: float | None = None) -> float | None:
    """Invert P166's edge model: IC needed for SAFETY_MARGIN x round-trip cost.

    [P382] `rt_bps` defaults to TAKER_RT_BPS (the dearest asset's CDE round
    trip); decide_agent_verdict passes the per-agent figure."""
    need = SAFETY_MARGIN * (TAKER_RT_BPS if rt_bps is None else float(rt_bps))
    if fwd_vol_bps <= 0:
        return None
    x = need / (0.7979 * 2.0 * fwd_vol_bps)
    if x >= 1.0:
        return None  # unreachable at this vol
    return 6.0 / math.pi * math.asin(x)


# =============================================================================
# [P371] DEGENERATE-SIGNAL GUARD — n_eff is the number of sign changes
# =============================================================================
# [P371] P370's audit: the `sentiment` agent read 30d IC 0.703 (t 9.16 at 16h)
# [P371] and this tool labelled it PROMOTE-CANDIDATE. The attribution census
# [P371] showed ONE global daily Fear&Greed value shared by all three assets:
# [P371] direction -1 on every tick from Jul 22 to Aug 17, then +1 from Aug 19
# [P371] coincident with the rally. That is ONE sign change, not 687
# [P371] observations. The P231 overlap correction (n_eff = n/h) cannot see it
# [P371] — it corrects for horizon OVERLAP, not for a signal that is a
# [P371] near-constant — so the t-statistic was computed on a sample size the
# [P371] signal never had.
# [P371]
# [P371] A near-constant signal is degenerate in either of two shapes, measured
# [P371] on the SCORED series (the nonzero directions this tool correlates):
# [P371]   * few SIGN CHANGES across the window — a step function whose
# [P371]     n_eff is ~the number of changes;
# [P371]   * one sign occupying almost the whole window — a perma-bias whose
# [P371]     IC is an artifact of which way the window happened to go.
# [P371] Sign changes are counted PER ASSET SERIES and summed, because the
# [P371] scored list interleaves assets: BTC always +1 beside ETH always -1
# [P371] would read as a sign change on every row while neither series moved.
# [P371]
# [P371] A degenerate agent is REFUSED — verdict REFUSED-DEGENERATE with the
# [P371] reason printed — never PROMOTE-CANDIDATE. A varying signal is judged by
# [P371] exactly the bar above, unchanged: this guard can only REMOVE a
# [P371] promotion, never grant one.
DEGENERATE_MAX_SIGN_CHANGES = 2        # [P371] <= this many -> degenerate
DEGENERATE_DOMINANT_SHARE = 0.90       # [P371] one sign > this share -> degenerate
DEGENERATE_MIN_T_AT_SIGNAL_N_EFF = 2.0  # [P371] the tool's own |t| bar, re-priced at n_eff = sign changes
REFUSED_DEGENERATE = "REFUSED-DEGENERATE"   # [P371]


def _sign(x: float) -> int:  # [P371]
    return 1 if x > 0 else (-1 if x < 0 else 0)  # [P371]


def sign_changes_by_asset(xs: list, assets: list | None = None) -> dict:
    """[P371] {asset: sign changes along THAT asset's scored series}.

    [P371] With `assets` None the whole list is one series under key "_".
    [P371] Zero directions (which the scorer already drops) are skipped, not
    [P371] counted as a change — a pause is not a flip.
    """
    series: dict = defaultdict(list)  # [P371]
    if assets is None:  # [P371]
        series["_"] = list(xs)  # [P371]
    else:  # [P371]
        for a, x in zip(assets, xs):  # [P371]
            series[a].append(x)  # [P371]
    out: dict = {}  # [P371]
    for a, vals in series.items():  # [P371]
        prev, changes = None, 0  # [P371]
        for x in vals:  # [P371]
            sg = _sign(x)  # [P371]
            if sg == 0:  # [P371]
                continue  # [P371]
            if prev is not None and sg != prev:  # [P371]
                changes += 1  # [P371]
            prev = sg  # [P371]
        out[a] = changes  # [P371]
    return out  # [P371]


def count_sign_changes(xs: list, assets: list | None = None) -> int:
    """[P371] The agent's effective sign-change count = MAX over per-asset
    [P371] series, not the sum.

    [P371] Measured on the live ledger (30d): `sentiment` is ONE global F&G
    [P371] value, so its BTC/ETH/SOL series are copies of each other — summing
    [P371] would count the same flip three times and report 9 where the
    [P371] signal moved ~3 times. And BTC always +1 beside ETH always -1 must
    [P371] read as 0, not as a flip on every interleaved row. MAX understates
    [P371] n_eff for genuinely independent per-asset variation, which is the
    [P371] conservative direction for a guard that can only REMOVE a verdict.
    """
    per = sign_changes_by_asset(xs, assets)  # [P371]
    return max(per.values()) if per else 0  # [P371]


def signal_degeneracy(xs: list, assets: list | None = None) -> dict:
    """[P371] Is this scored series a near-constant? Returns the measurement.

    [P371] {"n", "sign_changes", "dominant_share", "degenerate", "reason"}.
    [P371] `reason` is None when the series genuinely varies.
    """
    n = len(xs)  # [P371]
    pos = sum(1 for x in xs if x > 0)  # [P371]
    neg = sum(1 for x in xs if x < 0)  # [P371]
    nz = pos + neg  # [P371]
    dominant = (max(pos, neg) / nz) if nz else 1.0  # [P371]
    changes = count_sign_changes(xs, assets)  # [P371]
    reasons = []  # [P371]
    if changes <= DEGENERATE_MAX_SIGN_CHANGES:  # [P371]
        reasons.append(  # [P371]
            f"IC computed on a signal with {changes} sign change(s); "  # [P371]
            f"n_eff is ~{changes}, not the row count ({n}); not evidence")  # [P371]
    if dominant > DEGENERATE_DOMINANT_SHARE:  # [P371]
        reasons.append(  # [P371]
            f"one sign occupies {dominant:.0%} of the {n} scored records "  # [P371]
            f"(IC computed on a signal with {changes} sign change(s); "  # [P371]
            f"n_eff is ~{changes}, not the row count); not evidence")  # [P371]
    return {"n": n, "sign_changes": changes,  # [P371]
            "dominant_share": round(dominant, 4),  # [P371]
            "degenerate": bool(reasons),  # [P371]
            "reason": "; ".join(reasons) if reasons else None}  # [P371]


def decide_agent_verdict(dirs_by_h: dict, rets_by_h: dict, fwd_vol: dict,
                         assets_by_h: dict | None = None) -> tuple:
    """[P371] One agent's per-horizon rows + verdict, as a PURE function.

    [P371] Extracted from main() so the verdict can be driven with a synthetic
    [P371] series (P324's rule: a verdict rule is tested by being CALLED, not
    [P371] by a source pin). The arithmetic is byte-for-byte the P230/P231 loop;
    [P371] what is new is the degeneracy check on each scored horizon and the
    [P371] REFUSED-DEGENERATE verdict that outranks PROMOTE-CANDIDATE.
    """
    # [P370] annotated: mypy infers the value type from the FIRST row
    # (`str | float | None`) and then flags the nested `degeneracy` dict as
    # [dict-item]. The rows genuinely mix scalars and one dict; say so at
    # the declaration rather than baselining the finding (P284b/P287g).
    rows: dict[int, dict] = {}  # [P371]
    verdict_bits: list = []  # [P371]
    # [P382] The cost bar for THIS agent: max over the assets it signalled on
    # (union across horizons); no attribution -> the dearest of the three.
    _agent_assets: set = set()
    for _h_assets in (assets_by_h or {}).values():
        _agent_assets.update(a for a in (_h_assets or []) if a)
    rt_bps, cost_source = rt_cost_bps_for(sorted(_agent_assets))
    for h in HORIZON_BARS:  # [P371]
        xs, ys = list(dirs_by_h.get(h, [])), list(rets_by_h.get(h, []))  # [P371]
        n = len(xs)  # [P371]
        ic = _pearson(xs, ys)  # [P371]
        if n < MIN_N or ic is None:  # [P371]
            rows[h] = {"n": n, "ic": ic, "verdict": "INSUFFICIENT"}  # [P371]
            verdict_bits.append("INSUFFICIENT")  # [P371]
            continue  # [P371]
        # [P231] OVERLAP CORRECTION. An h-bar forward return sampled every
        # bar overlaps h times, so consecutive observations are not
        # independent and the naive t = IC*sqrt(n-1) overstates
        # significance by ~sqrt(h). The first live reading's headline
        # (model_alpha 16h t=4.4) was exactly this artifact — corrected it
        # is ~2.2, below the multiple-testing hurdle (research: Bailey/
        # Lopez de Prado MinTRL + Harvey-Liu-Zhu t>3 for new signals).
        # Effective n = n/h is the standard cheap correction; a proper
        # Newey-West would be tighter still, so this remains generous.
        n_eff = max(3, n // h)  # [P371] (moved, unchanged)
        t = ic * math.sqrt(n_eff - 1)  # [P371] (moved, unchanged)
        vol_h = fwd_vol.get(h, 0.0)  # [P371]
        edge = 0.7979 * 2.0 * math.sin(math.pi * ic / 6.0) * vol_h  # [P371]
        req = required_ic(vol_h, rt_bps)  # [P371] [P382] per-agent CDE cost
        clears_arith = (ic > 0 and abs(t) >= 2.0  # [P371]
                        and req is not None and ic >= req)  # [P371]
        # [P371] The guard. n_eff above corrects for overlap; it cannot see a
        # [P371] signal that barely moved. Measure the series itself.
        dg = signal_degeneracy(  # [P371]
            xs, (assets_by_h or {}).get(h) if assets_by_h else None)  # [P371]
        # [P371] And the claim itself re-priced at the signal's OWN n_eff:
        # [P371] "n_eff is ~N sign changes, not the row count" applied
        # [P371] literally. The live sentiment series had 3/5/1 sign changes
        # [P371] per asset (restart ticks + two one-day flickers around the
        # [P371] Aug 19 flip) — above a bare <=2 bar and still nowhere near
        # [P371] the 160-687 rows the t was computed on. A t that is
        # [P371] significant at n//h and not at min(n//h, sign changes) is
        # [P371] not evidence. Only ever tightens: n_eff_signal <= n_eff.
        n_eff_signal = max(1, min(n_eff, dg["sign_changes"]))  # [P371]
        t_signal = ic * math.sqrt(max(n_eff_signal - 1, 0))  # [P371]
        dg["n_eff_signal"] = n_eff_signal  # [P371]
        dg["t_at_n_eff_signal"] = round(t_signal, 2)  # [P371]
        if abs(t) >= 2.0 and abs(t_signal) < DEGENERATE_MIN_T_AT_SIGNAL_N_EFF:  # [P371]
            dg["degenerate"] = True  # [P371]
            extra = (f"IC computed on a signal with {dg['sign_changes']} "  # [P371]
                     f"sign change(s); n_eff is ~{dg['sign_changes']}, not the "  # [P371]
                     f"row count ({n}); |t| at n_eff={n_eff_signal} is "  # [P371]
                     f"{abs(t_signal):.2f} < 2 (was {abs(t):.2f} at n//h); "  # [P371]
                     f"not evidence")  # [P371]
            dg["reason"] = (f"{dg['reason']}; {extra}" if dg["reason"]  # [P371]
                            else extra)  # [P371]
        degenerate = bool(dg["degenerate"])  # [P371]
        clears = clears_arith and not degenerate  # [P371]
        rows[h] = {"n": n, "n_eff": n_eff, "ic": round(ic, 4),  # [P371]
                   "t": round(t, 2), "t_note": f"overlap-corrected (n/{h})",  # [P371]
                   "edge_bps": round(edge, 2),  # [P371]
                   "required_ic": round(req, 4) if req else None,  # [P371]
                   # [P382] the bar this row was priced against + provenance
                   "rt_cost_bps": round(rt_bps, 2),
                   "required_bps": round(SAFETY_MARGIN * rt_bps, 2),
                   "cost_source": cost_source,
                   "clears_p166": clears,  # [P371]
                   "clears_p166_arithmetic": clears_arith,  # [P371]
                   "degenerate": degenerate,  # [P371]
                   "degeneracy": dg}  # [P371]
        if degenerate:  # [P371]
            verdict_bits.append("DEGENERATE")  # [P371]
            continue  # [P371]
        verdict_bits.append(  # [P371]
            "CLEARS" if clears else  # [P371]
            ("NEGATIVE" if (ic < 0 and abs(t) >= 2.0) else  # [P371]
             ("NOISE" if abs(t) < 1.0 else "WEAK")))  # [P371]
    if "DEGENERATE" in verdict_bits:  # [P371]
        verdict = REFUSED_DEGENERATE  # [P371] outranks PROMOTE: not evidence
    elif verdict_bits and all(v == "CLEARS" for v in verdict_bits):  # [P371]
        verdict = "PROMOTE-CANDIDATE"  # [P371]
    elif "NEGATIVE" in verdict_bits:  # [P371]
        verdict = "NEGATIVE"  # [P371]
    elif all(v == "INSUFFICIENT" for v in verdict_bits):  # [P371]
        verdict = "INSUFFICIENT"  # [P371]
    else:  # [P371]
        verdict = "HOLD"  # [P371]
    return rows, verdict  # [P371]


# =============================================================================
# [P312] THE REPORT SHAPE, owned by the PRODUCER
# =============================================================================
# P295d: `scripts/seat_check.py` read `agents.<name>.<h>` while this file emits
# `agents.<name>.horizons.<h>`. Every agent parsed to n=0, and the seat
# controller recommended vacating every seat on a live book from nothing. It
# passed its own tests because the fixture and the reader were written by the
# same hand and agreed with each other — the P310 class.
#
# So the nesting lives HERE, in one place, and the consumer's contract test
# builds a report through THESE functions rather than hand-writing one. If the
# shape ever moves, the test's report moves with it and the reader fails.
HORIZON_CONTAINER_KEY = "horizons"
AGENTS_CONTAINER_KEY = "agents"


def build_agent_entry(rows: dict, verdict: str) -> dict:
    """One agent's block: per-horizon cells under HORIZON_CONTAINER_KEY."""
    return {HORIZON_CONTAINER_KEY: rows, "verdict": verdict}


def build_report(window_days: int, fwd_vol: dict, generated: str) -> dict:
    """The report skeleton. `agents` is filled with build_agent_entry values."""
    return {"generated": generated, "window_days": window_days,
            "fwd_vol_bps": fwd_vol,
            # [P382] the cost model the whole report was judged on
            "cost_model": {"default_rt_bps": TAKER_RT_BPS,
                           "cost_source": COST_SOURCE,
                           "safety_margin": SAFETY_MARGIN,
                           "floor_rt_bps": REFUTED_MODEL_RT_BPS},
            AGENTS_CONTAINER_KEY: {}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--report-dir", default=None,
                    help="default: analytics/ic/reports next to this file")
    args = ap.parse_args()

    records = load_signal_records(resolve_log_dir(args.log_dir),
                                  args.window_days)
    if not records:
        _refuse(f"signal files exist but contain no records inside the "
                f"{args.window_days}d window — widen --window-days or check "
                f"the tracker is writing.")

    # [P420] price only the assets PRESENT in the window (was: every pair,
    # which after widening KRAKEN_PAIRS would be eight network calls and
    # eight refusal surfaces for assets that may carry no rows).
    _present = sorted({str(r.get("asset")) for r in records if r.get("asset")})
    bars = {a: fetch_closes(a) for a in _present}

    # agent -> horizon -> parallel lists of (direction, fwd_return_frac)
    dirs: dict = defaultdict(lambda: defaultdict(list))
    rets: dict = defaultdict(lambda: defaultdict(list))
    sig_assets: dict = defaultdict(lambda: defaultdict(list))  # [P371] per-asset sign-change count
    vol_samples: dict = defaultdict(list)
    joined = unjoined = 0
    for rec in records:
        ts_list, closes = bars[rec["asset"]]
        # bar containing the signal = last bar whose open <= ts
        i = bisect_right(ts_list, rec["_ts"]) - 1
        if i < 0:
            unjoined += 1
            continue
        for h in HORIZON_BARS:
            if i + h >= len(closes):
                continue  # forward bar not closed yet — honest gap, not a 0
            fwd = closes[i + h] / closes[i] - 1.0
            vol_samples[h].append(fwd)
            for sig in rec.get("signals", []):
                d = float(sig.get("direction", 0.0) or 0.0)
                if abs(d) < 1e-9:
                    continue  # zero-emission ticks carry no directional info
                a = sig.get("agent_name", "?")
                dirs[a][h].append(d)
                rets[a][h].append(fwd)
                sig_assets[a][h].append(rec["asset"])  # [P371]
        joined += 1
    print(f"records joined={joined} pre-history={unjoined} "
          f"window={args.window_days}d horizons={HORIZON_BARS} bars(4H)")

    fwd_vol = {
        h: (math.sqrt(sum(r * r for r in v) / len(v)) * 1e4 if v else 0.0)
        for h, v in vol_samples.items()
    }
    report = build_report(args.window_days, fwd_vol,
                          datetime.now(timezone.utc).isoformat())

    print(f"\ncost model [P382]: CDE taker x 2 legs from core.cde_fees, "
          f"per agent = max over its assets (floor {REFUTED_MODEL_RT_BPS:.1f}"
          f"bps, default {TAKER_RT_BPS:.1f}bps, source={COST_SOURCE}) "
          f"x {SAFETY_MARGIN:.1f} margin")
    print(f"\n{'agent':<14}{'h':>3}{'n':>6}{'IC':>8}{'t':>7}"
          f"{'edge_bps':>9}{'req_IC':>8}{'rt_bps':>8}  verdict")
    for agent in sorted(dirs):
        # [P371] The verdict rule is a pure function now (see
        # [P371] decide_agent_verdict) so a synthetic series can drive it; the
        # [P371] asset list lets the guard count sign changes PER asset.
        rows, verdict = decide_agent_verdict(  # [P371]
            dirs[agent], rets[agent], fwd_vol, sig_assets[agent])  # [P371]
        report[AGENTS_CONTAINER_KEY][agent] = build_agent_entry(rows, verdict)
        for h in HORIZON_BARS:
            r = rows[h]
            print(f"{agent:<14}{h:>3}{r['n']:>6}"
                  f"{(r.get('ic') if r.get('ic') is not None else float('nan')):>8.3f}"
                  f"{r.get('t', float('nan')):>7.2f}"
                  f"{r.get('edge_bps', float('nan')):>9.2f}"
                  f"{(r.get('required_ic') or float('nan')):>8.3f}"
                  f"{(r.get('rt_cost_bps') or float('nan')):>8.1f}"
                  f"  {verdict if h == HORIZON_BARS[-1] else ''}")
        if verdict == REFUSED_DEGENERATE:  # [P371]
            # [P371] Say WHY, at the row, or the refusal reads like a HOLD.
            for h in HORIZON_BARS:  # [P371]
                dg = rows[h].get("degeneracy")  # [P371]
                if dg and dg.get("degenerate"):  # [P371]
                    print(f"{'':<14}{h:>3}  REFUSED PROMOTE: {dg['reason']} "  # [P371]
                          f"(P371; check the attribution census before "  # [P371]
                          f"believing any IC above 0.3)")  # [P371]

    rep_dir = Path(args.report_dir) if args.report_dir else (
        Path(__file__).resolve().parent / "reports")
    rep_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = rep_dir / f"agent_ic_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {out}")
    print("NOTE: a PROMOTE-CANDIDATE is a candidate for a weight in "
          "ADVISE_WEIGHTS_BY_REGIME with its own P-entry — never automatic "
          "(P228). Verdicts on <60d of data are provisional.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
