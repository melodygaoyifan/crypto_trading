"""
HMATS v5.1 Phase Pre-6 - Shadow Strategy IC + Promotion Gate
==============================================================

Reads JSONL ledgers under data/strategy_shadow/ (microstructure_*.jsonl,
cascade_*.jsonl), joins each (ts, asset, strategy) signal to forward
returns sourced from the 4H OHLCV parquets, computes per-strategy IC
across configurable horizons, and emits a promotion verdict per strategy:

    PROMOTE : [P166] every one of these, at every horizon with enough samples —
                * IC positive (nothing downstream inverts a negative-IC strategy)
                * |IC| > 0.05 floor, and Sharpe > 0.5
                * |t| = |IC|*sqrt(n-1) >= 2.0  (distinguishable from zero)
                * expected edge >= 6.0bps round-trip cost x 2.0 margin, priced
                  off the measured forward-return volatility
              A missing volatility measurement is a REFUSAL, not a skip.
    HOLD    : not yet 30 days of data, OR mixed signal, OR any bar above unmet
    KILL    : 14d window has IC < 0.05 (kill-criteria per v5.1 prompt)

Output:
    analytics/shadow_ic/reports/shadow_ic_{utc_ts}.json
    + console human-readable summary table

Iron Laws honored:
  4. fail-closed: missing parquet / unparseable JSONL line / NaN → that
     row dropped, run continues.
  7. Phase Pre-6 is the framework; promotion VERDICT is computed here but
     not auto-applied. Phase 10 (Day 57+) reads these reports and gates.

WHERE THIS RUNS — [P213] OPERATOR-LOCAL ONLY. NOT a server-side capability.
    The price series lives in `training/training_data/`, which `.dockerignore`
    excludes (line 41), so the parquets are NOT in the engine image and CI does
    not have them either. This module IS in the image (`analytics/` is not
    excluded), which is precisely the trap: run it in the container and every
    strategy comes back `ohlcv_missing` with a report written anyway — output
    that reads like "the strategies have no signal" when the truth is "this tool
    cannot run here". That conflation is what hid P199 for months, so it is now
    a hard, named failure (see `main`) rather than a quietly empty report.

    Run it on the operator's machine:
        python -X utf8 training/scripts/refresh_ohlcv_4h.py   # refresh prices
        python -X utf8 analytics/shadow_ic/compute_shadow_ic.py --window-days 30

    Making it server-side would mean shipping the OHLCV parquets into the image
    or mounting them from a volume. Deliberately NOT done: it is an occasional
    analysis tool, and the parquets are large and refreshed from Binance monthly
    archives on the operator's box. Revisit only if the gate needs to run
    unattended.

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
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


REPO = Path(__file__).resolve().parents[2]
LEDGER_DIR = REPO / "data" / "strategy_shadow"
OHLCV_DIR = REPO / "training" / "training_data" / "drl_training"
REPORT_DIR = REPO / "analytics" / "shadow_ic" / "reports"


# ---------------------------------------------------------------------------
# [P166] Cost-aware promotion constants
# ---------------------------------------------------------------------------
#
# Round-trip execution cost, in bps, charged once per entry+exit pair.
#
# 3.0 bps taker x 2 sides = 6.0. This deliberately assumes **100% taker** even
# though `analytics/sixty_day_review` reports maker_fee_ratio = 0.994, because
# that metric classifies by ORDER TYPE (n_limit / n_classified), not by how the
# fill actually cleared -- a limit order that crosses the spread pays taker. The
# realized round-trip cost measured over the 85 closed trades in
# `data/trade_attribution.jsonl` is **31.1 bps median / 33.0 bps mean** (Kraken
# tier), i.e. ~400x the 0.078 bps that `training/backtest_framework.FeeSchedule`
# assumes by default. 6.0 is the forward-looking Coinbase number; it is a floor,
# not a measurement, which is why COST_MARGIN exists.
DEFAULT_ROUND_TRIP_COST_BPS = 6.0

# Require the estimated edge to cover costs this many times over. The margin
# absorbs (a) spread and market impact, which the fee number excludes entirely,
# and (b) the optimism of the linear IC->edge model below, which assumes a
# full-size position on the sign of the signal with no capacity constraint.
DEFAULT_COST_MARGIN = 2.0

# An IC must be distinguishable from zero before it can be acted on.
# SE(IC) ~= 1/sqrt(n-1), so the legacy gate -- IC > 0.05 at min_samples = 30 --
# accepted a reading 0.27 standard errors from zero. Clearing |t| >= 2.0 at
# IC = 0.05 needs n ~= 1600.
DEFAULT_MIN_IC_T_STAT = 2.0

# Absolute floor retained from the legacy gate. Kept as a floor (not the whole
# test) so a low-volatility asset cannot produce a trivially small cost-derived
# requirement and promote on a statistically real but economically empty edge.
DEFAULT_IC_FLOOR = 0.05

# E|z| for a standard normal: converts "correlation" into "expected return when
# you take a full-size position on the sign of the signal".
_SIGN_EDGE_FACTOR = math.sqrt(2.0 / math.pi)  # ~0.7979


class Verdict(Enum):
    PROMOTE = "PROMOTE"
    HOLD = "HOLD"
    KILL = "KILL"
    INSUFFICIENT_SAMPLES = "INSUFFICIENT_SAMPLES"


@dataclass
class PromotionAssessment:
    """[P166] The verdict plus the arithmetic behind it.

    `blockers` is the whole point: a HOLD that does not say which bar was
    missed is indistinguishable from a HOLD that is one sample away from
    promoting, and the operator cannot tell whether to wait or to kill.
    """
    verdict: Verdict
    blockers: List[str] = field(default_factory=list)
    per_horizon: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    round_trip_cost_bps: float = 0.0
    cost_margin: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "blockers": list(self.blockers),
            "per_horizon": {str(k): v for k, v in self.per_horizon.items()},
            "round_trip_cost_bps": self.round_trip_cost_bps,
            "cost_margin": self.cost_margin,
        }


def spearman_to_pearson(rho_s: float) -> float:
    """Convert a Spearman rank correlation to the Pearson equivalent.

    Exact for the bivariate normal: r = 2*sin(pi*rho_s/6). Near zero the two
    are almost identical (rho_s=0.05 -> r=0.0523), but the conversion is cheap
    and keeps the edge formula below dimensionally honest -- that formula is
    derived for a *linear* correlation.
    """
    rho_s = max(-1.0, min(1.0, float(rho_s)))
    return 2.0 * math.sin(math.pi * rho_s / 6.0)


def expected_edge_bps(ic_spearman_val: float, fwd_vol_bps: float) -> float:
    """Expected per-round-trip edge, in bps, from an IC and a forward vol.

    E[r * sign(x)] = r_pearson * sigma_r * E|z| for jointly-normal (x, r).

    This is what the legacy gate never did: IC is dimensionless, costs are in
    bps, and the two were never brought into the same units, so `IC > 0.05`
    could not possibly know whether it cleared a 6 bps round trip. It does not
    -- at a 4-bar (16h) horizon, IC 0.05 on ~107 bps of forward vol is worth
    about 4.4 bps, which loses money on every venue this system has traded.
    """
    return abs(_SIGN_EDGE_FACTOR * spearman_to_pearson(ic_spearman_val) * float(fwd_vol_bps))


def required_ic_for_costs(
    fwd_vol_bps: float,
    round_trip_cost_bps: float = DEFAULT_ROUND_TRIP_COST_BPS,
    cost_margin: float = DEFAULT_COST_MARGIN,
) -> float:
    """Invert `expected_edge_bps`: the smallest IC that pays for its own costs.

    Returns +inf when no correlation is sufficient (vol too low to ever cover
    the cost of trading it) -- the caller must treat that as "cannot promote",
    never as "no requirement".
    """
    if fwd_vol_bps <= 0:
        return math.inf
    needed_bps = float(round_trip_cost_bps) * float(cost_margin)
    r_pearson = needed_bps / (_SIGN_EDGE_FACTOR * float(fwd_vol_bps))
    if r_pearson >= 2.0:          # asin domain: no correlation can get there
        return math.inf
    return (6.0 / math.pi) * math.asin(r_pearson / 2.0)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_shadow_ledgers(
    ledger_dir: Path,
    # [P199] Was ("microstructure", "cascade") — which read only the two
    # families that emit NOTHING, and silently ignored funding_*.jsonl and
    # ml_factor_*.jsonl, the only ones producing signal. The gate meant to
    # validate the v5.1 promotion could not see the strategies it was judging.
    prefixes: Tuple[str, ...] = ("microstructure", "cascade", "funding", "ml_factor"),
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
    # [P199] Prefer the OHLCV-only series over the DRL training parquet.
    # `_4H_full.parquet` is a 130-column TRAINING artifact, regenerated only by a
    # full rebuild_pipeline run — it sat frozen at 2026-03-31 while the shadow
    # ledgers started 2026-04-30. Zero overlap, so every record scored N=0 and
    # this gate reported INSUFFICIENT_SAMPLES for months, which is
    # indistinguishable from "the strategies have no signal". Coupling an
    # analytics price series to a training artifact is what caused that.
    # `_4H_ohlcv.parquet` is refreshed by training/scripts/refresh_ohlcv_4h.py
    # and is validated to reproduce the training parquet's bars exactly.
    # The training parquet stays as a fallback so this never hard-fails.
    candidates = [
        OHLCV_DIR / f"{asset}_4H_ohlcv.parquet",
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
        # [P166] Forward-return dispersion, in bps, per horizon. This is the
        # missing scale factor that lets a dimensionless IC be compared against
        # a cost in bps. It is measured from the SAME joined pairs the IC is
        # computed on, so it cannot drift from them. A horizon with fewer than
        # two pairs gets no entry at all -- deliberately absent rather than
        # 0.0, so `assess_promotion` can fail closed instead of silently
        # concluding that a zero-vol asset needs zero edge (P164/P159).
        fwd_vol_bps_per_h: Dict[int, float] = {}
        for h in horizons_bars:
            ys = per_h[h][1]
            if len(ys) < 2:
                continue
            mean_y = sum(ys) / len(ys)
            var_y = sum((y - mean_y) ** 2 for y in ys) / (len(ys) - 1)
            fwd_vol_bps_per_h[h] = (var_y ** 0.5) * 10_000.0

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
            "fwd_vol_bps_per_horizon": fwd_vol_bps_per_h,  # [P166]
            "annualized_sharpe": sharpe,
            "n_directional": len(per_trade_returns),
        }

    return out


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def assess_promotion(
    ic_per_h: Dict[int, float],
    n_per_h: Dict[int, int],
    sharpe: float,
    window_days: int,
    fwd_vol_bps_per_h: Optional[Dict[int, float]] = None,
    min_samples: int = 30,
    promote_ic: float = DEFAULT_IC_FLOOR,
    kill_ic: float = 0.05,
    promote_sharpe: float = 0.5,
    round_trip_cost_bps: float = DEFAULT_ROUND_TRIP_COST_BPS,
    cost_margin: float = DEFAULT_COST_MARGIN,
    min_ic_t_stat: float = DEFAULT_MIN_IC_T_STAT,
    require_positive_ic: bool = True,
) -> PromotionAssessment:
    """[P166] The promotion gate, with the reasoning attached.

    The legacy gate promoted on `min(|IC|) > 0.05 and sharpe > 0.5`. Three
    things were wrong with that, each sufficient on its own to promote a
    strategy that is guaranteed to lose money:

      1. NO COST TERM. IC is dimensionless; fees and spread are in bps; the
         two were never converted into common units. IC 0.05 against the ~107
         bps of 16h forward vol these assets show is worth ~4.4 bps per round
         trip, against 6 bps of Coinbase taker fees before any spread. The
         gate's "pass" mark sat *below* break-even.
      2. NO SIGNIFICANCE TERM. SE(IC) ~= 1/sqrt(n-1), so at the shipped
         min_samples=30 an IC of 0.05 is 0.27 SE from zero. The gate could not
         distinguish an edge from a coin flip, and 30 samples is reached in
         about five days of 4H bars.
      3. abs() ON THE PROMOTE BRANCH. `decide_strategy_action` in
         promotion_gate/promotion_plan.py maps PROMOTE straight to
         PROMOTE_TO_FUSION with no sign handling anywhere downstream, so a
         strategy with IC -0.16 (P143 measured exactly that for model_alpha)
         would be promoted and then traded in the direction it predicts
         against.

    KILL and INSUFFICIENT_SAMPLES semantics are unchanged. Every new condition
    only ever *removes* a PROMOTE, so this cannot make the gate looser.
    """
    fwd_vol_bps_per_h = fwd_vol_bps_per_h or {}
    assessment = PromotionAssessment(
        verdict=Verdict.HOLD,
        round_trip_cost_bps=round_trip_cost_bps,
        cost_margin=cost_margin,
    )

    if not n_per_h or all(n < min_samples for n in n_per_h.values()):
        assessment.verdict = Verdict.INSUFFICIENT_SAMPLES
        assessment.blockers.append(
            f"all horizons below min_samples={min_samples}"
        )
        return assessment

    # Use only horizons that have enough samples
    valid_horizons = sorted(h for h, n in n_per_h.items() if n >= min_samples)
    if not valid_horizons:
        assessment.verdict = Verdict.INSUFFICIENT_SAMPLES
        assessment.blockers.append(f"no horizon reaches min_samples={min_samples}")
        return assessment

    valid_ics = [abs(ic_per_h[h]) for h in valid_horizons]

    if window_days <= 14:
        # Short window: KILL aggressively if all IC weak. Unchanged -- a short
        # window is only ever used to cut, never to promote, so the cost and
        # significance bars have nothing to add here.
        if max(valid_ics) < kill_ic:
            assessment.verdict = Verdict.KILL
            assessment.blockers.append(
                f"|IC| < kill_ic={kill_ic} at every valid horizon (short window)"
            )
        else:
            assessment.verdict = Verdict.HOLD
            assessment.blockers.append(
                f"window_days={window_days} <= 14: too short to promote"
            )
        return assessment

    # ---- Longer window (>=30d): every promotion bar must clear ----
    blockers: List[str] = []

    for h in valid_horizons:
        ic_h = float(ic_per_h[h])
        n_h = int(n_per_h[h])
        detail: Dict[str, Any] = {"ic": ic_h, "n": n_h}

        # (2) Significance. t = IC * sqrt(n - 1) is the standard large-sample
        # approximation for a rank correlation.
        t_stat = abs(ic_h) * math.sqrt(max(n_h - 1, 0))
        detail["t_stat"] = t_stat
        if t_stat < min_ic_t_stat:
            blockers.append(
                f"h={h}: IC {ic_h:+.4f} is {t_stat:.2f} SE from zero "
                f"(need |t| >= {min_ic_t_stat}; n={n_h}, "
                f"n_required~={int(math.ceil((min_ic_t_stat / max(abs(ic_h), 1e-9)) ** 2)) + 1})"
            )

        # (1) Costs. Absent vol => FAIL CLOSED. A cost check that could not run
        # is not a cost check that passed (P159, P164).
        vol_bps = fwd_vol_bps_per_h.get(h)
        if vol_bps is None or not math.isfinite(float(vol_bps)) or float(vol_bps) <= 0.0:
            detail["fwd_vol_bps"] = None
            blockers.append(
                f"h={h}: forward-return volatility unavailable — cannot verify "
                f"the edge covers {round_trip_cost_bps:.1f}bps x {cost_margin:.1f} "
                f"of round-trip cost; refusing to promote on an unchecked cost bar"
            )
        else:
            vol_bps = float(vol_bps)
            edge = expected_edge_bps(ic_h, vol_bps)
            need_ic = required_ic_for_costs(vol_bps, round_trip_cost_bps, cost_margin)
            detail["fwd_vol_bps"] = vol_bps
            detail["edge_bps"] = edge
            detail["required_bps"] = round_trip_cost_bps * cost_margin
            detail["required_ic"] = need_ic
            if edge < round_trip_cost_bps * cost_margin:
                blockers.append(
                    f"h={h}: edge {edge:.2f}bps < required "
                    f"{round_trip_cost_bps * cost_margin:.2f}bps "
                    f"({round_trip_cost_bps:.1f}bps round trip x {cost_margin:.1f} margin); "
                    f"IC {abs(ic_h):.4f} vs required {need_ic:.4f} on "
                    f"{vol_bps:.1f}bps forward vol"
                )

        # (3) Sign. Only checked on the promote path; KILL still uses |IC|.
        if require_positive_ic and ic_h <= 0.0:
            blockers.append(
                f"h={h}: IC {ic_h:+.4f} is not positive — nothing downstream "
                f"inverts a negative-IC strategy, so promoting it would trade "
                f"the signal backwards"
            )

        assessment.per_horizon[h] = detail

    # Absolute IC floor, retained from the legacy gate.
    if min(valid_ics) <= promote_ic:
        blockers.append(
            f"min |IC| {min(valid_ics):.4f} <= floor {promote_ic}"
        )

    if sharpe <= promote_sharpe:
        blockers.append(f"sharpe {sharpe:+.2f} <= {promote_sharpe}")

    if not blockers:
        assessment.verdict = Verdict.PROMOTE
        return assessment

    assessment.blockers = blockers
    if max(valid_ics) < kill_ic:
        assessment.verdict = Verdict.KILL
    else:
        assessment.verdict = Verdict.HOLD
    return assessment


def determine_verdict(
    ic_per_h: Dict[int, float],
    n_per_h: Dict[int, int],
    sharpe: float,
    window_days: int,
    min_samples: int = 30,
    promote_ic: float = DEFAULT_IC_FLOOR,
    kill_ic: float = 0.05,
    promote_sharpe: float = 0.5,
    fwd_vol_bps_per_h: Optional[Dict[int, float]] = None,
    **kwargs: Any,
) -> Verdict:
    """Thin wrapper over `assess_promotion` for callers that want only the
    verdict. Kept so `promotion_gate/promotion_plan.py` and the existing tests
    keep working unchanged. New code should prefer `assess_promotion`, which
    also returns *why*."""
    return assess_promotion(
        ic_per_h,
        n_per_h,
        sharpe,
        window_days,
        fwd_vol_bps_per_h=fwd_vol_bps_per_h,
        min_samples=min_samples,
        promote_ic=promote_ic,
        kill_ic=kill_ic,
        promote_sharpe=promote_sharpe,
        **kwargs,
    ).verdict


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def assess_record(record: Dict[str, Any], window_days: int) -> PromotionAssessment:
    """[P166] Single source of truth for "what does this IC record mean?".

    `render_summary` and the JSON report used to call `determine_verdict`
    separately with their own argument lists. Two call sites deriving the same
    verdict independently is how a console PROMOTE and a report HOLD end up in
    the same run; route both through here instead."""
    return assess_promotion(
        record.get("ic_per_horizon", {}) or {},
        record.get("n_per_horizon", {}) or {},
        record.get("annualized_sharpe", 0.0) or 0.0,
        window_days,
        fwd_vol_bps_per_h=record.get("fwd_vol_bps_per_horizon", {}) or {},
    )


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
        assessment = assess_record(v, window_days)
        n_max = max(n_per_h.values()) if n_per_h else 0
        ic_strs = " ".join(f"{ic_per_h.get(h, 0.0):+.3f}" for h in horizons_bars)
        lines.append(
            f"  {strat:<24} {asset:<5} {n_max:>6} {ic_strs} {sharpe:+8.2f} "
            f"{assessment.verdict.value:>20}"
        )
        # [P166] A verdict without its arithmetic is not auditable. Print the
        # edge-vs-cost line for every horizon, then why promotion was refused.
        for h in horizons_bars:
            d = assessment.per_horizon.get(h)
            if not d:
                continue
            if d.get("edge_bps") is None:
                lines.append(f"      h={h:<3} edge=UNKNOWN (no forward vol) t={d['t_stat']:.2f}")
            else:
                lines.append(
                    f"      h={h:<3} edge={d['edge_bps']:6.2f}bps  "
                    f"need={d['required_bps']:6.2f}bps  "
                    f"vol={d['fwd_vol_bps']:8.1f}bps  "
                    f"IC={d['ic']:+.4f} (req {d['required_ic']:.4f})  t={d['t_stat']:.2f}"
                )
        for b in assessment.blockers:
            lines.append(f"      BLOCKED: {b}")
    lines.append("=" * 90)
    lines.append(
        f"  cost model: {DEFAULT_ROUND_TRIP_COST_BPS:.1f}bps round trip "
        f"x {DEFAULT_COST_MARGIN:.1f} margin | significance: |t| >= "
        f"{DEFAULT_MIN_IC_T_STAT:.1f} | IC floor: {DEFAULT_IC_FLOOR}   [P166]"
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compute IC + verdict on shadow-strategy ledgers.")
    p.add_argument("--ledger-dir", default=str(LEDGER_DIR))
    p.add_argument("--window-days", type=int, default=14)
    p.add_argument("--horizons", default="4,12,24",
                   help="forward-return horizons in 4H bars, comma-separated")
    p.add_argument("--prefixes",
                   default="microstructure,cascade,funding,ml_factor",  # [P199]
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

    # [P213] "No price series for ANY asset" is a WRONG-ENVIRONMENT error, not a
    # result. Without this the run prints a full table of ohlcv_missing and
    # writes a report — indistinguishable from "the strategies have no signal",
    # which is exactly the conflation that hid P199 for months. Refuse, name the
    # cause, name the fix.
    #
    # Deliberately ALL, not ANY: one missing asset is a genuine data gap the run
    # should report per-strategy and continue through. Every asset missing means
    # the tool is running somewhere it cannot work.
    _priced = [v for v in per_strategy.values() if v.get("error") != "ohlcv_missing"]
    if per_strategy and not _priced:
        _assets = sorted({a for (_s, a) in per_strategy})
        print(
            f"REFUSING TO REPORT: no 4H OHLCV price series for ANY asset "
            f"({', '.join(_assets)}) in {OHLCV_DIR}.\n"
            f"  This is almost certainly the wrong environment, not a result. "
            f"`training/training_data/` is excluded by .dockerignore, so the "
            f"parquets are NOT in the engine image or in CI.\n"
            f"  FIX: run this on the operator machine, refreshing first:\n"
            f"    python -X utf8 training/scripts/refresh_ohlcv_4h.py\n"
            f"  Emitting an all-`ohlcv_missing` report instead would read as "
            f"'the strategies have no signal' — the P199 conflation.",
            file=sys.stderr,
        )
        return 2

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
                # [P166] Same assessment object the console summary printed.
                **assess_record(v, args.window_days).to_dict(),
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
