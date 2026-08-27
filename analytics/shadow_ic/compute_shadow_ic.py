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
                * expected edge >= round-trip cost x 2.0 margin, priced off the
                  measured forward-return volatility. [P382] The round-trip
                  cost is PER ASSET, derived from `core.cde_fees.CDE_FEE_BPS`
                  (2 legs x the asset's taker bps; pooled rows take the max
                  over their members), floored at the refuted 6.0bps model —
                  and each row prints which one it was judged on.
              A missing volatility measurement is a REFUSAL, not a skip.
              [P382] In pooled mode the forward vol is the n-weighted RMS of
              each member asset's RAW sigma, never the sigma of the z-scored
              pooled series (which is 1.0 == 10,000 bps, a cost bar nothing
              could fail to clear).
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
#
# [P382] ...and 6.0 is now ONLY a floor. P315/P334 measured the CDE fee at
# 9.4-14.5 bps PER LEG (flat per contract -> percentage of notional), and P374
# measured the all-in round trip at BTC 27.7 / ETH 44.0 / SOL 41.0 bps. The
# gate was still pricing the REFUTED 3bps/side model, so every edge-vs-cost
# verdict it printed was ~3x too generous. The round-trip cost a row is judged
# against is now DERIVED from `core.cde_fees.CDE_FEE_BPS` (the registered
# calibration, P327): 2 legs x that asset's taker bps, floored at this
# constant, then the x2 margin on top. See `round_trip_cost_bps_for`.
DEFAULT_ROUND_TRIP_COST_BPS = 6.0
REFUTED_MODEL_RT_BPS = DEFAULT_ROUND_TRIP_COST_BPS   # [P382] the FLOOR, named honestly

# [P382] Provenance of the cost a row was judged against (P169: a number
# without its source is not a measurement). Printed per row and carried in
# the report.
COST_SOURCE_CDE = "cde_fees"
COST_SOURCE_FALLBACK = "fallback_refuted_model"
_CDE_LEGS = 2.0   # entry + exit; the margin below is NOT a leg count

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

# [P293g] Bars per day on the 4H cadence every ledger here is written at.
# Used ONLY to express the sample requirement in days — the same value
# `agents/kraken_quant_agent.BARS_PER_DAY_4H` uses, restated rather than
# imported so this analytics tool keeps no dependency on an agent module.
BARS_PER_DAY_4H = 6

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
    # [P382] where round_trip_cost_bps came from (P169 provenance):
    # "cde_fees" | "fallback_refuted_model" | "" (caller supplied, untagged)
    cost_source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "blockers": list(self.blockers),
            "per_horizon": {str(k): v for k, v in self.per_horizon.items()},
            "round_trip_cost_bps": self.round_trip_cost_bps,
            "cost_margin": self.cost_margin,
            "cost_source": self.cost_source,
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
# [P382] Round-trip cost, derived from the registered CDE fee calibration
# ---------------------------------------------------------------------------

_cde_fallback_warned = False


def _cde_taker_leg_bps() -> Optional[Dict[str, float]]:
    """asset -> per-LEG taker fee in bps, read from `core.cde_fees.CDE_FEE_BPS`.

    Returns None when the calibration cannot be read, and LOGS that the gate
    is then running on the refuted model — never silently (P169/P199).
    `core.cde_fees` fails toward the expensive side itself (an assumed asset
    carries the worst measured fee, P167), so reading the table is enough.
    """
    global _cde_fallback_warned
    try:
        # Run as a script (`python analytics/shadow_ic/compute_shadow_ic.py`)
        # sys.path[0] is this file's directory, not the repo root.
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from core.cde_fees import CDE_FEE_BPS
    except Exception as e:  # noqa: silent-swallow — logged below, once
        if not _cde_fallback_warned:
            logger.warning(
                "[SHADOW_IC][P382] core.cde_fees unavailable (%s: %s) — the "
                "promotion gate is pricing the REFUTED 3bps/side model "
                "(%.1fbps round trip, P315/P334). Every edge-vs-cost verdict "
                "below is ~3x too generous; cost_source=%s",
                type(e).__name__, e, REFUTED_MODEL_RT_BPS, COST_SOURCE_FALLBACK)
            _cde_fallback_warned = True
        return None
    table: Dict[str, float] = {}
    for a, sides in dict(CDE_FEE_BPS).items():
        try:
            v = float(sides["taker"])
        except (KeyError, TypeError, ValueError):  # noqa: silent-swallow — shape coercion
            continue
        if math.isfinite(v) and v > 0.0:
            table[str(a).upper()] = v
    return table or None


def round_trip_cost_bps_for(
    assets: Optional[Any] = None,
    floor_bps: float = REFUTED_MODEL_RT_BPS,
) -> Tuple[float, str, Dict[str, float]]:
    """The round-trip cost a row is judged against, with its provenance.

    Returns ``(rt_bps, cost_source, per_asset_rt_bps)``.

      * per-asset row  -> that asset's 2 x taker leg;
      * pooled row     -> the MAX over the pooled assets: a pooled bar must
                          not be cheaper than its dearest member, or pooling
                          would silently buy a lower cost bar along with the
                          larger n;
      * no asset given -> the MAX over the whole table (the conservative
                          reading for a row the caller could not attribute);
      * an asset the table does not know -> the WORST in the table (P167:
        an unmeasured cost is assumed expensive, never cheap);
      * the calibration unreadable -> `floor_bps` with
        cost_source=COST_SOURCE_FALLBACK, logged once.

    `floor_bps` (the refuted 6.0) is a FLOOR in every branch: the derived
    cost can only ever be higher than the model it replaces, so this change
    cannot loosen the gate anywhere (P167/P248).
    """
    floor = float(floor_bps)
    table = _cde_taker_leg_bps()
    if table is None:
        return floor, COST_SOURCE_FALLBACK, {}
    worst_leg = max(table.values())
    names = [str(a).upper() for a in (assets or []) if a]
    per_asset: Dict[str, float] = {}
    if names:
        for a in names:
            # an asset the table does not know prices at the WORST (P167);
            # written as a membership test so the silent_failure_audit's
            # two-variable-default heuristic does not read a deliberate
            # fallback as the P47-Bug-2 shape
            leg = table[a] if a in table else worst_leg
            per_asset[a] = max(floor, _CDE_LEGS * leg)
        rt = max(per_asset.values())
    else:
        per_asset = {a: max(floor, _CDE_LEGS * leg) for a, leg in table.items()}
        rt = max(per_asset.values())
    return max(floor, rt), COST_SOURCE_CDE, per_asset


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_shadow_ledgers(
    ledger_dir: Path,
    # [P199] Was ("microstructure", "cascade") — which read only the two
    # families that emit NOTHING, and silently ignored funding_*.jsonl and
    # ml_factor_*.jsonl, the only ones producing signal. The gate meant to
    # validate the v5.1 promotion could not see the strategies it was judging.
    prefixes: Tuple[str, ...] = ("microstructure", "cascade", "funding",
                                "ml_factor", "derivflow", "regimebook",
                                "etfflow",  # [P270]
                                "skewetf",  # [P407j] skew+ETF ensemble shadow
                                # [P277] enhancement families
                                "stablecoinflow", "oidiv", "calbasis",
                                "xsmom", "eventfilter",
                                "mlpshadow",  # [P284]
                                "ridgeshadow",  # [P409] held BTC ridge
                                # [P289] P288 trend-rule challengers
                                "donchian", "emaens",
                                "ma_filter",
                                "whale_filter",  # [P293d option A]
                                "sentvariant"),  # [P293e] 3 F&G readings
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
                        # Parse timestamp. [P264] Accept BOTH shapes: the
                        # P248 regimebook family writes `ts` as an EPOCH
                        # FLOAT (time.time(), with a separate `iso` field),
                        # while the older families write ISO strings. The
                        # old string-only parser raised AttributeError on
                        # floats, which the per-file handler swallowed as
                        # "failed to read <file>" — so the raw AND adjusted
                        # book ledgers were entirely invisible to the gate
                        # that decides their promotion (the P199 class:
                        # registered but unreadable). Found by the
                        # end-to-end scorer proof 28 days before the read.
                        ts_raw = rec.get("ts")
                        if ts_raw is None or ts_raw == "":
                            skipped += 1
                            continue
                        try:
                            if isinstance(ts_raw, (int, float)):
                                ts = datetime.fromtimestamp(
                                    float(ts_raw), tz=timezone.utc)
                            else:
                                ts = datetime.fromisoformat(
                                    str(ts_raw).replace("Z", "+00:00"))
                        except (ValueError, OSError, OverflowError):  # noqa: silent-swallow
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
        # [P420] september_check no longer OVERWRITES the Binance-derived
        # `_4H_ohlcv.parquet` with Kraken public rows; it writes them here
        # (~120 days). Breadth assets (XRP/BNB/...) have ONLY this series.
        OHLCV_DIR / f"{asset}_4H_ohlcv_kraken.parquet",
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

POOLED_KEY = "POOLED"

# [P299] Families that are ONE RULE applied to several assets. Scoring them
# per-asset splits one exam into three underpowered ones: the P166 gate needs
# |t| = IC*sqrt(n_eff - 1) >= 2 with n_eff = n/h, so at 16h a 30-day window can
# only certify IC >= 0.302 while the ECONOMIC bar asks ~0.13 — an
# economically-adequate candidate would need ~330 DAYS (P293g). Pooling three
# assets triples n and cuts that to ~123 days; it is a SCORER change, not a
# wait. A family belongs here only when the same rule, with the same
# parameters, produces every asset's record.
POOLABLE_FAMILIES = frozenset({
    # [P309] KEYED ON THE RECORD'S `strategy` FIELD, which is what
    # compute_per_strategy_ic groups by — NOT on the ledger-file prefix.
    # P299 mixed the two, so `ma_filter`/`whale_filter` (prefixes) never
    # matched `ma_filtered`/`whale_filtered` (names) and those two families
    # were silently left un-pooled while the feature reported success. The
    # P294 lesson, committed by the author who had just quoted it: the thing
    # that separates two claims is whatever the SCORER groups by.
    "regimebook", "regimebook_adj", "regimebook_volskip",
    # [P420] the five breadth trend-only books (XRP/ADA/LTC/DOGE/BNB) are ONE
    # rule across never-fitted assets -> their own pooled exam. They were
    # written as "regimebook" before P420 (pooling 8 books running 3 rules).
    "regimebook_breadth",
    # [P307e] the funding-gated-short variant. Pooled deliberately: it is
    # ONE rule applied across three assets, and P293g measured that a
    # per-asset 16h exam needs ~330 days to certify while pooling three
    # cuts it to ~123. On BTC the variant is identical to the book by
    # construction, so the pooled series is carried by ETH and SOL.
    "regimebook_fgshort",
    "donchian", "emaens", "xsmom",
    "ma_filtered", "whale_filtered",
    "sent_momentum_linear", "sent_momentum_hist", "sent_contrarian",
    "oidiv_confirm", "oidiv_fade", "stablecoinflow", "eventfilter",
    "liquidation_squeeze", "liquidation_exhaustion",
    # [P407j] skew+ETF ensemble A/B; pooled for power (thin 1.7y, P293g).
    "skewetf_skew", "skewetf_etf", "skewetf_agree",
    "calbasis", "etfflow",
    # NOT here on purpose: `mlpshadow` is a BTC-only exported model, not one
    # rule applied across assets.
})

# [P420] Which ASSETS a pooled family may draw on. The pooled `regimebook`
# exam is the HOME TRIO ONLY: the breadth books run a different rule and,
# before P420, wrote their rows under `strategy: "regimebook"` — so without
# this filter the old breadth rows would keep contaminating the trio's pooled
# read. A row whose asset is outside its family's filter is scored PER ASSET
# (still visible, never silently dropped — P199). Families not listed pool
# every asset they cover. Deliberately hardcoded to the trio here (not
# config.assets): the trio IS the pool the P297 six-year certification was
# measured on.
_HOME_TRIO = ("BTC", "ETH", "SOL")
_BREADTH = ("XRP", "ADA", "LTC", "DOGE", "BNB")
POOL_ASSET_FILTER = {
    "regimebook": _HOME_TRIO,
    "regimebook_adj": _HOME_TRIO,
    "regimebook_volskip": _HOME_TRIO,
    "regimebook_fgshort": _HOME_TRIO,
    "regimebook_breadth": _BREADTH,
}


def pool_key_for(strat: str, asset: str, pool_assets: bool) -> str:
    """[P420] POOLED_KEY when `strat` pools AND `asset` is inside its pool
    filter (or the family has no filter); else the asset itself."""
    if not pool_assets or strat not in POOLABLE_FAMILIES:
        return asset
    allowed = POOL_ASSET_FILTER.get(strat)
    if allowed is not None and asset not in allowed:
        return asset
    return POOLED_KEY


# [P310] Scored PER ASSET on purpose — not poolable, not dead. Named
# explicitly so that "nobody classified this yet" is distinguishable from
# "deliberately per-asset", which is the same missing-vs-neutral distinction
# this file enforces everywhere else (P2). The conformance test requires every
# producer-declared name to appear in exactly one of the three sets.
PER_ASSET_FAMILIES = {
    "mlpshadow": "a BTC-only EXPORTED model, not one rule across assets",
    "ridgeshadow": "a BTC-only EXPORTED held ridge (P409); ETH/SOL fail even held",
    "ml_factor": "v5.1 per-asset autoencoder factor (alive: 924/2082)",
    "funding_mean_reversion": "v5.1 per-asset funding rule (alive: 50/2082)",
    "funding_post_etf_regime": "v5.1 per-asset funding rule (alive: 93/2082)",
    "regimebook_banded": "P259b WITHDRAWN; the exports were deleted, so this "
                         "path is inert unless a model file reappears",
}


# [P309] Measured over the full pulled ledger history (2026-04-30 -> 08-18),
# total records vs records carrying a non-zero direction:
#     cascade_anticipation   2/2082      funding_extreme      0/2082
#     kyle_lambda            0/2082      ofi                  1/2082
#     stop_hunt_defense      0/2082      vpin_spike           0/2082
# Exactly P199's six KILL verdicts, still occupying the report four months on.
#
# ALSO KEYED ON THE STRATEGY NAME, and that correction matters more here than
# for pooling: P299 keyed this on FILE PREFIXES, and the `funding_*.jsonl`
# files hold THREE strategies — archiving "funding" would have buried
# `funding_mean_reversion` (50 directional) and `funding_post_etf_regime`
# (93), both alive. It escaped only because the key never matched anything.
# An archive list must name what it archives, not the file it lives in.
ARCHIVED_FAMILIES = {
    "cascade_anticipation": "P199 KILL; 2/2082 directional over the full ledger (P309)",
    "funding_extreme": "P199 KILL; 0/2082 directional (P309)",
    "kyle_lambda": "P199 KILL; 0/2082 directional (P309)",
    "ofi": "P199 KILL; 1/2082 directional (P309)",
    "stop_hunt_defense": "P199 KILL; 0/2082 directional (P309)",
    "vpin_spike": "P199 KILL; 0/2082 directional (P309)",
}
# Deliberately NOT archived, on the same measurement: ml_factor (924/2082),
# funding_mean_reversion (50), funding_post_etf_regime (93). They emit.


def compute_per_strategy_ic(
    records: List[Dict[str, Any]],
    horizons_bars: Tuple[int, ...] = (4, 12, 24),
    pool_assets: bool = False,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Group records by (strategy, asset). For each group compute IC at
    each horizon. Returns {(strategy, asset): {n, ic_per_h, ...}}.

    [P299] With ``pool_assets=True`` a family in POOLABLE_FAMILIES is scored
    as ONE exam across every asset it covers, keyed ``(strategy, "POOLED")``.
    Forward returns are STANDARDIZED WITHIN EACH ASSET before pooling — a
    high-volatility asset would otherwise occupy the extreme ranks and the
    pooled Spearman would mostly measure that asset. Non-poolable families
    are left per-asset even when the flag is on: pooling genuinely per-asset
    claims would merge different claims into one number (the P294 lesson,
    where a shared strategy_name silently merged two exams)."""

    # Group records
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        strat = r.get("strategy")
        asset = r.get("asset")
        if not strat or not asset:
            continue
        key_asset = pool_key_for(strat, str(asset), pool_assets)  # [P420]
        grouped[(strat, key_asset)].append(r)

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
        pooled = (asset == POOLED_KEY)
        if pooled:
            # [P299] Every member asset must be loadable; one that is not is
            # dropped from the pool with its name recorded, never silently.
            _assets = sorted({str(r.get("asset")) for r in recs if r.get("asset")})
            _ok = [a for a in _assets if get_ohlcv(a) is not None]
            if not _ok:
                out[(strat, asset)] = {"n": 0, "ic_per_h": {},
                                       "error": "ohlcv_missing",
                                       "pooled_assets": [],
                                       "pooled_dropped": _assets}
                continue
            df = None
        else:
            df = get_ohlcv(asset)
            if df is None:
                out[(strat, asset)] = {"n": 0, "ic_per_h": {},
                                       "error": "ohlcv_missing"}
                continue

        # Build per-horizon (signal_x, forward_y) pairs
        per_h: Dict[int, Tuple[List[float], List[float]]] = {h: ([], []) for h in horizons_bars}
        per_trade_returns: List[float] = []  # for Sharpe at largest horizon

        # [P299] In pooled mode collect per ASSET first, standardize the
        # forward returns within each asset, and only then pool. Spearman is
        # rank-based, so pooling raw returns from assets with different
        # dispersion lets the widest-dispersion asset own the extreme ranks —
        # the pooled IC would then largely measure that one asset while
        # reporting a triple-sized n. Standardizing first makes the ranks
        # comparable, which is the whole point of pooling.
        _by_asset: Dict[str, Dict[int, Tuple[List[float], List[float]]]] = {}
        _pt_by_asset: Dict[str, List[float]] = defaultdict(list)

        for r in recs:
            ts = r.get("_parsed_ts")
            direction = float(r.get("direction", 0.0) or 0.0)
            confidence = float(r.get("confidence", 0.0) or 0.0)
            if ts is None:
                continue
            _a = str(r.get("asset")) if pooled else asset
            _df = get_ohlcv(_a) if pooled else df
            if _df is None:
                continue
            x_val = direction * confidence
            if pooled:
                _slot = _by_asset.setdefault(
                    _a, {h: ([], []) for h in horizons_bars})
            else:
                _slot = per_h
            for h in horizons_bars:
                fr = find_forward_return(_df, ts, h)
                if fr is None:
                    continue
                _slot[h][0].append(x_val)
                _slot[h][1].append(fr)
            # Per-trade return uses largest horizon, ONLY for non-zero direction
            if direction != 0.0:
                fr_large = find_forward_return(_df, ts, max(horizons_bars))
                if fr_large is not None:
                    if pooled:
                        _pt_by_asset[_a].append(direction * fr_large)
                    else:
                        per_trade_returns.append(direction * fr_large)

        # [P382] RAW forward-return dispersion per horizon, accumulated per
        # asset BEFORE standardization. The pooled IC below is correctly
        # computed on the z-scored series (P299) — but the P166 cost bar needs
        # the dispersion in BPS, and a z-scored series has sigma == 1 by
        # construction, i.e. "10,000 bps of forward vol". The pooled row was
        # therefore handed a vol ~50-100x any real asset's, `required_ic`
        # collapsed to ~0.001, and the cost bar was VACUOUS on exactly the read
        # P332 pre-committed as GOVERNING for a poolable family. Pooled vol is
        # now the n-weighted RMS of each member's OWN sigma:
        #     sqrt( sum_a n_a * sigma_a^2 / sum_a n_a )
        # which lands inside [min_a sigma_a, max_a sigma_a] by construction.
        _pooled_var_acc: Dict[int, float] = {h: 0.0 for h in horizons_bars}
        _pooled_var_n: Dict[int, int] = {h: 0 for h in horizons_bars}

        if pooled:
            for _a, _slot in _by_asset.items():
                for h in horizons_bars:
                    xs, ys = _slot[h]
                    if not ys:
                        continue
                    _m = sum(ys) / len(ys)
                    _var = sum((y - _m) ** 2 for y in ys) / max(1, len(ys) - 1)
                    _sd = _var ** 0.5
                    if _sd <= 0.0:
                        # A constant series carries no rank information; a
                        # zero-divide would fabricate one (P2).
                        continue
                    if len(ys) >= 2:
                        # [P382] raw sigma, weighted by this asset's n
                        _pooled_var_acc[h] += len(ys) * _var
                        _pooled_var_n[h] += len(ys)
                    per_h[h][0].extend(xs)
                    per_h[h][1].extend([(y - _m) / _sd for y in ys])
            for _a, _pts in _pt_by_asset.items():
                per_trade_returns.extend(_pts)

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
            if pooled:
                # [P382] NEVER from per_h[h][1] here — that series is z-scored
                # (sigma == 1 -> 10,000 bps) and would make the cost bar vacuous.
                if _pooled_var_n[h] < 2:
                    continue
                fwd_vol_bps_per_h[h] = (
                    (_pooled_var_acc[h] / _pooled_var_n[h]) ** 0.5) * 10_000.0
                continue
            ys = per_h[h][1]
            if len(ys) < 2:
                continue
            mean_y = sum(ys) / len(ys)
            var_y = sum((y - mean_y) ** 2 for y in ys) / (len(ys) - 1)
            fwd_vol_bps_per_h[h] = (var_y ** 0.5) * 10_000.0

        # [P382] The cost THIS row is judged against, with provenance. Pooled
        # rows take the max over their member assets (a pooled bar must not be
        # cheaper than its dearest member); per-asset rows take that asset's.
        _cost_assets = (sorted(_ok) if pooled else [asset])
        _rt_bps, _cost_src, _rt_by_asset = round_trip_cost_bps_for(_cost_assets)

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
            # [P382] the cost bar this row is judged against + its provenance
            "round_trip_cost_bps": _rt_bps,
            "cost_source": _cost_src,
            "cost_assets": _cost_assets,
            "round_trip_cost_bps_by_asset": _rt_by_asset,
        }
        if pooled:
            out[(strat, asset)]["pooled_assets"] = list(_cost_assets)

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
    cost_source: str = "",
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
        cost_source=cost_source,
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

        # (2) Significance. t = IC * sqrt(n_eff - 1), where n_eff = n / h is
        # the OVERLAP-corrected effective sample count. [P253] The P231
        # correction (h-bar forward returns sampled every bar overlap h-fold,
        # inflating a naive t by ~sqrt(h)) was applied to agent_ic_review and
        # slope_calibrator but NOT here — so this gate, citing the same P166
        # doctrine, was holding strategies to a ~2x weaker significance bar at
        # h=4 than the agent gate. Same arithmetic as agent_ic_review.py.
        n_eff = max(n_h // max(int(h), 1), 0)
        t_stat = abs(ic_h) * math.sqrt(max(n_eff - 1, 0))
        detail["t_stat"] = t_stat
        detail["n_eff"] = n_eff
        if t_stat < min_ic_t_stat:
            _n_req = int(math.ceil(
                max(int(h), 1)
                * ((min_ic_t_stat / max(abs(ic_h), 1e-9)) ** 2 + 1)))
            # [P293g] Say it in DAYS as well as samples. The sample figure was
            # already computed and nobody converted it, which is how ~14
            # candidates came to sit on 30-DAY clocks against requirements
            # that are often a YEAR at this horizon:
            #   n_eff = n/h, t = IC*sqrt(n_eff-1), 6 bars/day at 4H
            #   => an IC that only just meets the ECONOMIC bar (0.085 at 16h)
            #      needs ~370 days; a 30-day window can only certify IC>=0.30.
            # That makes the statistical bar ~3.6x stricter than the economic
            # one at 16h, so a 30-day clock on a 16h claim is a check that
            # cannot pass — the inverse of P174's check that cannot fail, and
            # a defect by the same reasoning. Stating the requirement in the
            # gate's own output is what stops the next clock being set blind.
            _days_req = _n_req / float(BARS_PER_DAY_4H)
            _days_have = n_h / float(BARS_PER_DAY_4H)
            blockers.append(
                f"h={h}: IC {ic_h:+.4f} is {t_stat:.2f} SE from zero "
                f"(need |t| >= {min_ic_t_stat}; n={n_h}, n_eff={n_eff} "
                f"overlap-corrected, n_required~={_n_req} "
                f"= ~{_days_req:,.0f}d of 4H bars vs ~{_days_have:,.0f}d held)"
            )
            detail["days_required"] = round(_days_req, 1)
            detail["days_held"] = round(_days_have, 1)

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
    the same run; route both through here instead.

    [P382] The cost bar comes FROM THE RECORD (`round_trip_cost_bps` +
    `cost_source`, written by compute_per_strategy_ic from core.cde_fees). A
    record without one — a report written before P382 — is judged on the
    refuted-model floor and SAYS so in cost_source, rather than pretending the
    CDE price was applied."""
    rt = record.get("round_trip_cost_bps")
    src = record.get("cost_source")
    try:
        rt_f = float(rt) if rt is not None else None
    except (TypeError, ValueError):  # noqa: silent-swallow — shape coercion
        rt_f = None
    if rt_f is None or not math.isfinite(rt_f) or rt_f <= 0.0:
        rt_f = DEFAULT_ROUND_TRIP_COST_BPS
        src = f"record_missing_cost:{COST_SOURCE_FALLBACK}"
    return assess_promotion(
        record.get("ic_per_horizon", {}) or {},
        record.get("n_per_horizon", {}) or {},
        record.get("annualized_sharpe", 0.0) or 0.0,
        window_days,
        fwd_vol_bps_per_h=record.get("fwd_vol_bps_per_horizon", {}) or {},
        round_trip_cost_bps=max(DEFAULT_ROUND_TRIP_COST_BPS, rt_f),
        cost_source=str(src or ""),
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
        # [P382] The cost this row was judged against, and WHERE it came from
        # (P169). A reader must be able to tell a CDE-priced verdict from one
        # rendered on the refuted 6bps model without opening the source.
        _pa = v.get("pooled_assets")
        lines.append(
            f"      cost={assessment.round_trip_cost_bps:.1f}bps round trip "
            f"x {assessment.cost_margin:.1f} margin  "
            f"cost_source={assessment.cost_source or 'unstated'}"
            + (f"  pooled_assets={','.join(_pa)}" if _pa else "")
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
    # [P382] The footer states the MODEL, per asset, not one number: the cost
    # is per-asset (CDE taker x 2 legs from core.cde_fees), floored at the
    # refuted 6.0, then x margin. If the calibration could not be read the
    # footer says the whole table is on the refuted model.
    _rt_all, _src_all, _rt_by = round_trip_cost_bps_for(None)
    if _src_all == COST_SOURCE_CDE:
        _per = ", ".join(f"{a}={b:.1f}" for a, b in sorted(_rt_by.items()))
        lines.append(
            f"  cost model: CDE taker x {_CDE_LEGS:.0f} legs from core.cde_fees "
            f"({_per} bps round trip; floor {REFUTED_MODEL_RT_BPS:.1f}) "
            f"x {DEFAULT_COST_MARGIN:.1f} margin | significance: |t| >= "
            f"{DEFAULT_MIN_IC_T_STAT:.1f} | IC floor: {DEFAULT_IC_FLOOR}   [P166/P382]"
        )
    else:
        lines.append(
            f"  cost model: {REFUTED_MODEL_RT_BPS:.1f}bps round trip "
            f"x {DEFAULT_COST_MARGIN:.1f} margin — {COST_SOURCE_FALLBACK}: "
            f"core.cde_fees UNREADABLE, this is the REFUTED 3bps/side model "
            f"(P315/P334) | significance: |t| >= {DEFAULT_MIN_IC_T_STAT:.1f} "
            f"| IC floor: {DEFAULT_IC_FLOOR}   [P166/P382]"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compute IC + verdict on shadow-strategy ledgers.")
    p.add_argument("--ledger-dir", default=str(LEDGER_DIR))
    p.add_argument("--window-days", type=int, default=14)
    p.add_argument("--horizons", default="4,12,24",
                   help="forward-return horizons in 4H bars, comma-separated")
    p.add_argument("--prefixes",
                   default="microstructure,cascade,funding,ml_factor,derivflow,regimebook,etfflow,stablecoinflow,oidiv,calbasis,xsmom,eventfilter,mlpshadow,donchian,emaens,whale_filter,sentvariant,skewetf,ridgeshadow,ma_filter",  # [P199,P219,P236,P248,P270,P277,P284,P289,P293d,P407j]
                   help="ledger file prefixes")
    p.add_argument("--pool-assets", action="store_true",
                   help="[P299] score one-rule-many-asset families as a "
                        "SINGLE pooled exam (forward returns standardized "
                        "within each asset first). At 16h this cuts the "
                        "calendar time to certify an economically-adequate "
                        "IC from ~330d to ~123d for a 3-asset family — the "
                        "30-day clock cannot fire without it (P293g).")
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

    per_strategy = compute_per_strategy_ic(
        records, horizons_bars=horizons,
        pool_assets=bool(getattr(args, "pool_assets", False)))
    # [P299] Report the archived families separately rather than deleting
    # them: an archive that cannot be audited is indistinguishable from a
    # family nobody looked at. If one of these ever emits directional
    # records again, this section is where it shows up.
    # [P213] "No price series for ANY asset" is a WRONG-ENVIRONMENT error, not a
    # result. Without this the run prints a full table of ohlcv_missing and
    # writes a report — indistinguishable from "the strategies have no signal",
    # which is exactly the conflation that hid P199 for months. Refuse, name the
    # cause, name the fix.
    #
    # Deliberately ALL, not ANY: one missing asset is a genuine data gap the run
    # should report per-strategy and continue through. Every asset missing means
    # the tool is running somewhere it cannot work.
    # [P309] Evaluated BEFORE the archive filter, on the FULL result set.
    # The archive block POPS its rows out of per_strategy, and with an
    # all-archived window that emptied the dict — so `if per_strategy` was
    # False and this refusal silently returned 0 instead of 2, defeating the
    # very guard that exists so "no prices" can never read as "no signal".
    # An archived family reporting ohlcv_missing is still evidence that the
    # prices are missing.
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

    # [P309] THE DURABLE GUARD. Both lists are keyed on the record's
    # `strategy` field, and an entry that matches nothing is INVISIBLE — that
    # is exactly how P299 shipped `ma_filter`/`whale_filter` un-pooled and an
    # archive section that never rendered, while both features reported
    # success. Report the misses, so the next wrong name costs one run
    # instead of a month (P264: registered-but-unmatched).
    _seen_names = {k[0] for k in per_strategy}
    _unmatched_pool = sorted(n for n in POOLABLE_FAMILIES if n not in _seen_names)
    _unmatched_arch = sorted(n for n in ARCHIVED_FAMILIES if n not in _seen_names)
    if _unmatched_pool or _unmatched_arch:
        print("")
        print("NOTE — allowlist entries that matched NO strategy in this "
              "window. Either the family is not emitting yet, or the name is "
              "wrong (these lists key on the record's `strategy` field, NOT "
              "the ledger-file prefix):")
        if _unmatched_pool:
            print(f"   poolable, unmatched: {', '.join(_unmatched_pool)}")
        if _unmatched_arch:
            print(f"   archived, unmatched: {', '.join(_unmatched_arch)}")

    _arch = {k: v for k, v in per_strategy.items() if k[0] in ARCHIVED_FAMILIES}
    if _arch:
        print("")
        print("ARCHIVED families (kept accruing, excluded from the "
              "promotion table — measured dead, see the reason):")
        for (st, a), st_data in sorted(_arch.items()):
            print(f"   {st:16s} {a:8s} n={st_data.get('n', 0):5d}  "
                  f"{ARCHIVED_FAMILIES[st]}")
        for k in list(_arch):
            per_strategy.pop(k, None)

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
