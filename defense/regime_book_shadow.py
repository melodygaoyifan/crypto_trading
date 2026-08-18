"""[P248-GP2] Regime-book shadow harness — the P247 roster, observation-only.

Records, every 4H tick, the target position each leak-corrected regime book
would hold, to `data/strategy_shadow/regimebook_{ASSET}.jsonl` — the P219
ledger pattern, scoreable by `compute_shadow_ic`, feeding the 30d P166
forward gate. NO orders are placed or influenced (Iron Law 7).

The v1 books (everything below is computable EXACTLY at runtime from a
>=560-bar 4H close series + a 30d daily funding history):

  BTC  hold-bull / funding_short(1.0)-bear / funding_contrarian(0.5)-peace
  ETH  trend-only (hold-bull / flat elsewhere)
  SOL  hold-bull / flat-bear / flat-peace  [DEGRADED — see below]

DEGRADATION, recorded not hidden: SOL's measured book carries a
ridge_defensive bear leg (its whole edge over trend-only), but the
rt_ridge_variant_probe measured the runtime-safe feature subset at CV
-1.93% vs full-feature +5.50% — the edge lives in the denoised/external/
fv2/regime-posterior features. Shadowing a reduced variant would test a
strategy nobody measured. The bear leg therefore ships only with FULL
feature parity via the runtime DRL feature path (P214/P221/P1a parity
tests) — the explicitly scoped next build. Until then SOL's ledger rows
carry book_version="v1_degraded_no_bear_leg" so its forward IC is never
mistaken for the full book's.

Regime labels are the lab's causal a-priori definitions (SMA200 x 540-bar
momentum agreement). Funding z is CAUSAL: previous-day daily close rate,
z-scored over a trailing 30-day window (the P247-F1 convention — never a
same-day read).

Each record separates price-claim from carry so the forward review can
attribute alpha vs beta vs carry (Plan V3 rule): the claim direction is
the TARGET POSITION (what the book would hold), confidence = |target|
(P236 followup: the scorer multiplies direction x confidence — a field
the scorer multiplies by is part of the claim).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

SMA_W, MOM_W = 200, 540
FUND_Z_WINDOW = 30
MIN_BARS = MOM_W + 20

BOOKS_VERSION = {
    "BTC": "v1_full",
    "ETH": "v1_full",           # trend-only IS the full measured ETH book
    # [P299] SOL is TREND-ONLY, not degraded. Its behaviour has always been
    # ETH's certified book verbatim — hold in bull (the shared bull block
    # gives SOL 1.0), flat otherwise — and the only thing it lacks is a BEAR
    # leg, which P250 deleted as a leak artifact and which P262 never
    # certified for SOL in the first place. Labelling a book "degraded" for
    # missing a leg that was never certified made `available` False on every
    # row, which (a) made the whole candidate unscoreable and (b) fed the live
    # seat a 0.0 that reads as an opinion. Trend-only IS the P262-certified
    # mechanism; this is the honest name for what it already does.
    "SOL": "v1_trend_only",
    # [P271] breadth assets below: trend-only, see BREADTH_ASSETS
}

# [P271] The five NEVER-FITTED assets the P262 unread-era probe certified:
# trend/hold beat flat on 5/5 (BNB +406%, DOGE +672%, XRP +276%, ADA +391%,
# LTC +68%, 2020-2026 at 10bps RT) — evidence from data no selection ever
# touched, which the home window structurally cannot provide. All five are
# listed as CDE 20DEC30 nano perps (probed 2026-08-15) and all five serve
# Kraken public 4H OHLC (probed 2026-08-16), so their books are computable
# by this harness with zero new data sources. Book form: TREND-ONLY (bull ->
# long, else flat) — exactly the mechanism the probe certified; no funding
# legs (P262 marks funding legs as the UNCERTIFIED slice even on BTC).
# These are OBSERVATION-ONLY forward ledgers: trading any of them requires
# an operator decision + routing + sleeve asset extension (P141), judged on
# this ledger's P166 read. Volumes are thin (XRP/ADA ~3.8k ct/day, BNB ~200)
# — recorded, and a reason the forward read matters more than the probe.
BREADTH_ASSETS = ("XRP", "ADA", "LTC", "DOGE", "BNB")
for _ba in BREADTH_ASSETS:
    BOOKS_VERSION[_ba] = "v1_breadth_trend_only"


def regime_label(closes) -> str:
    """Causal 3-state label from a 4H close series (last bar = now)."""
    n = len(closes)
    if n < MIN_BARS:
        return "warmup"
    sma = sum(closes[-SMA_W:]) / SMA_W
    mom = closes[-1] / closes[-MOM_W] - 1.0
    above, up = closes[-1] > sma, mom > 0
    if above and up:
        return "bull"
    if (not above) and (not up):
        return "bear"
    return "peace"


def causal_funding_z(daily_rates) -> Optional[float]:
    """z of YESTERDAY'S daily funding close vs its trailing 30d window.
    daily_rates: chronological list of daily close rates where the LAST
    entry is yesterday's completed day (the caller must never append an
    in-progress day — that is the P247-F1 leak)."""
    if daily_rates is None or len(daily_rates) < FUND_Z_WINDOW:
        return None
    w = list(daily_rates[-FUND_Z_WINDOW:])
    mu = sum(w) / len(w)
    var = sum((x - mu) ** 2 for x in w) / len(w)
    sd = var ** 0.5
    if sd <= 0:
        return 0.0
    return (w[-1] - mu) / sd


# [P256] Partial-adjustment parameters (k_exit, k_flip, min_hold) — the
# mechanism-lab design-era winners (training/mechanism_lab.py --stage adjust,
# report mechanism_lab_p256.json): the ONLY mechanism of four that earned on
# all three books (BTC +1.07->+1.23, ETH +0.09->+0.20, SOL +1.21->+1.72
# design-era net). In-design selection = a HYPOTHESIS (P244); the adjusted
# ledger below is its forward exam alongside the raw book's.
ADJ_PARAMS = {"BTC": (1, 2, 0), "ETH": (3, 1, 0), "SOL": (1, 1, 6)}

# [P270] High-vol entry-skip overlay — ETH ONLY, the entry_filter_lab's sole
# THREE-era winner (design +0.088->+0.136, pre-design +1.473->+1.536, and
# the single ledgered validation read +0.500->+0.566; SOL's validation
# increment was exactly 0.0 — no ledger, and BTC's base stood in-design).
# The threshold is the CONSTANT causal expanding-q0.80 value exported at
# read time (training/reports/volskip_validation_read_p270.json), refreshed
# per re-export — the lab's expanding quantile moves glacially over 6y, and
# a constant snapshot is the same deliberate lab/live divergence the P259
# banded export recorded for its sigma. Entries are gated; exits NEVER are
# (the P195/P236 asymmetry); a flip whose entry leg is vol-blocked degrades
# to a flatten.
VOLSKIP_THR = {"ETH": 0.018769}   # rolling-20 4H log-ret std, q0.80
VOLSKIP_MIN_CLOSES = 21


def adjust_step(state: dict, want: float, k_exit: int, k_flip: int,
                min_hold: int) -> float:
    """One step of the asymmetric persistence mechanism — MUST stay in exact
    parity with training/mechanism_lab.apply_adjust (a lab/runtime skew here
    would make the forward ledger measure a different mechanism than the one
    the lab selected; pinned by a parity test). Semantics: exits need k_exit
    consecutive bars, flips k_flip; entries from flat are instant. Restart
    resets the streak — a restart only DELAYS a change (conservative, the
    P198 trade-off).

    [P260] min_hold's TRUE semantics, per the fresh-mind review: `held`
    accrues only on bars where the demand AGREES with the current position,
    so under CONTINUOUS exit demand a position never reaches min_hold and a
    plain exit-to-flat is blocked INDEFINITELY (escape only via a flip
    request or renewed agreement). This is NOT "hold at least N bars then
    exit freely" — it is closer to "plain exits require the demand to have
    relented at least once". The lab MEASURED and selected exactly this
    behavior (SOL's winner is min_hold=6), and the forward ledger honestly
    tests it — so the mechanism is deliberately NOT changed here; the
    docstring is corrected instead, and any ENFORCE decision on a min_hold
    config must first re-lab whether the intended semantics ("N bars then
    free") would have won too (pinned: TestAdjustSemantics)."""
    cur = state.setdefault("cur", 0.0)
    if want == cur:
        state["streak"] = 0
        state["want_prev"] = want
        state["held"] = state.get("held", 0) + 1
        return cur
    state["streak"] = (state.get("streak", 0) + 1
                       if want == state.get("want_prev") else 1)
    state["want_prev"] = want
    need = k_flip if (want != 0 and cur != 0) else (
        k_exit if want == 0 else 1)
    if cur != 0 and state.get("held", 0) < min_hold and want == 0:
        return cur
    if state["streak"] >= need:
        state["cur"] = want
        state["streak"] = 0
        state["held"] = 0
    return state["cur"]


# [P259] The banded-forecast overlay (operator spec: models designed around
# the three measured failure mechanisms). banded_step is THE single source —
# training/banded_forecast_lab.py imports it, so lab and live cannot drift
# (the P172 one-resolver rule, preferred over parity tests).
BANDED_MODEL_PATH = {a: Path("configs") / "regimebook" / f"{a}_banded.json"
                     for a in ("BTC", "ETH", "SOL")}


def banded_step(state: dict, s: float, t_enter: float, t_exit: float,
                regime_gated: bool, lab_regime: int) -> float:
    """One step of the asymmetric band on a normalized forecast s.
    lab_regime: 0=peace 1=bull 2=bear (regime_model_lab.REGIME_ID)."""
    cur = state.setdefault("cur", 0.0)
    if s != s:  # NaN
        return cur
    if cur == 0.0:
        if s > t_enter:
            cur = 1.0
        elif s < -t_enter:
            cur = -1.0
    else:
        if cur > 0 and s < t_exit:
            # [P260] `1.0 if s > t_enter` is vestigial (unreachable: s <
            # t_exit <= t_enter here) — left as-is rather than restructured,
            # because this function's byte-behavior is what the recorded lab
            # reports measured (single-source rule); noted per the
            # fresh-mind review.
            cur = 1.0 if s > t_enter else 0.0
            if s < -t_enter:
                cur = -1.0
        elif cur < 0 and s > -t_exit:
            cur = -1.0 if s < -t_enter else 0.0
            if s > t_enter:
                cur = 1.0
    if regime_gated and cur != 0.0:
        if cur > 0 and lab_regime == 2:
            cur = 0.0
        elif cur < 0 and lab_regime == 1:
            cur = 0.0
    state["cur"] = cur
    return cur


def banded_features_from_closes(closes, funding_z: Optional[float]):
    """The P259 live feature vector, computed from the harness's OWN inputs
    (self-fetched closes + persisted funding z) — the live-parity-by-
    construction constraint. Order MUST match the export's feature list
    (training/banded_forecast_lab.close_features names)."""
    import math as _m
    n = len(closes)
    if n < 545:
        return None
    c = list(map(float, closes))

    def pct(k):
        return c[-1] / c[-1 - k] - 1.0

    lr = [_m.log(c[i] / c[i - 1]) for i in range(1, n)]

    def std(xs):
        m = sum(xs) / len(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    vol20 = std(lr[-20:])
    vol120 = std(lr[-120:])
    sma200 = sum(c[-200:]) / 200.0
    return [pct(6), pct(24), pct(72), pct(168), pct(540),
            c[-1] / sma200 - 1.0, vol20, vol120,
            vol20 / (vol120 + 1e-12),
            0.0 if funding_z is None else float(funding_z)]


def book_target(asset: str, regime: str, funding_z: Optional[float]) -> tuple:
    """(target_position, leg_name). The p247_leakfix winners, verbatim.
    A funding cell with NO causal funding history goes FLAT with a named
    reason — absence must never read as a neutral zero signal (P2/P199)."""
    if regime == "warmup":
        return 0.0, "warmup"
    if regime == "bull":
        if asset in ("BTC", "SOL", "ETH"):
            # ETH's measured book is trend-only: hold in bull like the rest?
            # NO — ETH bull cell measured FLAT (every candidate negative);
            # trend-only's bull leg is hold. ETH book = trend-only per P247,
            # whose bull leg IS hold. BTC/SOL bull = hold.
            return (1.0, "hold") if asset != "ETH" else (1.0, "trend_hold")
    if asset == "BTC":
        if funding_z is None:
            return 0.0, "flat_no_funding_history"
        if regime == "bear":
            return (-1.0, "funding_short") if funding_z > 1.0 else (0.0, "flat")
        # peace: contrarian at 0.5
        if funding_z > 0.5:
            return -1.0, "funding_contrarian_short"
        if funding_z < -0.5:
            return 1.0, "funding_contrarian_long"
        return 0.0, "flat"
    if asset == "ETH":
        return 0.0, "trend_flat"        # trend-only: flat outside bull
    if asset == "SOL":
        # [P299] Same leg ETH takes: trend-only is flat outside bull. This is
        # a POSITION ("be flat"), not a missing capability.
        return 0.0, "trend_flat"
    if asset in BREADTH_ASSETS:
        # [P271] trend-only, the P262-certified mechanism verbatim: the
        # bull leg is handled by the shared bull block above ONLY for
        # BTC/ETH/SOL, so breadth needs its own explicit legs — falling
        # through to the generic flat would silently record a dead book.
        if regime == "bull":
            return 1.0, "trend_hold"
        return 0.0, "trend_flat"
    return 0.0, "flat"


KRAKEN_PAIRS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD",
                # [P271] breadth — all five probed OK at interval=240
                "XRP": "XRPUSD", "ADA": "ADAUSD", "LTC": "LTCUSD",
                "DOGE": "XDGUSD", "BNB": "BNBUSD"}
BINANCE_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
SOL_MODEL_PATH = Path("configs") / "regimebook" / "SOL_bear_ridge.json"
FEATURE_STASH_MAX_AGE_S = 2 * 3600   # one 4H tick's worth of tolerance


def _http_json(url: str, timeout: float = 10.0):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "hmats-regimebook"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class RegimeBookShadow:
    """Per-tick recorder. SELF-CONTAINED by design: fetches its own closes
    (Kraken public OHLC) and funding history (Binance fapi, best-effort) so
    the wiring into main.py is minimal and a fault here can never touch the
    order path. Fail-soft everywhere, but never silent (P160)."""

    def __init__(self, data_dir: str = "data", repo_root: Optional[Path] = None):
        self._dir = Path(data_dir) / "strategy_shadow"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fund_hist_path = Path(data_dir) / "regimebook_funding_daily.json"
        self._fund_hist = self._load_funding_history()
        # [P253] PER-ASSET, was a single scalar: the first asset to succeed
        # set it for the day, so ETH/SOL were skipped until tomorrow and
        # their funding history could never accrue. Harmless while only BTC
        # has funding legs; latent the moment another asset gains one.
        self._last_funding_refresh_day: Dict[str, str] = {}
        self._feature_stash: dict = {}    # asset -> (ts, {name: value})
        self._last_records: Dict[str, dict] = {}   # [P256] asset -> last rec
        # [P303] Restore from the LEDGER so the seat survives a restart.
        # `tick()` runs at LOOP level AFTER decide (P248), so the first tick of
        # any process had nothing to seat and logged "no fresh book target
        # exists" — measured live on 2026-08-18 for all three assets right
        # after a deploy, while the ledger on disk already held
        # BTC=bear/funding_short/-1.0. With `regimebook_mode: enforce` that
        # costs the CERTIFIED book (P297: 6 years, 3/3 eras) a full 4H tick
        # per deploy, and this engine deploys often — the same restart class
        # as the P301/P302 warmups, on the one signal now holding the seat.
        #
        # The rows are already written and already carry `ts`, so nothing new
        # is persisted; `last_direction()` applies its own 6h staleness bound
        # to whatever is restored, so a genuinely old ledger still yields NO
        # SEAT rather than a stale one (P2: absent is not flat).
        self._restore_last_records()
        self._adj_state: Dict[str, dict] = {}      # [P256] adjust mechanism
        # [P259] banded-forecast overlay: model per asset (None = no export,
        # leg silent) + band state machine per asset
        self._banded_models: Dict[str, Optional[dict]] = {
            a: self._load_banded(a, repo_root) for a in ("BTC", "ETH", "SOL")}
        self._banded_state: Dict[str, dict] = {}
        self._volskip_state: Dict[str, dict] = {}  # [P270] asset -> {"cur"}
        self._last_closes: Dict[str, list] = {}    # [P277] asset -> closes (enh consumers)
        self._sol_model = self._load_sol_model(repo_root)
        self._warned: set = set()
        self._funding_stale: Dict[str, bool] = {}  # [P265] per-asset stale-z latch (re-armable)

    # ---------------- [P256] the SEAT accessor -------------------------
    def advisory_snapshot(self, assets) -> dict:
        """[P272] Passive-holdings advisory: the certified book label for
        assets the OPERATOR holds outside the system (recorded decision
        2026-08-16: the 12,300-XRP Kraken bag and ~$3k BTC are deliberate
        holds the system does not trade). This surfaces what the certified
        trend/hold mechanism says about them in the 4H heartbeat — purely
        informational, trades nothing, and a label CHANGE is flagged once
        (the first snapshot after the flip) so the operator sees regime
        turns without alert spam (P202).

        Returns {asset: {"regime", "direction", "changed"}} from the LAST
        recorded tick (the harness runs after the heartbeat, so labels are
        one tick stale — immaterial at regime horizon; the caller hedges
        the text). Assets with no record yet are omitted — absence must
        never render as a confident label (P2)."""
        if not hasattr(self, "_advisory_prev"):
            self._advisory_prev: dict = {}
        out = {}
        for a in assets or ():
            rec = self._last_records.get(a)
            if not rec:
                continue
            label = f"{rec.get('regime')}/{rec.get('direction', 0.0):+.0f}"
            changed = (a in self._advisory_prev
                       and self._advisory_prev[a] != label)
            self._advisory_prev[a] = label
            out[a] = {"regime": rec.get("regime"),
                      "direction": rec.get("direction", 0.0),
                      "changed": changed}
        return out

    def _restore_last_records(self) -> None:
        """[P303] Rehydrate `_last_records` from the newest ledger row per
        asset. Never raises: a failed restore is exactly today's behaviour
        (no seat on the first tick)."""
        try:
            import glob
            for path in glob.glob(str(self._dir / "regimebook_*.jsonl")):
                name = Path(path).name
                # only the RAW book feeds the seat — adj/banded/volskip are
                # separate exams and must never be mistaken for it
                stem = name[len("regimebook_"):-len(".jsonl")]
                if "_" in stem:
                    continue
                try:
                    with open(path, encoding="utf-8") as fh:
                        lines = [ln for ln in fh if ln.strip()]
                    if not lines:
                        continue
                    rec = json.loads(lines[-1])
                except Exception:  # noqa: silent-swallow — skip a bad ledger
                    continue
                if isinstance(rec, dict) and rec.get("ts"):
                    self._last_records[stem] = rec
            if self._last_records:
                logger.info(
                    "[REGIMEBOOK] restored last targets for %s from the ledger "
                    "— the seat does not lose a tick to this restart",
                    ", ".join(sorted(self._last_records)))
        except Exception as e:  # noqa: silent-swallow — logged
            logger.warning("[REGIMEBOOK] ledger restore failed (%s) — the "
                           "first tick will have no seat input", type(e).__name__)

    def last_direction(self, asset: str, max_age_s: float = 6 * 3600):
        """The book's most recent recorded target for `asset`, or None.

        Returns ``(direction, leg, age_s)`` when a record exists and is
        younger than ``max_age_s`` (default 6h = one 4H tick + slack, the
        P156 staleness-bound rule). None means "no fresh opinion" — the
        caller must treat that as NO SEAT INPUT, never as flat (the P2
        missing-vs-neutral rule): the book saying "flat" is a position;
        the book being absent is not.

        NOTE the one-tick lag by construction: tick() runs at LOOP level
        AFTER decide (P248 wiring), so a seat consumer inside decide reads
        the PREVIOUS loop's target. The book is a regime-level signal
        (bull/bear/peace + daily funding z) — a 4h lag is immaterial next
        to its decision horizon, but it is a fact, not an accident.
        """
        rec = self._last_records.get(asset)
        if not rec:
            return None
        age = time.time() - float(rec.get("ts", 0.0) or 0.0)
        if age > max_age_s:
            return None
        # [P299] AVAILABILITY GATES THE SEAT. The consumer at main.py's
        # regimebook seat assigns quant_direction UNCONDITIONALLY, including
        # 0.0 — so serving a row whose own `available` flag is False would let
        # a book that CANNOT take a position flatten the incumbent, which is
        # precisely the UNAVAILABLE-is-not-FLAT error (P2) the ledger's
        # `available` field was added to make visible. A book can degrade
        # dynamically too (a feature-coverage gap, see _record_for), so this
        # is not only about SOL. Absent capability -> no seat input at all.
        if rec.get("available") is False:
            return None
        return float(rec.get("direction", 0.0) or 0.0), str(rec.get("leg", "")), age

    # ---------------- funding history (persisted, P154 rule) ----------
    def _load_funding_history(self):
        try:
            if self._fund_hist_path.exists():
                return json.loads(self._fund_hist_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning("[REGIMEBOOK] funding history unreadable (%s) — "
                           "cold start; funding cells flat until %d days accrue",
                           type(e).__name__, FUND_Z_WINDOW)
        return {}

    def record_daily_funding(self, asset: str, day_iso: str, rate: float):
        """Append a COMPLETED day's closing funding rate. The caller passes
        yesterday's day, never today's (P247-F1)."""
        h = self._fund_hist.setdefault(asset, {})
        if day_iso in h:
            return
        h[day_iso] = float(rate)
        # keep a bounded window
        for k in sorted(h)[:-3 * FUND_Z_WINDOW]:
            del h[k]
        try:
            tmp = self._fund_hist_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._fund_hist), encoding="utf-8")
            tmp.replace(self._fund_hist_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("[REGIMEBOOK] funding history persist failed: %s",
                           type(e).__name__)

    def _funding_series(self, asset: str):
        h = self._fund_hist.get(asset)
        if not h:
            return None
        return [h[k] for k in sorted(h)]

    def carry_rate_bar(self, asset: str) -> Optional[float]:
        """[P287] Per-4H-bar funding carry from the newest COMPLETED day.

        The history stores the day's LAST 8h funding EVENT rate
        (refresh_funding_daily); an 8h event spans two 4H bars, so the
        per-bar carry is rate/2 — the P245 convention (shorts collect when
        positive). This closes the docstring's promise that every ledger
        row separates price-claim from carry: pre-P287 `carry_rate_bar`
        was None on every row ever written, so the BTC funding legs — the
        roster's only uncertified component (P262) — could not have their
        carry attributed from the ledger at the ~09-09 read (the P245
        lesson: perp realized PnL = price + carry). None when the history
        is absent or stale (same bound as the z, P265) — absent funding
        stays honest, never a fabricated 0.0 (P2)."""
        h = self._fund_hist.get(asset)
        if not h:
            return None
        age = self.funding_age_days(asset)
        if age is None or age > self.FUND_MAX_AGE_DAYS:
            return None
        try:
            return float(h[max(h)]) / 2.0
        except (TypeError, ValueError):  # noqa: silent-swallow — malformed stored rate = no carry claim; the z path warns on its own
            return None

    # [P265] Staleness bound on the funding input. Once >=30 days of history
    # are persisted, a sustained fapi outage used to leave `_fund_hist`
    # frozen: causal_funding_z still returned a NUMBER (the z of the last
    # successfully fetched day, arbitrarily old) and the BTC funding legs
    # kept asserting +/-1 directions from it into the forward ledger,
    # unmarked — the September P166 read would have judged a frozen input,
    # not the strategy (and per P262, the funding legs are the roster's ONLY
    # uncertified component). Normal cadence: the last completed day is
    # yesterday (age 1); 3 days covers the pre-refresh midnight window and
    # one missed day.
    FUND_MAX_AGE_DAYS = 3

    def funding_age_days(self, asset: str) -> Optional[int]:
        """Days since the newest COMPLETED day in the funding history."""
        h = self._fund_hist.get(asset)
        if not h:
            return None
        try:
            last = datetime.strptime(max(h), "%Y-%m-%d").date()
        except (ValueError, TypeError):  # noqa: silent-swallow — malformed day key = unknown age; callers treat None as no-bound
            return None
        return (datetime.now(timezone.utc).date() - last).days

    # ---------------- self-contained data fetches ----------------------
    def fetch_closes_4h(self, asset: str):
        """~720 4H closes from Kraken's PUBLIC OHLC endpoint (the proven
        trend_regime_review pattern). None on any failure."""
        pair = KRAKEN_PAIRS.get(asset)
        if not pair:
            return None
        try:
            data = _http_json(
                f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=240")
            if data.get("error"):
                raise RuntimeError(str(data["error"])[:80])
            result = data.get("result", {})
            key = next((k for k in result if k != "last"), None)
            rows = result.get(key, [])
            closes = [float(r[4]) for r in rows]
            return closes if len(closes) >= MIN_BARS else None
        except Exception as e:  # noqa: silent-swallow — logs via _warn_once (the linter cannot see method-call logging); tick skipped is the stated consequence
            self._warn_once(f"closes:{asset}",
                            f"[REGIMEBOOK] {asset}: OHLC fetch failed "
                            f"({type(e).__name__}) — tick skipped")
            return None

    def refresh_funding_daily(self, asset: str):
        """Once per UTC day: pull the last ~1000 8h funding events from
        Binance fapi, aggregate to daily closes, record every COMPLETED day
        (today is excluded — appending an in-progress day is the P247-F1
        leak). Best-effort: geo-blocks or outages leave the funding cells
        flat-with-reason, never a fabricated z."""
        today = datetime.now(timezone.utc).date().isoformat()
        if self._last_funding_refresh_day.get(asset) == today:
            return
        sym = BINANCE_SYMBOLS.get(asset)
        if not sym:
            return
        try:
            rows = _http_json("https://fapi.binance.com/fapi/v1/fundingRate"
                              f"?symbol={sym}&limit=1000", timeout=15.0)
            by_day = {}
            for r in rows:
                d = datetime.fromtimestamp(
                    int(r["fundingTime"]) / 1000, tz=timezone.utc).date().isoformat()
                by_day[d] = float(r["fundingRate"])   # last event of the day wins
            for d in sorted(by_day):
                if d < today:                          # completed days only
                    self.record_daily_funding(asset, d, by_day[d])
            self._last_funding_refresh_day[asset] = today
        except Exception as e:  # noqa: BLE001
            self._warn_once(f"funding:{asset}",
                            f"[REGIMEBOOK] {asset}: funding backfill failed "
                            f"({type(e).__name__}) — funding cells stay flat "
                            f"until history accrues")

    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg)

    # ---------------- SOL full-parity bear leg -------------------------
    def _load_sol_model(self, repo_root):
        path = (Path(repo_root) / SOL_MODEL_PATH) if repo_root else SOL_MODEL_PATH
        try:
            if path.exists():
                m = json.loads(path.read_text(encoding="utf-8"))
                if {"features", "mean", "scale", "coef", "intercept",
                    "train_sigma"} <= set(m):
                    logger.info("[REGIMEBOOK] SOL bear ridge loaded "
                                f"({len(m['features'])} features, "
                                f"fitted {m.get('fitted_at', '?')})")
                    return m
                # [P253c] The schema-mismatch case returned None with NO log
                # — a present-but-malformed export was indistinguishable from
                # the deliberate no-model degradation. Say which keys are
                # missing; the degradation itself is still correct.
                logger.warning(
                    "[REGIMEBOOK] SOL model at %s FAILED the schema check "
                    "(missing: %s) — degrading to v1_degraded_no_bear_leg",
                    path,
                    sorted({"features", "mean", "scale", "coef", "intercept",
                            "train_sigma"} - set(m)))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[REGIMEBOOK] SOL model unreadable: {type(e).__name__}")
        return None

    def _load_banded(self, asset: str, repo_root):
        """[P259] Load the banded-overlay export. Absent = leg silent (BTC/
        ETH earned it in the lab; SOL's book stood, so no SOL export exists
        by decision). Malformed = logged, never guessed."""
        path = ((Path(repo_root) / BANDED_MODEL_PATH[asset]) if repo_root
                else BANDED_MODEL_PATH[asset])
        try:
            if path.exists():
                m = json.loads(path.read_text(encoding="utf-8"))
                need = {"features", "mean", "scale", "coef", "intercept",
                        "forecast_sigma", "band"}
                if need <= set(m):
                    logger.info("[REGIMEBOOK] %s banded overlay loaded "
                                "(%d features, fitted %s)", asset,
                                len(m["features"]), m.get("fitted_at", "?"))
                    return m
                logger.warning(
                    "[REGIMEBOOK] %s banded export FAILED schema (missing "
                    "%s) — leg silent", asset, sorted(need - set(m)))
        except Exception as e:  # noqa: BLE001
            logger.warning("[REGIMEBOOK] %s banded export unreadable: %s",
                           asset, type(e).__name__)
        return None

    def _banded_overlay_target(self, asset: str, closes, book_target_val,
                               regime: str, fz):
        """[P259] The overlay: the BOOK's position wherever it has an
        opinion; the banded forecaster only in the cells the book leaves
        flat. Returns (overlay_target, banded_raw, forecast_s) or None when
        the leg cannot run (no model / short history)."""
        m = self._banded_models.get(asset)
        if m is None:
            return None
        x = banded_features_from_closes(closes, fz)
        if x is None:
            return None
        z = [(xi - mu) / sd for xi, mu, sd in
             zip(x, m["mean"], m["scale"])]
        f = sum(c * zi for c, zi in zip(m["coef"], z)) + m["intercept"]
        s = f / (float(m["forecast_sigma"]) or 1e-9)
        band = m["band"]
        lab_regime = {"bull": 1, "bear": 2}.get(regime, 0)
        banded = banded_step(self._banded_state.setdefault(asset, {}),
                             float(s), float(band["t_enter"]),
                             float(band["t_exit"]),
                             bool(band.get("regime_gated")), lab_regime)
        overlay = (float(book_target_val) if float(book_target_val) != 0.0
                   else float(banded))
        return overlay, float(banded), float(s)

    def _volskip_target(self, asset: str, closes, raw_target: float):
        """[P270] One live step of the entry_filter_lab volskip mechanism.

        Returns (filtered_target, vol20, allow) or None when the asset has
        no exported threshold. Single-step form of apply_entry_filter:
          - exits (raw -> 0) are ALWAYS honored;
          - entries from flat and the entry LEG of a flip require
            vol20 <= VOLSKIP_THR (a blocked flip degrades to flatten);
          - unavailable vol reads as allow=False — absence must not open
            positions the filter would have vetoed, and can never force an
            exit (P2: absence is not a signal).
        State starts flat (cur=0): a one-time initialization transient, the
        same accepted shape as the adjust ledger's in-memory state.
        """
        thr = VOLSKIP_THR.get(asset)
        if thr is None:
            return None
        vol20 = None
        allow = False
        try:
            if closes is not None and len(closes) >= VOLSKIP_MIN_CLOSES:
                import math as _m
                w = closes[-VOLSKIP_MIN_CLOSES:]
                rets = [_m.log(w[i + 1] / w[i]) for i in range(len(w) - 1)
                        if w[i] > 0 and w[i + 1] > 0]
                if len(rets) >= VOLSKIP_MIN_CLOSES - 1:
                    mu = sum(rets) / len(rets)
                    # ddof=1 to match pandas.Series.rolling(20).std()
                    var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
                    vol20 = var ** 0.5
                    allow = vol20 <= thr
        except Exception:  # noqa: silent-swallow — vol failure means allow=False (entries blocked, exits untouched); ledger row records vol=None
            vol20, allow = None, False
        st = self._volskip_state.setdefault(asset, {"cur": 0.0})
        cur = float(st.get("cur", 0.0))
        want = float(raw_target)
        if want == cur:
            pass
        elif want == 0.0:
            cur = 0.0                      # exit — always honored
        elif cur == 0.0:
            cur = want if allow else 0.0   # entry from flat
        else:
            cur = want if allow else 0.0   # flip: entry leg gated
        st["cur"] = cur
        return cur, vol20, allow

    def observe_features(self, asset: str, market_data: dict,
                         agent_signals: Optional[dict] = None):
        """Stash the SOL model's named features from the tick's live dicts.
        Coverage is COUNTED, never assumed (P2 class): the bear leg only
        activates at 100% coverage, and rows report what is missing."""
        try:
            if self._sol_model is None or asset != "SOL":
                return
            vals = {}
            for name in self._sol_model["features"]:
                v = (market_data or {}).get(name)
                if v is None and agent_signals:
                    v = agent_signals.get(name)
                if v is not None:
                    try:
                        vals[name] = float(v)
                    except (TypeError, ValueError):  # noqa: silent-swallow — value coercion; a non-numeric feature is simply absent from the stash (P223 precedent)
                        pass
            self._feature_stash[asset] = (time.time(), vals)
        except Exception as e:  # noqa: BLE001
            self._warn_once("stash", f"[REGIMEBOOK] feature stash failed: "
                                     f"{type(e).__name__}")

    def _sol_bear_target(self):
        """(target, leg, version, coverage_note). Falls back to the declared
        degraded flat unless model + FRESH stash + 100% coverage all hold."""
        m = self._sol_model
        if m is None:
            # [P299] No model is the PERMANENT state (P250 deleted the export
            # and the dir does not exist), so this is not a degradation — it
            # is simply the trend-only book, which says flat outside bull.
            # The model path is kept so a future re-export revives the richer
            # book without a code change.
            return 0.0, "trend_flat", "v1_trend_only", None
        ts_vals = self._feature_stash.get("SOL")
        if not ts_vals or (time.time() - ts_vals[0]) > FEATURE_STASH_MAX_AGE_S:
            return 0.0, "flat_degraded_stale_features", "v1_degraded_no_bear_leg", None
        vals = ts_vals[1]
        missing = [f for f in m["features"] if f not in vals]
        if missing:
            note = f"missing {len(missing)}/{len(m['features'])}: {missing[:5]}"
            return 0.0, "flat_degraded_parity_gap", "v1_degraded_no_bear_leg", note
        x = [(vals[f] - mu) / (sc or 1.0)
             for f, mu, sc in zip(m["features"], m["mean"], m["scale"])]
        pred = sum(xi * c for xi, c in zip(x, m["coef"])) + m["intercept"]
        z = pred / (m["train_sigma"] or 1e-9)
        if abs(z) < 0.25:                       # the lab's deadband
            return 0.0, "ridge_defensive_flat", "v2_full_bear", None
        tgt = max(-1.0, min(0.0, z))            # defensive: short-only clip
        return tgt, "ridge_defensive", "v2_full_bear", None

    # ---------------- per-tick record ---------------------------------
    def record_tick(self, asset: str, closes, price: float,
                    carry_rate_bar: Optional[float] = None):
        """Compute the book's target and append the ledger row. Returns the
        record (or None on failure) for tests/diagnostics."""
        try:
            regime = regime_label(closes)
            fz = causal_funding_z(self._funding_series(asset))
            _fage = self.funding_age_days(asset)
            if (fz is not None and _fage is not None
                    and _fage > self.FUND_MAX_AGE_DAYS):
                # [P265] Frozen funding input -> flat-with-reason, exactly
                # what the refresh docstring always claimed. Warned on every
                # TRANSITION to stale (not once per process — week 2 of an
                # outage must not be silent) and recovery is announced.
                if not self._funding_stale.get(asset):
                    self._funding_stale[asset] = True
                    logger.warning(
                        "[REGIMEBOOK] %s: funding history is %dd old (> %dd)"
                        " — funding cells go FLAT (a frozen z must not keep"
                        " trading into the forward ledger, P265)",
                        asset, _fage, self.FUND_MAX_AGE_DAYS)
                fz = None
            elif self._funding_stale.get(asset):
                self._funding_stale[asset] = False
                logger.info("[REGIMEBOOK] %s: funding history fresh again "
                            "(age=%sd) — funding cells re-enabled",
                            asset, _fage)
            target, leg = book_target(asset, regime, fz)
            version = BOOKS_VERSION.get(asset, "unknown")
            coverage_note = None
            if asset == "SOL" and regime == "bear":
                target, leg, version, coverage_note = self._sol_bear_target()
            rec = {
                "ts": time.time(),
                "iso": datetime.now(timezone.utc).isoformat(),
                "strategy": "regimebook",
                "asset": asset,
                "book_version": version,
                "regime": regime,
                "leg": leg,
                "coverage_note": coverage_note,
                # [P295] AVAILABILITY — is this book STRUCTURALLY ABLE to
                # take a position right now, as distinct from choosing not to?
                #
                # SOL's bear leg needs the exported ridge model, and P250
                # DELETED configs/regimebook/SOL_bear_ridge.json (it was the
                # leak artifact — its "+64.2%" was the leaked feature). The
                # directory does not exist, so SOL emits direction 0.0 forever
                # while ETH ALSO emits 0.0 because its trend-only book is
                # correctly flat outside a bull regime. Those two zeros mean
                # completely different things, and a consumer reading only
                # `direction` cannot tell them apart — a broken candidate
                # would score as a flat OPINION and could win a seat by
                # default (P2, and the reason the P294 seat controller has an
                # UNAVAILABLE state at all).
                #
                # False = cannot express a position at all. It is NOT a claim
                # about the market.
                "available": not str(version).startswith("v1_degraded"),
                "funding_z": None if fz is None else round(fz, 4),
                # [P265] stale-z rows are now filterable post-hoc
                "funding_age_days": _fage,
                "direction": float(target),
                # scorer multiplies direction x confidence (P236): |target|,
                # so flat rows contribute zero, never a saturated claim (P224)
                "confidence": abs(float(target)),
                "price": float(price),
                # [P287] explicit caller value wins; otherwise derived from
                # the persisted funding history (None when absent/stale)
                "carry_rate_bar": (carry_rate_bar
                                   if carry_rate_bar is not None
                                   else self.carry_rate_bar(asset)),
            }
            path = self._dir / f"regimebook_{asset}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            # [P256] Stash for the SEAT (last_direction below). Written only
            # on a successful record so a failed tick can never serve a
            # half-computed target.
            self._last_records[asset] = rec
            # [P256] The ADJUSTED book's forward ledger — same glob prefix
            # (regimebook_*), distinct strategy name, so the scorer groups it
            # as its own candidate with zero scorer changes. Fail-soft: the
            # raw record above is already written and returned.
            # [P271] adj leg only for assets the mechanism lab actually
            # measured (BTC/ETH/SOL) — the old (1, 1, 0) .get() default is a
            # near-passthrough that would duplicate every breadth row under
            # a second strategy name and dilute the scorer with unmeasured
            # candidates.
            if asset in ADJ_PARAMS:
                try:
                    ke, kf, mh = ADJ_PARAMS[asset]
                    adj = adjust_step(self._adj_state.setdefault(asset, {}),
                                      float(target), ke, kf, mh)
                    arec = dict(rec, strategy="regimebook_adj",
                                direction=float(adj),
                                confidence=abs(float(adj)),
                                adj_params=[ke, kf, mh])
                    apath = self._dir / f"regimebook_adj_{asset}.jsonl"
                    with open(apath, "a", encoding="utf-8") as f:
                        f.write(json.dumps(arec) + "\n")
                except Exception as _adj_e:  # noqa: silent-swallow — the adjusted ledger is an overlay; its failure must not lose the raw record (logged)
                    logger.warning("[REGIMEBOOK] %s adjusted-ledger write "
                                   "failed: %s", asset, type(_adj_e).__name__)
            # [P259] The banded-forecast OVERLAY ledger (BTC/ETH exports only
            # — the lab earners). Same glob prefix, distinct strategy name.
            try:
                bo = self._banded_overlay_target(asset, closes, target,
                                                 regime, fz)
                if bo is not None:
                    _ov, _bd, _s = bo
                    brec = dict(rec, strategy="regimebook_banded",
                                direction=float(_ov),
                                confidence=abs(float(_ov)),
                                banded_raw=_bd, forecast_s=round(_s, 4))
                    bpath = self._dir / f"regimebook_banded_{asset}.jsonl"
                    with open(bpath, "a", encoding="utf-8") as f:
                        f.write(json.dumps(brec) + "\n")
            except Exception as _bd_e:  # noqa: silent-swallow — overlay leg; failure must not lose the raw record (logged)
                logger.warning("[REGIMEBOOK] %s banded-ledger write failed: "
                               "%s", asset, type(_bd_e).__name__)
            # [P270] The volskip ENTRY-FILTER ledger (ETH only — the
            # entry_filter_lab's sole three-era winner). Same glob prefix,
            # distinct strategy name, zero scorer changes. Fail-soft: the
            # raw record above is already written and returned.
            try:
                vsk = self._volskip_target(asset, closes, float(target))
                if vsk is not None:
                    _vt, _vol, _allow = vsk
                    vrec = dict(rec, strategy="regimebook_volskip",
                                direction=float(_vt),
                                confidence=abs(float(_vt)),
                                vol20=None if _vol is None else round(_vol, 6),
                                vol_thr=VOLSKIP_THR.get(asset),
                                entry_allowed=bool(_allow))
                    vpath = self._dir / f"regimebook_volskip_{asset}.jsonl"
                    with open(vpath, "a", encoding="utf-8") as f:
                        f.write(json.dumps(vrec) + "\n")
            except Exception as _vs_e:  # noqa: silent-swallow — overlay leg; failure must not lose the raw record (logged)
                logger.warning("[REGIMEBOOK] %s volskip-ledger write failed: "
                               "%s", asset, type(_vs_e).__name__)
            return rec
        except Exception as e:  # noqa: BLE001
            logger.warning("[REGIMEBOOK] %s tick record failed: %s — shadow "
                           "ledger is STALE for this tick", asset,
                           type(e).__name__)
            return None

    # ---------------- one-call orchestrator for main.py ----------------
    def tick(self, assets=("BTC", "ETH", "SOL") + BREADTH_ASSETS):
        """The single loop-level entry point: per asset, refresh funding
        (daily), fetch closes, record. Every failure is per-asset and
        fail-soft; a summary line makes silence impossible (P155)."""
        summary = []
        for asset in assets:
            try:
                self.refresh_funding_daily(asset)
                closes = self.fetch_closes_4h(asset)
                if closes is None:
                    summary.append(f"{asset}=SKIP(no_closes)")
                    continue
                # [P253c] Kraken's OHLC endpoint returns the IN-PROGRESS 4H
                # candle as its last row. The lab's books were designed and
                # measured on COMPLETED bars, so labelling the live regime on
                # a partial bar is a small train/serve skew in exactly the
                # harness whose job is parity. Drop it: the label and price
                # come from the last completed bar, matching the lab.
                closes = closes[:-1]
                # [P277] stash for the enhancement shadows (xsmom + 24h
                # price change) — written only from COMPLETED bars
                self._last_closes[asset] = closes
                rec = self.record_tick(asset, closes, price=closes[-1])
                summary.append(
                    f"{asset}={rec['regime']}/{rec['leg']}/{rec['direction']:+.1f}"
                    if rec else f"{asset}=SKIP(write_failed)")
            except Exception as e:  # noqa: BLE001
                summary.append(f"{asset}=SKIP({type(e).__name__})")
        logger.info("[REGIMEBOOK] " + " | ".join(summary))
        return summary
