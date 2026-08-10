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
    "SOL": "v1_degraded_no_bear_leg",
}


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


def adjust_step(state: dict, want: float, k_exit: int, k_flip: int,
                min_hold: int) -> float:
    """One step of the asymmetric persistence mechanism — MUST stay in exact
    parity with training/mechanism_lab.apply_adjust (a lab/runtime skew here
    would make the forward ledger measure a different mechanism than the one
    the lab selected; pinned by a parity test). Semantics: exits need k_exit
    consecutive bars, flips k_flip, fresh entries hold >= min_hold bars
    against plain exits (flips still honor k_flip); entries from flat are
    instant. Restart resets the streak — a restart only DELAYS a change
    (conservative, the P198 trade-off)."""
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
        return 0.0, "flat_degraded"     # bear ridge leg pending full parity
    return 0.0, "flat"


KRAKEN_PAIRS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD"}
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
        self._adj_state: Dict[str, dict] = {}      # [P256] adjust mechanism
        self._sol_model = self._load_sol_model(repo_root)
        self._warned: set = set()

    # ---------------- [P256] the SEAT accessor -------------------------
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
            return 0.0, "flat_degraded_no_model", "v1_degraded_no_bear_leg", None
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
                "funding_z": None if fz is None else round(fz, 4),
                "direction": float(target),
                # scorer multiplies direction x confidence (P236): |target|,
                # so flat rows contribute zero, never a saturated claim (P224)
                "confidence": abs(float(target)),
                "price": float(price),
                "carry_rate_bar": carry_rate_bar,
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
            try:
                ke, kf, mh = ADJ_PARAMS.get(asset, (1, 1, 0))
                adj = adjust_step(self._adj_state.setdefault(asset, {}),
                                  float(target), ke, kf, mh)
                arec = dict(rec, strategy="regimebook_adj",
                            direction=float(adj), confidence=abs(float(adj)),
                            adj_params=[ke, kf, mh])
                apath = self._dir / f"regimebook_adj_{asset}.jsonl"
                with open(apath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(arec) + "\n")
            except Exception as _adj_e:  # noqa: silent-swallow — the adjusted ledger is an overlay; its failure must not lose the raw record (logged)
                logger.warning("[REGIMEBOOK] %s adjusted-ledger write failed: "
                               "%s", asset, type(_adj_e).__name__)
            return rec
        except Exception as e:  # noqa: BLE001
            logger.warning("[REGIMEBOOK] %s tick record failed: %s — shadow "
                           "ledger is STALE for this tick", asset,
                           type(e).__name__)
            return None

    # ---------------- one-call orchestrator for main.py ----------------
    def tick(self, assets=("BTC", "ETH", "SOL")):
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
                rec = self.record_tick(asset, closes, price=closes[-1])
                summary.append(
                    f"{asset}={rec['regime']}/{rec['leg']}/{rec['direction']:+.1f}"
                    if rec else f"{asset}=SKIP(write_failed)")
            except Exception as e:  # noqa: BLE001
                summary.append(f"{asset}=SKIP({type(e).__name__})")
        logger.info("[REGIMEBOOK] " + " | ".join(summary))
        return summary
