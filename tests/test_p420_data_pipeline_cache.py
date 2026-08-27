"""[P420] The decision tick was a TA/GMM cache HIT — fork-3 tasks 1, 2, 15.

`run_live` sleeps to the 4H candle in 30s chunks and runs the whole pipeline
on every chunk (P353). The watchdog pass at boundary+~30s is the cache MISS
that primes the new bar; the decision fetch at +90s (`for_decision=True`)
then HIT and skipped the entire MISS block — so the P354 wavelet append, the
P414c `_gmm_raw_features` stash and the P339 intrabar keys ran only on
RESTART ticks (server-verified 2026-08-27: `[JUMP-REGIME]` lines only after
each of the day's 7 restarts, never at the 4H decision ticks).

Pins:
  * a `for_decision=True` call on a primed cache appends EXACTLY ONE wavelet
    sample, rebuilds the GMM stash, and advances the smoother exactly once;
  * a `for_decision=False` call on a HIT appends NOTHING (no double-append
    within one decision) and still serves the intrabar + stash keys;
  * the forced miss does NOT re-feed the ATR calculator (it appends per call);
  * `vol_percentile` ranks the last COMPLETED bar, never the in-progress one;
  * the stash is withheld on the distribution-shift fallback.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time

import numpy as np
import pytest

import data_mgmt.market_data_pipeline as mdp
from data_mgmt.market_data_pipeline import (
    MarketDataPipeline, bar_progress_4h, intrabar_volume_keys,
    gmm_volume_rank_input, GMM_VOL_PCT_WINDOW,
)

BAR_MS = 4 * 3600 * 1000
N_BARS = 320


def _bars(now_ms: int, in_progress_elapsed_s: float = 90.0):
    """N_BARS synthetic 4H bars; the LAST bar opened `in_progress_elapsed_s`
    ago (in progress) and carries a tiny volume, like the live frame."""
    rng = np.random.default_rng(418)
    open_last = int(now_ms - in_progress_elapsed_s * 1000)
    out = []
    px = 100.0
    for i in range(N_BARS):
        ts = open_last - (N_BARS - 1 - i) * BAR_MS
        px *= float(np.exp(rng.normal(0, 0.01)))
        vol = float(rng.lognormal(8, 0.3))
        out.append([ts, px, px * 1.01, px * 0.99, px, vol])
    out[-1][5] = 3.0  # the partial bar: ~0.1% of a full bar's volume
    return out


class _ATR:
    def __init__(self):
        self.calls = 0

    def update(self, *a, **k):
        self.calls += 1


class _AdaptiveStop:
    def __init__(self):
        self.atr_calculator = _ATR()


class _GMM:
    """Fake GaussianMixture: label = self.label (settable per call)."""
    def __init__(self, k=3):
        self.k = k
        self.label = 0

    def predict_proba(self, X):
        p = np.full((1, self.k), 0.05)
        p[0, self.label] = 1.0 - 0.05 * (self.k - 1)
        return p


def _pipeline(tmp_path, monkeypatch, gmm: _GMM | None = None,
              adaptive_stop=None):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    names = ["QUIET_ACCUMULATION", "MOMENTUM_RALLY", "WEAK_CONSOLIDATION"]
    kw = {}
    if gmm is not None:
        # scaler with a huge scale => |z| tiny => never the OOD fallback
        kw = dict(gmm_models={"BTC": gmm},
                  gmm_configs={"BTC": {
                      "scaler_mean": [0.0] * 9,
                      "scaler_scale": [1e6] * 9,
                      "regime_names": names,
                      "feature_cols": [f"f{i}" for i in range(9)]}})
    p = MarketDataPipeline(sync_executor=None, assets=["BTC"],
                           adaptive_stop=adaptive_stop, **kw)
    now_ms = int(time.time() * 1000)
    bars = _bars(now_ms)

    async def _fake_fetch(asset):
        return {
            "asset": asset, "data_valid": True,
            "ohlcv_raw": [list(b) for b in bars],
            "latest_bar_open_ts_ms": bars[-1][0],
            "current_price": bars[-1][4],
            "orderbook_depth_1pct_usd": 1_000_000.0,
        }
    monkeypatch.setattr(p, "_fetch_live_data", _fake_fetch)
    return p, bars


def _run(p, **kw):
    return asyncio.run(p.fetch_and_prepare("BTC", **kw))


def _wv_len(p):
    return len(p._wavelet_buffers["BTC"]["rsi_14"])


# ---------------------------------------------------------------------------
# task 1 — the decision tick forces a MISS
# ---------------------------------------------------------------------------
class TestDecisionTickForcesTheMiss:
    def test_primed_cache_then_decision_call_appends_exactly_one_wavelet_sample(
            self, tmp_path, monkeypatch, caplog):
        p, _ = _pipeline(tmp_path, monkeypatch)
        with caplog.at_level(logging.ERROR):
            _run(p)                          # watchdog pass: MISS, primes
            assert _wv_len(p) == 0, "a watchdog pass must never append"
            _run(p, for_decision=True)       # decision: forced MISS
        assert _wv_len(p) == 1, (
            "the decision tick must append exactly one wavelet sample — "
            "before P420 it HIT the primed cache and appended nothing")
        assert not [r for r in caplog.records if "indicator calculation "
                    "failed" in r.getMessage()], "the pipeline errored"

    def test_a_watchdog_hit_after_the_decision_appends_nothing(
            self, tmp_path, monkeypatch):
        p, _ = _pipeline(tmp_path, monkeypatch)
        _run(p)
        _run(p, for_decision=True)
        _run(p)                              # watchdog HIT
        _run(p)
        assert _wv_len(p) == 1, "a HIT (for_decision=False) must not append"

    def test_two_decision_calls_are_two_samples_never_more(
            self, tmp_path, monkeypatch):
        """No double-append INSIDE one decision: one for_decision call = one
        sample, whatever the cache state was before it."""
        p, _ = _pipeline(tmp_path, monkeypatch)
        _run(p, for_decision=True)           # cold: genuine MISS + append
        assert _wv_len(p) == 1
        _run(p, for_decision=True)           # primed: forced MISS + append
        assert _wv_len(p) == 2

    def test_the_forced_miss_does_not_refeed_the_atr_calculator(
            self, tmp_path, monkeypatch):
        """WIRE-2's calculator APPENDS a true range per call; the priming
        pass already fed this bar."""
        stop = _AdaptiveStop()
        p, _ = _pipeline(tmp_path, monkeypatch, adaptive_stop=stop)
        _run(p)
        assert stop.atr_calculator.calls == 1
        _run(p, for_decision=True)
        assert stop.atr_calculator.calls == 1, (
            "the forced miss re-fed the same bar into the ATR windows")
        _run(p)
        assert stop.atr_calculator.calls == 1

    def test_the_gmm_stash_is_rebuilt_on_the_decision_tick(
            self, tmp_path, monkeypatch):
        gmm = _GMM()
        p, _ = _pipeline(tmp_path, monkeypatch, gmm=gmm)
        r0 = _run(p)
        assert "_gmm_raw_features" in r0
        # a genuine HIT now serves the stash from the cache too
        r1 = _run(p)
        assert r1.get("_gmm_raw_features") == r0["_gmm_raw_features"]
        # the decision tick REBUILDS it (forced GMM miss): prove the rebuild
        # by making the fake model observable — a sentinel injected into the
        # cache must NOT survive a decision call, but does survive a HIT.
        p._ta_cache["BTC_gmm"]["raw_keys"]["_gmm_raw_features"] = ["SENTINEL"]
        assert _run(p).get("_gmm_raw_features") == ["SENTINEL"]
        assert _run(p, for_decision=True).get("_gmm_raw_features") != ["SENTINEL"]

    def test_the_smoother_advances_exactly_once_per_decision_call(
            self, tmp_path, monkeypatch):
        gmm = _GMM()
        p, _ = _pipeline(tmp_path, monkeypatch, gmm=gmm)
        gmm.label = 0
        _run(p, for_decision=True)           # state created: current=QUIET
        st = p._regime_smoother_state["BTC"]
        assert st["current"] == "QUIET_ACCUMULATION"
        gmm.label = 1                         # MOMENTUM_RALLY from here on
        _run(p, for_decision=True)
        assert st["pending"] == "MOMENTUM_RALLY" and st["count"] == 1
        _run(p)                               # watchdog HIT: no advance
        _run(p)
        assert st["pending"] == "MOMENTUM_RALLY" and st["count"] == 1, (
            "a watchdog pass advanced the smoother")
        _run(p, for_decision=True)            # second decision: confirms
        assert st["current"] == "MOMENTUM_RALLY" and st["count"] == 0

    def test_the_intrabar_and_structure_keys_are_served_on_a_hit(
            self, tmp_path, monkeypatch):
        p, bars = _pipeline(tmp_path, monkeypatch)
        _run(p)                               # prime
        r = _run(p)                           # HIT
        for k in ("bar_progress_4h", "volume_ratio_intrabar_pace",
                  "volume_ratio_effective", "structure_break_pct",
                  "structure_level"):
            assert k in r, f"{k} missing on a cache HIT"
        # and the intrabar trio is consistent with the pure helper at NOW
        exp = intrabar_volume_keys(r["volume_ratio"], bars[-1][0])
        assert abs(r["volume_ratio_effective"]
                   - exp["volume_ratio_effective"]) < 1e-9
        assert 0.0 < r["bar_progress_4h"] < 0.05   # ~90s into the bar

    def test_force_miss_defaults_to_false(self):
        sig = inspect.signature(MarketDataPipeline._predict_gmm_regime)
        assert sig.parameters["force_miss"].default is False
        src = inspect.getsource(MarketDataPipeline.fetch_and_prepare)
        assert "force_miss=for_decision" in src, (
            "the GMM cache is not forced on the decision tick — the TA "
            "MISS re-stores the same last_ts, so the GMM cache would still "
            "HIT and skip the stash")


# ---------------------------------------------------------------------------
# the pure helpers
# ---------------------------------------------------------------------------
class TestIntrabarHelpers:
    def test_bar_progress_is_a_function_of_now(self):
        open_ms = 1_000_000_000_000
        assert bar_progress_4h(open_ms, now_ts=open_ms / 1000 + 90) == \
            pytest.approx(90 / 14400)
        assert bar_progress_4h(open_ms, now_ts=open_ms / 1000 + 3.5 * 3600) \
            == pytest.approx(3.5 / 4)
        assert bar_progress_4h(open_ms, now_ts=open_ms / 1000 + 5 * 3600) \
            == 1.0

    def test_absent_or_bad_bar_ts_means_no_correction(self):
        for bad in (0, None, "x", float("nan")):
            assert bar_progress_4h(bad, now_ts=1.0) == 1.0
            k = intrabar_volume_keys(0.4, bad, now_ts=1.0)
            assert k["volume_ratio_effective"] == 0.4

    def test_pace_correction_matches_the_p339_arithmetic(self):
        open_ms = 1_000_000_000_000
        k = intrabar_volume_keys(0.02, open_ms, now_ts=open_ms / 1000 + 90)
        assert k["bar_progress_4h"] == pytest.approx(90 / 14400)
        assert k["volume_ratio_intrabar_pace"] == pytest.approx(0.02 / 0.20)
        assert k["volume_ratio_effective"] == pytest.approx(0.1)
        k2 = intrabar_volume_keys(0.9, open_ms, now_ts=open_ms / 1000 + 7200)
        assert k2["volume_ratio_intrabar_pace"] == pytest.approx(0.9 / 0.5)
        assert k2["volume_ratio_effective"] == pytest.approx(1.8)
        assert intrabar_volume_keys(9.0, open_ms, now_ts=open_ms / 1000 + 90
                                    )["volume_ratio_intrabar_pace"] == 5.0


# ---------------------------------------------------------------------------
# task 2 — vol_percentile ranks the last COMPLETED bar
# ---------------------------------------------------------------------------
class TestGmmVolumeRank:
    def test_in_progress_bar_is_excluded_from_the_ranking(self):
        open_ms = 1_000_000_000_000
        vols = [10.0, 20.0, 30.0, 40.0, 0.03]    # last = partial bar
        win, val = gmm_volume_rank_input(vols, open_ms,
                                         now_ts=open_ms / 1000 + 90)
        assert win == [10.0, 20.0, 30.0, 40.0] and val == 40.0

    def test_construction_the_partial_bars_volume_never_enters_the_rank(self):
        """Perturb ONLY the in-progress bar's volume: window and value must
        be bit-identical (the P164 construction-test shape)."""
        open_ms = 1_000_000_000_000
        base = list(np.random.default_rng(1).lognormal(8, 0.3, 200))
        a = gmm_volume_rank_input(base + [0.5], open_ms,
                                  now_ts=open_ms / 1000 + 90)
        b = gmm_volume_rank_input(base + [1e9], open_ms,
                                  now_ts=open_ms / 1000 + 90)
        assert a == b

    def test_a_completed_bar_is_ranked_as_before(self):
        open_ms = 1_000_000_000_000
        vols = [10.0, 20.0, 30.0]
        win, val = gmm_volume_rank_input(vols, open_ms,
                                         now_ts=open_ms / 1000 + 5 * 3600)
        assert win == vols and val == 30.0

    def test_absent_bar_ts_keeps_the_legacy_full_frame_rank(self):
        vols = [1.0, 2.0, 3.0]
        assert gmm_volume_rank_input(vols, 0) == (vols, 3.0)
        assert gmm_volume_rank_input(vols, None) == (vols, 3.0)

    def test_window_is_the_training_width(self):
        assert GMM_VOL_PCT_WINDOW == 1024
        vols = list(range(1, 1500))
        win, val = gmm_volume_rank_input(vols, 0)
        assert len(win) == 1024 and val == 1499.0

    def test_only_vol_percentile_is_volume_derived_in_the_gmm_vector(self):
        """The other eight inputs are price/return functions — a partial
        bar's close IS the current price, so they stay as they are."""
        src = inspect.getsource(MarketDataPipeline._predict_gmm_regime)
        head = src[:src.index("features = _np.array([")]
        code = "\n".join(ln for ln in head.splitlines()
                         if not ln.strip().startswith("#"))
        # the pre-P420 defect shape: ranking the raw frame's last row
        assert "vols[-1]" not in code and "_np.sort(vols)" not in code
        assert "gmm_volume_rank_input(" in code

    def test_live_pipeline_vol_percentile_ignores_the_partial_bar(
            self, tmp_path, monkeypatch):
        """End to end: doubling the partial bar's volume must not move the
        GMM feature vector the live path stashes."""
        gmm = _GMM()
        p, bars = _pipeline(tmp_path, monkeypatch, gmm=gmm)
        r0 = _run(p, for_decision=True)
        f0 = list(r0["_gmm_raw_features"])
        bars[-1][5] *= 1000.0   # the fake fetch copies from `bars`
        r1 = _run(p, for_decision=True)
        f1 = list(r1["_gmm_raw_features"])
        assert f0[5] == f1[5], "vol_percentile moved with the partial bar"


# ---------------------------------------------------------------------------
# task 3 (pipeline half) — the stash is withheld on the OOD fallback
# ---------------------------------------------------------------------------
class TestStashOrdering:
    def test_distribution_shift_fallback_leaves_no_stash(self, tmp_path,
                                                         monkeypatch):
        gmm = _GMM()
        p, _ = _pipeline(tmp_path, monkeypatch, gmm=gmm)
        # a scaler with scale 1e-9 makes every |z| enormous => fallback
        p._gmm_configs["BTC"]["scaler_scale"] = [1e-9] * 9
        r = _run(p, for_decision=True)
        assert r.get("_gmm_fallback") == "distribution_shift"
        assert "_gmm_raw_features" not in r, (
            "a stash on the fallback path lets the jump shadow count an ADX "
            "proxy label as a GMM switch")

    def test_the_stash_rides_the_gmm_cache(self):
        src = inspect.getsource(MarketDataPipeline._predict_gmm_regime)
        blk = src[src.index("_gmm_raw_keys = {"):]
        blk = blk[:blk.index("}")]
        assert '"_gmm_raw_features"' in blk


# ---------------------------------------------------------------------------
# task 15 — recorded, not routed
# ---------------------------------------------------------------------------
def test_whale_deadband_third_copy_is_recorded_at_the_site():
    src = inspect.getsource(mdp)
    i = src.index("_whale_count = _wp.whale_count")
    assert "whale_direction_from_pressure" in src[i:i + 900], (
        "the third copy of the +/-0.3 whale deadband must name the "
        "single-sourced helper and why it is not imported (P293d/P172)")
