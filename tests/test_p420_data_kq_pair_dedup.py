"""[P420] kraken_quant pair strategies: one append per BAR, not per call —
and the BEAR-bucket data quality is no longer capped by two phantom names.

main.py calls generate_signal() once per ASSET per 4H tick with the same
cross-asset cache, so RelativeStrength and Kalman received the SAME pair
prices on 3 calls per tick and appended all three (2 of 3 a 4h-stale
duplicate): the P390 warmup clocks counted 3x, the RS/spread buffers held
triplets and the Kalman filter was updated 3x per bar on one observation.
The converter now carries `bar_ts` (the pipeline's latest_bar_open_ts_ms)
and each asset's price is appended only when ITS bar advanced.

BEAR expected `liquidations` + `liquidation_intensity`; `_has_field` probes
`<name>_<al>` and main.py writes `liquidation_volume_<al>` (no producer of
`liquidation_intensity*` exists) — dq was capped at 4/6 = 0.667 forever."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from agents.kraken_quant_agent import (
    KrakenQuantAgentV6, RelativeStrengthStrategy, KalmanCointegrationStrategy,
    Regime, _EXPECTED_FIELDS, kq_dedup_key, kq_bar_advanced, KQ_BAR_TS_KEY,
)

BAR = 14400.0


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))


def _rs_md(bar, btc, sol, ts=None):
    return {"timestamp": ts if ts is not None else bar * BAR + 60.0,
            KQ_BAR_TS_KEY: 1_700_000_000_000.0 + bar * BAR * 1000.0,
            "prices": {"BTC": float(btc), "SOL": float(sol)}}


def _k_md(bar, sol, eth):
    return {"timestamp": bar * BAR + 60.0,
            KQ_BAR_TS_KEY: 1_700_000_000_000.0 + bar * BAR * 1000.0,
            "prices": {"SOL": float(sol), "ETH": float(eth)}}


# ---------------------------------------------------------------------------
# the pure dedup helpers
# ---------------------------------------------------------------------------
class TestDedupHelpers:
    def test_bar_ts_wins_over_timestamp(self):
        assert kq_dedup_key({"bar_ts": 5.0, "timestamp": 9.0}) == 5.0

    def test_timestamp_is_the_fallback_and_none_means_always_append(self):
        assert kq_dedup_key({"timestamp": 9.0}) == 9.0
        assert kq_dedup_key({}) is None
        assert kq_dedup_key({"bar_ts": None, "timestamp": "x"}) is None
        assert kq_dedup_key({"bar_ts": float("nan"), "timestamp": 2.0}) == 2.0
        assert kq_bar_advanced({}, "BTC", None) is True

    def test_advanced_is_per_asset(self):
        last = {"BTC": 5.0}
        assert kq_bar_advanced(last, "BTC", 5.0) is False
        assert kq_bar_advanced(last, "SOL", 5.0) is True
        assert kq_bar_advanced(last, "BTC", 6.0) is True


# ---------------------------------------------------------------------------
# RelativeStrength
# ---------------------------------------------------------------------------
class TestRelativeStrengthDedup:
    def test_three_calls_in_one_tick_append_once(self):
        s = RelativeStrengthStrategy()
        for _ in range(3):
            s.update(_rs_md(1, 50000.0, 90.0, ts=None))
        assert len(s.price_buffer["BTC"]) == 1
        assert len(s.price_buffer["SOL"]) == 1

    def test_the_call_timestamp_alone_does_not_defeat_the_dedup(self):
        """the three live calls carry three different `timestamp`s (seconds
        apart) but ONE bar_ts — bar_ts must govern"""
        s = RelativeStrengthStrategy()
        for i in range(3):
            s.update(_rs_md(1, 50000.0, 90.0, ts=1000.0 + i))
        assert len(s.price_buffer["BTC"]) == 1

    def test_a_new_bar_appends_again(self):
        s = RelativeStrengthStrategy()
        s.update(_rs_md(1, 50000.0, 90.0))
        s.update(_rs_md(2, 50100.0, 91.0))
        assert len(s.price_buffer["BTC"]) == 2

    def test_dedup_is_per_asset_so_a_late_peer_still_lands(self):
        """first tick of a process: BTC's call carries only BTC; SOL's price
        arrives on SOL's own call in the same bar and must be appended"""
        s = RelativeStrengthStrategy()
        md = _rs_md(1, 50000.0, 90.0)
        s.update({**md, "prices": {"BTC": 50000.0}})
        s.update(md)                              # BTC dup, SOL new
        s.update(md)
        assert len(s.price_buffer["BTC"]) == 1
        assert len(s.price_buffer["SOL"]) == 1

    def test_legacy_callers_without_bar_ts_append_per_distinct_timestamp(self):
        s = RelativeStrengthStrategy()
        for i in range(3):
            s.update({"timestamp": 1000.0 + i * BAR,
                      "prices": {"BTC": 1.0, "SOL": 1.0}})
        assert len(s.price_buffer["BTC"]) == 3

    def test_rs_buffer_grows_once_per_bar_after_warmup(self):
        s = RelativeStrengthStrategy()
        for i in range(60):
            s.update(_rs_md(i, 50000.0 * (1.001 ** i), 90.0 * (1.002 ** i)))
        n_rs = len(s.rs_buffer)
        assert n_rs >= 1
        for _ in range(3):                        # duplicate-bar calls
            s.update(_rs_md(59, 50000.0 * (1.001 ** 59), 90.0 * (1.002 ** 59)))
        assert len(s.rs_buffer) == n_rs, "rs appended on a duplicate bar"
        s.update(_rs_md(60, 50000.0 * (1.001 ** 60), 90.0 * (1.002 ** 60)))
        assert len(s.rs_buffer) == n_rs + 1

    def test_last_bar_ts_survives_a_restart(self):
        s = RelativeStrengthStrategy()
        s.update(_rs_md(7, 50000.0, 90.0))
        s2 = RelativeStrengthStrategy()              # restore at construction
        assert s2._last_bar_ts.get("BTC") == kq_dedup_key(_rs_md(7, 1, 1))
        assert len(s2.price_buffer["BTC"]) == 1
        s2.update(_rs_md(7, 50000.0, 90.0))          # same bar after restart
        assert len(s2.price_buffer["BTC"]) == 1, (
            "a restart inside the bar re-appended the bar the file holds")
        s2.update(_rs_md(8, 50000.0, 90.0))
        assert len(s2.price_buffer["BTC"]) == 2


# ---------------------------------------------------------------------------
# Kalman
# ---------------------------------------------------------------------------
class TestKalmanDedup:
    def test_three_calls_in_one_tick_append_once(self):
        k = KalmanCointegrationStrategy(pair=("SOL", "ETH"))
        for _ in range(3):
            k.update(_k_md(1, 90.0, 2400.0))
        assert len(k.price_buffer["SOL"]) == 1
        assert len(k.price_buffer["ETH"]) == 1

    def test_filter_state_and_spread_move_once_per_bar(self):
        k = KalmanCointegrationStrategy(pair=("SOL", "ETH"))
        rng = np.random.default_rng(7)
        eth = 2400.0
        for i in range(60):
            eth *= float(np.exp(rng.normal(0, 0.005)))
            k.update(_k_md(i, eth * 0.04 * (1 + rng.normal(0, 0.002)), eth))
        theta = k.theta.copy()
        n_sp = len(k.spread_buffer)
        for _ in range(3):
            k.update(_k_md(59, eth * 0.04, eth))     # duplicate bar
        assert np.array_equal(k.theta, theta), "the filter updated on a dup"
        assert len(k.spread_buffer) == n_sp, "spread appended on a dup"
        k.update(_k_md(60, eth * 0.04, eth))
        assert len(k.spread_buffer) == n_sp + 1

    def test_last_bar_ts_survives_a_restart(self):
        k = KalmanCointegrationStrategy(pair=("SOL", "ETH"))
        k.update(_k_md(3, 90.0, 2400.0))
        k2 = KalmanCointegrationStrategy(pair=("SOL", "ETH"))
        assert k2._last_bar_ts.get("SOL") == kq_dedup_key(_k_md(3, 1, 1))
        k2.update(_k_md(3, 90.0, 2400.0))
        assert len(k2.price_buffer["SOL"]) == 1

    def test_the_p390_persist_sites_still_bracket_both_mutations(self):
        for cls in (RelativeStrengthStrategy, KalmanCointegrationStrategy):
            src = inspect.getsource(cls.update)
            assert src.count("self._persist_warmup()") >= 2


# ---------------------------------------------------------------------------
# the converter carries bar_ts
# ---------------------------------------------------------------------------
class TestConverterCarriesBarTs:
    def test_bar_ts_comes_from_latest_bar_open_ts_ms(self):
        n = KrakenQuantAgentV6._convert_market_data(
            "BTC", {"price": 1.0, "latest_bar_open_ts_ms": 1_700_000_000_000})
        assert n[KQ_BAR_TS_KEY] == 1_700_000_000_000.0

    def test_absent_or_bad_bar_ts_is_none(self):
        assert KrakenQuantAgentV6._convert_market_data(
            "BTC", {"price": 1.0})[KQ_BAR_TS_KEY] is None
        assert KrakenQuantAgentV6._convert_market_data(
            "BTC", {"price": 1.0, "latest_bar_open_ts_ms": "x"})[KQ_BAR_TS_KEY] is None

    def test_the_three_live_calls_share_one_bar_ts(self):
        """what makes the dedup work: the pipeline's bar open is 4H-aligned
        for every asset, so the three per-asset calls carry one key"""
        keys = {KrakenQuantAgentV6._convert_market_data(
            a, {"price": 1.0, "latest_bar_open_ts_ms": 1_700_000_000_000,
                "timestamp": 1000.0 + i})[KQ_BAR_TS_KEY]
            for i, a in enumerate(("BTC", "ETH", "SOL"))}
        assert len(keys) == 1


# ---------------------------------------------------------------------------
# task 5 — BEAR data quality
# ---------------------------------------------------------------------------
class TestBearExpectedFields:
    def test_phantom_names_are_gone(self):
        bear = _EXPECTED_FIELDS[Regime.BEAR]
        assert "liquidations" not in bear
        assert "liquidation_intensity" not in bear
        assert "liquidation_volume" in bear

    def test_a_main_py_shaped_tick_reaches_full_bear_quality(self):
        """main.py writes liquidation_volume_<al>, taker_ratio_<al>,
        open_interest_<al>, funding_rate_<al>, price_<al>"""
        flat = {"price_btc": 1.0, "open_interest_btc": 2.0,
                "liquidation_volume_btc": 3.0, "taker_ratio_btc": 1.1,
                "funding_rate_btc": 0.0001}
        exp = _EXPECTED_FIELDS[Regime.BEAR]
        present = [k for k in exp
                   if KrakenQuantAgentV6._has_field(flat, k, "BTC")]
        assert len(present) / len(exp) == 1.0, sorted(set(exp) - set(present))

    def test_the_old_names_would_still_be_missing_on_that_tick(self):
        """the pin that explains the 0.667 cap"""
        flat = {"liquidation_volume_btc": 3.0}
        assert not KrakenQuantAgentV6._has_field(flat, "liquidations", "BTC")
        assert not KrakenQuantAgentV6._has_field(
            flat, "liquidation_intensity", "BTC")
        assert KrakenQuantAgentV6._has_field(flat, "liquidation_volume", "BTC")

    def test_the_converter_still_maps_volume_into_the_nested_liquidations(self):
        n = KrakenQuantAgentV6._convert_market_data(
            "BTC", {"price": 1.0, "liquidation_volume_btc": 3.0})
        assert n["liquidations"] == {"BTC": 3.0}
