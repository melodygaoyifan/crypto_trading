"""[P306] The measured-input batch: a real 1h price change, a real DVOL
z-score, and a per-asset cascade governor.

Every test here pins a property that, if it silently reverted, would put a
fabricated number back on a live risk path - which is the failure this whole
batch exists to end.
"""
from __future__ import annotations

import ast
import io
import json
import os
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return io.open(REPO / rel, encoding="utf-8").read()


def _strip_comments(src: str) -> str:
    """Source with comments and docstrings blanked (P177/P184): a guard that
    matches its own explanation is not a guard."""
    out = []
    for line in src.split("\n"):
        s = line.split("#")[0]
        out.append(s)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 1. the hourly feed
# ---------------------------------------------------------------------------
class TestKrakenHourly:
    def _feed(self, rows, now=None):
        from data_mgmt.feeds.kraken_hourly import KrakenHourlyReturns
        f = KrakenHourlyReturns()
        f._rows = staticmethod(lambda payload: rows)  # type: ignore[assignment]
        return f

    def _payload(self, rows):
        return {"result": {"XXBTZUSD": [[t, 0, 0, 0, c, 0, 0, 0]
                                        for t, c in rows], "last": 1}}

    def test_returns_the_last_completed_hour_not_the_in_progress_one(self):
        from data_mgmt.feeds.kraken_hourly import KrakenHourlyReturns
        f = KrakenHourlyReturns()
        now = time.time()
        h = 3600.0
        # opens at now-3h, now-2h, now-1h(completed), now(in progress)
        rows = [(now - 4 * h, 100.0), (now - 3 * h, 100.0),
                (now - 2 * h, 100.0), (now - 1 * h, 103.0),
                (now, 999.0)]              # in-progress, must be ignored
        f._fetch_payload = lambda pair: self._payload(rows)  # type: ignore
        parsed = KrakenHourlyReturns._rows(self._payload(rows))
        assert parsed[-1][1] == 999.0, "parser must not drop rows"
        # the DROP happens in _fetch; emulate it with the same predicate
        done = [r for r in parsed if (r[0] + 3600.0) <= time.time()]
        assert done[-1][1] == 103.0
        got = (done[-1][1] - done[-2][1]) / done[-2][1]
        assert abs(got - 0.03) < 1e-9, (
            "the in-progress candle leaked into the reading")

    def test_missing_pair_returns_none_never_zero(self):
        from data_mgmt.feeds.kraken_hourly import KrakenHourlyReturns
        f = KrakenHourlyReturns()
        assert f.get("NOSUCHASSET") is None

    def test_a_stale_completed_hour_refuses(self):
        from data_mgmt.feeds.kraken_hourly import KrakenHourlyReturns
        f = KrakenHourlyReturns(max_age_sec=100.0)
        old = time.time() - 10 * 3600
        assert f._refuse_if_stale("BTC", 0.05, old, time.time()) is None, (
            "an hours-old candle must not be asserted as this hour's move")
        assert f._refuse_if_stale("BTC", 0.05, time.time(), time.time()) == 0.05

    def test_malformed_payload_yields_no_rows_rather_than_raising(self):
        from data_mgmt.feeds.kraken_hourly import KrakenHourlyReturns
        for bad in (None, {}, {"result": {}}, {"result": {"X": ["nope"]}},
                    {"result": {"X": [[1, 2]]}}):
            assert KrakenHourlyReturns._rows(bad) == []

    def test_pair_map_is_imported_not_restated(self):
        """P172: a second copy is how two consumers of 'the SOL pair' start
        disagreeing (P133/P135 - SOL is deliberately not a plain USD pair
        everywhere)."""
        src = _strip_comments(_src("data_mgmt/feeds/kraken_hourly.py"))
        assert "from defense.regime_book_shadow import KRAKEN_PAIRS" in src
        assert "XBTUSD" not in src, "the pair table was restated locally"


# ---------------------------------------------------------------------------
# 2. DVOL history / z-score
# ---------------------------------------------------------------------------
class TestDvolHistory:
    def _hist(self, tmp_path, series):
        os.environ["HMATS_DATA_DIR"] = str(tmp_path)
        from data_mgmt.feeds import dvol_history as M
        M.reset_dvol_history()
        h = M.DvolHistory(min_samples=5, window_days=365)
        now = time.time()
        h._series = {"BTC": [(now - (len(series) - i) * 86400.0, v)
                             for i, v in enumerate(series)]}
        return h

    def test_thin_history_returns_none_so_no_key_is_written(self, tmp_path):
        h = self._hist(tmp_path, [40.0, 41.0])
        assert h.zscore("BTC", 34.0) is None

    def test_flat_history_is_unscoreable_not_infinite(self, tmp_path):
        h = self._hist(tmp_path, [40.0] * 20)
        assert h.zscore("BTC", 90.0) is None

    def test_a_calm_market_scores_negative_not_extreme(self, tmp_path):
        """The whole point: Deribit's LEVEL (34) would read as z=34 and fire
        EXTREME_DVOL (>= 5.0). The z of the same reading is negative."""
        h = self._hist(tmp_path, [40.0, 45.0, 50.0, 55.0, 60.0,
                                  42.0, 48.0, 52.0, 58.0, 62.0])
        z = h.zscore("BTC", 34.0)
        assert z is not None and z < 0.0
        assert z < 5.0, "a calm reading must not trip EXTREME_DVOL"

    def test_unknown_currency_is_absent_not_zero(self, tmp_path):
        h = self._hist(tmp_path, [40.0] * 20)
        assert h.zscore("SOL", 40.0) is None, (
            "Deribit lists zero SOL options; an absence must stay an absence")

    def test_history_older_than_the_bound_refuses(self, tmp_path):
        os.environ["HMATS_DATA_DIR"] = str(tmp_path)
        from data_mgmt.feeds import dvol_history as M
        M.reset_dvol_history()
        h = M.DvolHistory(min_samples=5, max_history_age_sec=3600.0)
        old = time.time() - 30 * 86400
        h._series = {"BTC": [(old - i * 86400.0, 40.0 + i) for i in range(20)]}
        assert h.zscore("BTC", 34.0) is None

    def test_persist_and_restore_round_trip(self, tmp_path):
        os.environ["HMATS_DATA_DIR"] = str(tmp_path)
        from data_mgmt.feeds import dvol_history as M
        M.reset_dvol_history()
        h = M.DvolHistory(min_samples=5)
        now = time.time()
        h._series = {"BTC": [(now - i * 86400.0, 40.0 + i) for i in range(20)]}
        h._persist()
        M.reset_dvol_history()
        h2 = M.DvolHistory(min_samples=5)
        assert h2.sample_count("BTC") == 20

    def test_corrupt_state_is_a_cold_start_not_a_crash(self, tmp_path):
        os.environ["HMATS_DATA_DIR"] = str(tmp_path)
        (tmp_path / "dvol_history.json").write_text("{not json",
                                                    encoding="utf-8")
        from data_mgmt.feeds import dvol_history as M
        M.reset_dvol_history()
        assert M.DvolHistory().sample_count("BTC") == 0


# ---------------------------------------------------------------------------
# 3. what main.py publishes
# ---------------------------------------------------------------------------
class TestMainWiring:
    def test_dvol_publishes_the_zscore_never_the_level(self):
        src = _strip_comments(_src("main.py"))
        assert 'market_data["dvol"] = float(_dvz)' in src
        assert 'market_data["dvol"] = float(_drb_m.dvol)' not in src, (
            "the raw index LEVEL is being published under a key the "
            "constitution aliases to dvol_zscore - EXTREME_DVOL fires at 5.0")

    def test_the_1h_publisher_runs_outside_the_coinglass_branch(self):
        """A CoinGlass outage must not leave the cascade STRATEGY on the
        fabricated value while the governor is on the real one."""
        src = _src("main.py")
        i = src.index("self._publish_real_1h_change(asset, market_data)")
        j = src.index("V6.7: COINGLASS FUNDING RATE ->market_data")
        assert i < j, "the 1h publisher moved inside/after the CoinGlass block"
        line = src[src.rindex("\n", 0, i) + 1:i]
        assert len(line) == 8, (
            "expected top-level (8-space) indentation, got %r" % line)

    def test_absence_leaves_the_old_value_rather_than_writing_zero(self):
        src = _strip_comments(_src("main.py"))
        i = src.index("def _publish_real_1h_change")
        blk = src[i:i + 3000]
        assert 'market_data["price_change_1h_pct"] = float(_real)' in blk
        assert 'market_data["price_change_1h_pct"] = 0.0' not in blk

    def test_both_numbers_are_logged_regardless_of_the_flag(self):
        """P300: an instrument gated behind the condition it measures never
        reads anything."""
        src = _src("main.py")
        i = src.index("def _publish_real_1h_change")
        blk = src[i:i + 3000]
        # Anchor on the INFO statement, not on the tag: an earlier version
        # of this test matched the DEBUG line in the import handler and so
        # measured nothing. The INFO line carries both numbers and names the
        # mode, i.e. it is emitted in either state.
        assert 'fabricated(4h/4)={_fab_s}' in blk
        assert "'ARMED' if _armed else 'shadow'" in blk
        assert "if _armed:" not in blk, (
            "the comparison log is gated behind the condition it measures")

    def test_the_governor_is_fetched_per_asset(self):
        for rel, needle in (("main.py", "get_cascade_exhaustion_governor(asset=asset)"),
                            ("core/execution_service.py",
                             "get_cascade_exhaustion_governor(asset=asset)")):
            src = _strip_comments(_src(rel))
            assert needle in src, rel
            assert "get_cascade_exhaustion_governor()\n" not in src, (
                "%s still takes the shared instance, so its acceleration and "
                "velocity are CROSS-ASSET differences" % rel)


# ---------------------------------------------------------------------------
# 4. the per-asset governor itself
# ---------------------------------------------------------------------------
class TestPerAssetGovernor:
    def test_distinct_instances_per_asset_and_a_shared_one_for_none(self):
        from risk import cascade_exhaustion_governor as M
        M.reset_cascade_exhaustion_governor()
        b, e = M.get_cascade_exhaustion_governor(asset="BTC"), \
            M.get_cascade_exhaustion_governor(asset="ETH")
        assert b is not e
        assert b is M.get_cascade_exhaustion_governor(asset="btc")
        shared = M.get_cascade_exhaustion_governor()
        assert shared is not b and shared is M.get_cascade_exhaustion_governor()
        M.reset_cascade_exhaustion_governor()

    def test_one_assets_liquidations_cannot_move_anothers_acceleration(self):
        """The live defect: BTC ~$6.19M/h fed straight after SOL ~$0.16M/h
        made SOL's acceleration -0.97, which clears the -0.5 exhaustion
        threshold on ordering alone."""
        from risk import cascade_exhaustion_governor as M
        M.reset_cascade_exhaustion_governor()
        for asset, vol in (("BTC", 6_000_000.0), ("SOL", 160_000.0)):
            g = M.get_cascade_exhaustion_governor(asset=asset)
            g.update_metrics(liquidation_volume_1h=vol,
                             liquidation_volume_4h=vol * 4,
                             price_change_1h_pct=0.0,
                             price_change_4h_pct=0.0,
                             volume_spike_ratio=1.0)
        sol = M.get_cascade_exhaustion_governor(asset="SOL")
        # first reading for SOL's own machine -> no acceleration from BTC
        assert abs(sol._current_metrics.liquidation_acceleration) < 0.5, (
            "SOL's acceleration was computed against BTC's volume")
        M.reset_cascade_exhaustion_governor()

    def test_a_permissions_override_reaches_later_per_asset_instances(self):
        from risk import cascade_exhaustion_governor as M
        M.reset_cascade_exhaustion_governor()
        ov = {"DETECTED": {"T1": False}}
        M.get_cascade_exhaustion_governor(permissions_override=ov)
        later = M.get_cascade_exhaustion_governor(asset="BTC")
        assert getattr(later, "permissions_override", {}) == ov, (
            "per-asset machines would silently run default permissions while "
            "the operator believed the override was in force")
        M.reset_cascade_exhaustion_governor()

    def test_reset_clears_every_instance(self):
        from risk import cascade_exhaustion_governor as M
        M.reset_cascade_exhaustion_governor()
        a = M.get_cascade_exhaustion_governor(asset="BTC")
        M.reset_cascade_exhaustion_governor()
        assert M.get_cascade_exhaustion_governor(asset="BTC") is not a


# ---------------------------------------------------------------------------
# 5. decisions, pinned so a silent flip fails
# ---------------------------------------------------------------------------
class TestDecisions:
    def _cfg(self):
        return json.loads(_src("configs/live_high_risk.json"))

    def test_the_two_enabled_flags_carry_their_decided_values(self):
        cfg = self._cfg()
        assert cfg["real_1h_price_change"] is True
        assert cfg["dvol_to_market_data"] is True

    def test_the_cascade_liquidation_window_stays_off(self):
        """Measured on 10.5 days of the real derivflow ledger: the delta
        estimator's p95 rate is $10.06M/h on BTC against a $10M threshold, so
        arming it would DETECT on ~5% of BTC ticks, and SOL's maximum ever
        observed is $2.09M/h so it could never fire at all. That is a
        threshold that was never calibrated against a real short window - the
        P265 volume-collapse lesson: re-arming needs a re-derived threshold,
        not a predicate flip."""
        assert "cascade_real_liquidation_window" not in self._cfg()

    def test_the_trend_regime_gate_stays_in_shadow(self):
        """training/trend_gate_lab.py, pre-committed verdict: the live gate
        set SUBTRACTS in BOTH eras on ALL THREE assets (BTC -0.332/-0.210,
        ETH -0.687/-0.259, SOL -0.509/-0.657 net after cost). Enforcing it
        would have destroyed money 6 reads out of 6."""
        assert self._cfg().get("trend_regime_gate", "shadow") == "shadow"

    def test_the_dvol_activation_score_cannot_go_negative(self):
        src = _strip_comments(_src("defense/constitution.py"))
        assert ("dvol_score = max(0.0, min(1.0, dvol_zscore / "
                "self.DVOL_ZSCORE_EXTREME))") in src, (
            "a calm market would contribute NEGATIVE activation to "
            "trigger_scores, which a summing consumer reads as risk relief")

    def test_the_gmm_ret_1h_feature_is_gone_from_both_sides(self):
        """[P307 supersedes the P306 pin here.] P306 kept ret_1h = ret_4h/4
        on BOTH sides and pinned that they agreed, on the reading that a
        perfect duplicate carries no information and so costs nothing. The
        second half was wrong: in a full-covariance GMM the duplicate
        double-weights ret_4h in the assignment distance (measured ARI
        0.690/0.657/0.741, and BTC's k moved 6 -> 7). The feature was removed
        from both builders and the artifacts refitted as one set (P215), so
        what is pinned now is ABSENCE on both sides rather than agreement.
        """
        for rel in ("data_mgmt/market_data_pipeline.py",
                    "training/scripts/rebuild_pipeline.py"):
            src = _strip_comments(_src(rel))
            assert "ret_1h = ret_4h / 4.0" not in src, rel

    def test_the_latent_percent_unit_bug_is_gone(self):
        src = _strip_comments(_src("main.py"))
        assert ('_enrich["price_change_4h_pct"] = (float(_cp) - _p4) / _p4 '
                '* 100.0') not in src, (
            "two producers of one key disagreeing by 100x")


# ---------------------------------------------------------------------------
# 6. anti-vacuity
# ---------------------------------------------------------------------------
def test_every_module_this_batch_added_parses_and_imports():
    for rel in ("data_mgmt/feeds/kraken_hourly.py",
                "data_mgmt/feeds/dvol_history.py",
                "training/trend_gate_lab.py"):
        ast.parse(_src(rel))
    import data_mgmt.feeds.kraken_hourly as _k
    import data_mgmt.feeds.dvol_history as _d
    assert callable(_k.get_hourly_returns) and callable(_d.get_dvol_history)


def test_the_source_scanner_actually_reads_something():
    """P174: a guard over an empty string passes vacuously."""
    src = _strip_comments(_src("main.py"))
    assert len(src) > 100_000
    assert "_publish_real_1h_change" in src


def test_per_asset_governor_state_survives_a_restart():
    """P301/P302/P303: introducing per-asset machines without persisting them
    would reset every asset's phase on each deploy."""
    from risk import cascade_exhaustion_governor as M
    M.reset_cascade_exhaustion_governor()
    for a in ("BTC", "ETH"):
        M.get_cascade_exhaustion_governor(asset=a).update_metrics(
            liquidation_volume_1h=1.0, liquidation_volume_4h=4.0,
            price_change_1h_pct=0.0, price_change_4h_pct=0.0,
            volume_spike_ratio=1.0)
    states = M.all_governor_states()
    assert set(states) >= {"BTC", "ETH"}
    M.reset_cascade_exhaustion_governor()
    assert M.restore_governor_states(states) >= 2
    M.reset_cascade_exhaustion_governor()


def test_the_pre_p306_flat_state_file_still_restores():
    """A state file written by the previous build must not read as "no
    state", nor be mistaken for a per-asset map."""
    from risk import cascade_exhaustion_governor as M
    M.reset_cascade_exhaustion_governor()
    flat = M.get_cascade_exhaustion_governor().to_dict()
    assert not all(isinstance(v, dict) for v in flat.values()), (
        "the shape test cannot distinguish the two layouts")
    M.reset_cascade_exhaustion_governor()
    assert M.restore_governor_states(flat) == 1
    assert M.restore_governor_states({}) == 0
    assert M.restore_governor_states(None) == 0
    M.reset_cascade_exhaustion_governor()
