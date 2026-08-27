"""[P418] The P416 residue, closed: every remaining mechanism that could pay
fees or liquidate a held position without a trend change.

  1. FLASH_CRASH carve-out now reads the SLEEVE book (was: the empty Kraken
     book, so a +/-5% 4h bar market-flattened a held position -- P338 class).
  2. The three engine HOLD paths (BEST_OF_N_HOLD / BLACK_SWAN_SENTINEL /
     PRE_ALPHA_HOLD) stamp a declared `hold_current_position` marker the
     sleeve translator honors -- they used to liquidate via the zero-target
     branch while documented as HOLD.
  3. A transient skew fetch failure carries the HELD direction (was: seat
     handed to a different decider for one tick = a direction change with no
     signal change).
  4. An inter-tick venue stop-out now arms the P232 re-entry cooldown (was:
     stop fill -> immediate re-entry next tick, a full fee round trip).
  5. The FastRisk depth baseline refuses the pipeline's fabricated 500k
     floor, and volatility_30m finally HAS a producer (the operator-armed 4x
     vol-spike REDUCE_50 was structurally inert -- baseline never > 0).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# 2. hold_current_position marker
# ---------------------------------------------------------------------------
class TestHoldCurrentPosition:
    def test_translator_honors_the_marker_before_exposure_arithmetic(self):
        import main
        from integration.integration_v36 import TradeIntentV36
        it = TradeIntentV36(asset="ETH")
        it.veto_active = False
        it.direction = 0.0
        it.target_exposure = 0.0          # the liquidation-reading shape
        it.hold_current_position = True
        d, why = main.sleeve_direction_from_intent(it, 0.0)
        assert d is main.SLEEVE_HOLD and why == "hold_current_position"

    def test_without_the_marker_the_old_semantics_are_unchanged(self):
        import main
        from integration.integration_v36 import TradeIntentV36
        it = TradeIntentV36(asset="ETH")
        it.veto_active = False
        it.direction = 0.0
        it.target_exposure = 0.0
        d, why = main.sleeve_direction_from_intent(it, 0.0)
        assert d == 0.0, "a genuine zero-target intent must still flatten"

    def test_the_field_is_declared_default_false(self):
        from integration.integration_v36 import TradeIntentV36
        assert TradeIntentV36(asset="BTC").hold_current_position is False

    def test_all_three_hold_paths_stamp_the_marker(self):
        src = (REPO / "integration" / "integration_v36.py").read_text(
            encoding="utf-8-sig")
        for tag in ("[BEST_OF_N_HOLD]", "[BLACK_SWAN_SENTINEL]",
                    "[PRE_ALPHA_HOLD]"):
            i = src.find(tag)
            assert i > 0, tag
            window = src[i:i + 400]
            assert "hold_current_position = True" in window, (
                f"{tag} no longer stamps the P418 marker -- its 'HOLD' would "
                f"liquidate the sleeve via the zero-target branch again")


# ---------------------------------------------------------------------------
# 1. flash-crash carve-out reads the sleeve book
# ---------------------------------------------------------------------------
class TestFlashCrashCarveOut:
    def test_the_carveout_reads_the_sleeve_position_feed(self):
        src = (REPO / "integration" / "integration_v36.py").read_text(
            encoding="utf-8-sig")
        i = src.find("_skip_flash_for_existing")
        assert i > 0
        window = src[max(0, i - 800):i]
        assert "sleeve_position_contracts" in window, (
            "the carve-out is back to reading only the empty Kraken book -- "
            "a +/-5% 4h bar (a RALLY included) flattens a held position")


# ---------------------------------------------------------------------------
# 3. skew transient-fetch-failure carry
# ---------------------------------------------------------------------------
def _skew(last_good=None, fetch_rows=None):
    from defense.skew_flow_signal import SkewFlowSignal
    s = SkewFlowSignal.__new__(SkewFlowSignal)
    s._key = "test-key"
    s._cache = {}
    s._hold = {}
    if last_good is not None:
        s._last_good = last_good
    s._fetch_trailing = lambda asset: fetch_rows
    s._save = lambda: None
    return s


class TestSkewCarry:
    def test_fetch_failure_with_recent_good_carries_the_held_direction(self):
        s = _skew(last_good={"BTC": {"ts": time.time() - 3600,
                                     "dir": 1.0, "z": -1.9}})
        d, fresh = s.seat_direction("BTC")
        assert d == 1.0 and fresh is True

    def test_fetch_failure_with_no_good_history_still_skips_the_seat(self):
        s = _skew()
        d, fresh = s.seat_direction("BTC")
        assert fresh is False

    def test_fetch_failure_with_STALE_good_history_skips_the_seat(self):
        from defense.skew_flow_signal import _STALE_DAYS
        old = time.time() - (_STALE_DAYS + 1) * 86400
        s = _skew(last_good={"BTC": {"ts": old, "dir": 1.0, "z": -1.9}})
        d, fresh = s.seat_direction("BTC")
        assert fresh is False, (
            "carry must be bounded by the SAME staleness rule as the data -- "
            "an unbounded carry is a dead feed holding a position forever")


# ---------------------------------------------------------------------------
# 4. inter-tick stop-out arms the cooldown (wiring pins)
# ---------------------------------------------------------------------------
class TestStopOutCooldown:
    def test_detection_runs_before_the_cooldown_check(self):
        src = (REPO / "main.py").read_text(encoding="utf-8-sig")
        det = src.find("went flat BETWEEN")
        chk = src.find('if (_cd_n > 0 and _cd_pre == 0')
        assert 0 < det < chk, (
            "the stop-out detection must ARM the cooldown before the "
            "cooldown check reads it")

    def test_prev_contracts_tracker_exists_at_end_of_tick(self):
        src = (REPO / "main.py").read_text(encoding="utf-8-sig")
        assert src.count("self._sleeve_prev_ct[_m_a] = int(") == 1


# ---------------------------------------------------------------------------
# 5. depth sentinel + volatility producer
# ---------------------------------------------------------------------------
def _frt(**kw):
    from execution.fast_risk_tick import FastRiskTick
    return FastRiskTick(shadow_mode=True, **kw)


class TestDepthSentinel:
    def test_fabricated_500k_floor_never_becomes_the_baseline(self):
        t = _frt()
        t.set_4h_anchor("ETH", 2000.0, 0.0, 400_000.0)
        assert t._baseline_depth.get("ETH") == 400_000.0
        t.set_4h_anchor("ETH", 2000.0, 0.0, 500_000.0)
        assert t._baseline_depth.get("ETH") == 400_000.0, (
            "the pipeline's fetch-failure floor (exactly 500_000.0) anchored "
            "as a baseline makes a normal book read as a 70% collapse")

    def test_a_real_depth_still_anchors(self):
        t = _frt()
        t.set_4h_anchor("BTC", 60000.0, 0.0, 4_770_000.0)
        assert t._baseline_depth.get("BTC") == 4_770_000.0


class TestVolatilityProducer:
    def _feed(self, t, asset, prices, t0=1_000_000.0, step=34.0):
        t._px_hist[asset] = [(t0 + i * step, p) for i, p in enumerate(prices)]
        return t0 + (len(prices) - 1) * step

    def test_below_15_samples_reads_zero(self, monkeypatch):
        t = _frt()
        end = self._feed(t, "BTC", [60000.0 + i for i in range(10)])
        monkeypatch.setattr(time, "time", lambda: end)
        assert t._realized_vol_30m("BTC") == 0.0

    def test_calm_vs_spike_ratio_exceeds_4x(self, monkeypatch):
        t = _frt()
        calm = [60000.0 * (1 + 0.0001 * ((-1) ** i)) for i in range(40)]
        end = self._feed(t, "BTC", calm)
        monkeypatch.setattr(time, "time", lambda: end)
        v_calm = t._realized_vol_30m("BTC")
        assert v_calm > 0
        spike = [60000.0 * (1 + 0.002 * ((-1) ** i)) for i in range(40)]
        t2 = _frt()
        end2 = self._feed(t2, "BTC", spike)
        monkeypatch.setattr(time, "time", lambda: end2)
        v_spike = t2._realized_vol_30m("BTC")
        assert v_spike > v_calm * 4.0, (v_calm, v_spike)

    def test_anchor_arms_the_baseline_from_the_internal_estimator(self, monkeypatch):
        t = _frt()
        calm = [60000.0 * (1 + 0.0001 * ((-1) ** i)) for i in range(40)]
        end = self._feed(t, "BTC", calm)
        monkeypatch.setattr(time, "time", lambda: end)
        t.set_4h_anchor("BTC", 60000.0, 0.0, 4_000_000.0)
        assert t._baseline_volatility.get("BTC", 0.0) > 0.0, (
            "volatility_30m has no external producer; without the internal "
            "estimator the operator-armed 4x REDUCE_50 is structurally inert")

    def test_external_producer_takes_precedence(self):
        t = _frt()
        t.set_4h_anchor("BTC", 60000.0, 0.123, 4_000_000.0)
        assert t._baseline_volatility.get("BTC") == 0.123
