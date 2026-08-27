"""[P420] The ETF seat's persisted deadband hold is age-bounded and REPLAYED.

`deadband_hold` is a SEAT-FRESH reason, so a -1 persisted N days ago and
restored blindly could flatten a long BTC/ETH book today through the de-risk
path with no current signal behind it. The hold is now replayed from the full
completed flow history the feed already fetches (deterministic, restart-
invariant); if the replay cannot run the hold starts flat with a reason that
is NOT seat-fresh.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

from defense.etf_flow_shadow import (
    EtfFlowShadow, HOLD_MAX_AGE_DAYS, ZSCORE_WINDOW, ZSCORE_MIN_OBS,
    etf_flow_zscore_direction, replay_hold)


def _rows(flows, end_day=None):
    """Completed daily rows (oldest first) ending YESTERDAY, plus today's
    in-progress row (which the feed must ignore)."""
    today = (end_day or datetime.now(timezone.utc)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    out = []
    for i, f in enumerate(flows):
        day = today - timedelta(days=len(flows) - i)
        out.append({"timestamp": int(day.timestamp() * 1000), "flow_usd": f,
                    "price_usd": 100.0 + i})
    out.append({"timestamp": int(today.timestamp() * 1000),
                "flow_usd": 1e12, "price_usd": 999.0})   # in-progress day
    return out


def _persist(dirpath, hold, age_days, seat=None):
    d = dirpath / "strategy_shadow"
    d.mkdir(parents=True, exist_ok=True)
    (d / "etfflow_state.json").write_text(json.dumps({
        "last_direction": hold, "seat_state": seat or {},
        "saved_at": time.time() - age_days * 86400.0}), encoding="utf-8")


def _shadow(tmp_path, rows):
    s = EtfFlowShadow(data_dir=str(tmp_path))
    s._api_key = "test"
    s._fetch_rows = lambda a: rows     # type: ignore[assignment]
    return s


# 40 small alternating days, then a strong OUTFLOW spike, then in-band days:
# a continuous process ends holding -1.
FLOWS_SHORT = [((-1.0) ** i) * 1e6 for i in range(40)] + [-3e7, 2e5, -1e5, 1e5]


class TestReplay:
    def test_replay_hold_walks_the_band_rule(self):
        assert replay_hold(FLOWS_SHORT) == -1.0
        assert replay_hold([]) == 0.0 and replay_hold([1.0]) == 0.0
        # below min_obs everywhere -> never a claim
        assert replay_hold([1e6, -1e6] * 5) == 0.0

    def test_replay_matches_a_tick_by_tick_walk(self):
        prev = 0.0
        for i in range(1, len(FLOWS_SHORT)):
            hist = FLOWS_SHORT[max(0, i - 1 - ZSCORE_WINDOW):i - 1]
            prev = etf_flow_zscore_direction(FLOWS_SHORT[i - 1], 0.0, hist, prev)[0]
        assert replay_hold(FLOWS_SHORT) == prev

    def test_a_three_day_old_persisted_plus_one_is_replayed(self, tmp_path):
        _persist(tmp_path, {"BTC": 1.0}, age_days=3.0)
        s = _shadow(tmp_path, _rows(FLOWS_SHORT))
        assert s._last_direction == {}, "stale hold must not be restored as-is"
        assert "BTC" in s._replay_pending
        rec = s.record_tick("BTC")
        assert rec is not None
        assert rec["direction"] == -1.0 and rec["reason"] == "deadband_hold"
        assert rec["hold_source"] == "replayed"
        # and it is a live claim the seat may use (a continuous process held it)
        assert s.seat_direction("BTC") == (-1.0, True)

    def test_blind_restore_would_have_seated_the_wrong_sign(self, tmp_path):
        """The counterfactual that makes the fix matter: with the stale +1
        restored as-is the in-band newest day HOLDS +1 (the seat de-risks
        nothing, or worse), where the history says -1."""
        s = _shadow(tmp_path, _rows(FLOWS_SHORT))
        s._last_direction["BTC"] = 1.0        # what the old code did
        rec = s.record_tick("BTC")
        assert rec["direction"] == 1.0 and rec["reason"] == "deadband_hold"

    def test_fresh_persisted_hold_is_restored_as_is(self, tmp_path):
        _persist(tmp_path, {"BTC": 1.0}, age_days=0.3)
        s = _shadow(tmp_path, _rows(FLOWS_SHORT))
        assert s._last_direction == {"BTC": 1.0} and not s._replay_pending

    def test_stale_hold_with_no_history_starts_flat_and_not_seat_fresh(self, tmp_path):
        _persist(tmp_path, {"BTC": 1.0}, age_days=3.0)
        s = _shadow(tmp_path, None)
        rec = s.record_tick("BTC")
        assert rec["direction"] == 0.0 and rec["reason"] == "no_data"
        assert rec["hold_source"] == "restart_transient"
        assert s.seat_direction("BTC") == (0.0, False)

    def test_restart_transient_is_not_a_seat_fresh_reason(self):
        assert "restart_transient" not in EtfFlowShadow._SEAT_FRESH_REASONS
        assert "deadband_hold" in EtfFlowShadow._SEAT_FRESH_REASONS

    def test_saved_at_is_written_and_age_gates_the_restore(self, tmp_path):
        s = _shadow(tmp_path, _rows(FLOWS_SHORT))
        s._last_direction["ETH"] = -1.0
        s._save_state()
        st = json.loads((tmp_path / "strategy_shadow" / "etfflow_state.json")
                        .read_text(encoding="utf-8"))
        assert time.time() - st["saved_at"] < 60
        assert EtfFlowShadow(data_dir=str(tmp_path))._last_direction == {"ETH": -1.0}
        # age it past the bound by hand -> replay pending, hold not restored
        st["saved_at"] = time.time() - (HOLD_MAX_AGE_DAYS + 0.5) * 86400
        (tmp_path / "strategy_shadow" / "etfflow_state.json").write_text(
            json.dumps(st), encoding="utf-8")
        b = EtfFlowShadow(data_dir=str(tmp_path))
        assert b._last_direction == {} and b._replay_pending == {"ETH"}

    def test_an_unstamped_legacy_state_is_replayed_not_trusted(self, tmp_path):
        d = tmp_path / "strategy_shadow"
        d.mkdir(parents=True)
        (d / "etfflow_state.json").write_text(
            json.dumps({"last_direction": {"BTC": 1.0}}), encoding="utf-8")
        s = EtfFlowShadow(data_dir=str(tmp_path))
        assert s._last_direction == {} and "BTC" in s._replay_pending
