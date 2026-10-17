"""[P1R-2] Instrumentation the Phase 0 audit was blind without: intent class,
requested order type / post-only, the venue's maker/taker verdict, tier
context per fill, the watchdog REDUCE cause, hourly funding accrual, and
90-day log retention."""
import asyncio
import inspect
import json
import types

import pytest

from exchange.coinbase_sleeve import (ORDER_TYPE_BY_LIQUIDITY, CoinbaseSleeve,
                                      classify_intent_type)


class TestIntentType:
    @pytest.mark.parametrize("p,t,want", [
        (0, 2, "NEW_ENTRY"), (0, -1, "NEW_ENTRY"),
        (2, 4, "ESCALATE"), (-1, -3, "ESCALATE"),
        (4, 2, "PARTIAL_EXIT"), (-2, -1, "PARTIAL_EXIT"),
        (2, 0, "FULL_EXIT"), (-2, 0, "FULL_EXIT"),
        (1, -1, "FLIP"), (-2, 3, "FLIP"),
        (2, 2, "REBALANCE"), (0, 0, "REBALANCE"),
    ])
    def test_truth_table(self, p, t, want):
        assert classify_intent_type(p, t) == want

    def test_restart_cause_wins(self):
        assert classify_intent_type(0, 2, "restart_reconcile") == "RESTART_RECONCILE"

    @pytest.mark.parametrize("p,t", [(None, 1), (1, None), ("x", 1), (float("nan"), 1)])
    def test_unknown_is_none_never_a_class(self, p, t):
        assert classify_intent_type(p, t) is None


class TestOrderTypeMap:
    def test_every_recorder_label_is_mapped(self):
        import exchange.coinbase_sleeve as m
        src = inspect.getsource(m)
        for lab in ("maker", "taker_cross", "direct", "market_urgent"):
            assert f'"{lab}"' in src and lab in ORDER_TYPE_BY_LIQUIDITY
        assert ORDER_TYPE_BY_LIQUIDITY["maker"] == ("LIMIT", True)
        assert ORDER_TYPE_BY_LIQUIDITY["market_urgent"] == ("MARKET", False)
        assert not any(v[1] for k, v in ORDER_TYPE_BY_LIQUIDITY.items() if k != "maker")


def _sleeve():
    s = CoinbaseSleeve.__new__(CoinbaseSleeve)
    s._last_positions = {"BTC": {"signed_contracts": -2}}
    s._reconcile_ok = True
    return s


def _last(tmp_path):
    rows = (tmp_path / "fill_quality.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return json.loads(rows[-1])


class TestRecorder:
    def test_new_fields_present_and_classified(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        s = _sleeve()
        s.set_order_context("BTC", decider="fast_risk_watchdog", cause="risk_exit:fast_risk_reduce",
                            target_contracts=-1)
        s._record_fill_quality("BTC", "oid", "BUY", 1, "market_urgent", True,
                               decision_mid=81000.0,
                               order_payload={"status": "FILLED", "average_filled_price": "81010",
                                              "filled_size": "1", "_fill_liquidity": "TAKER"})
        r = _last(tmp_path)
        assert r["intent_type"] == "PARTIAL_EXIT"
        assert r["order_type"] == "MARKET" and r["post_only_requested"] is False
        assert r["liquidity_indicator"] == "TAKER"
        assert r["cause"] == "risk_exit:fast_risk_reduce"
        assert "rolling_30d_volume_usd" in r and "fee_tier_taker_bps" in r

    def test_unfilled_has_no_venue_verdict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        s = _sleeve()
        s._record_fill_quality("BTC", "oid", "SELL", 1, "maker", False,
                               order_payload={"status": "OPEN", "_fill_liquidity": "MAKER"})
        r = _last(tmp_path)
        assert r["status"] == "unresolved" and r["liquidity_indicator"] is None
        assert r["order_type"] == "LIMIT" and r["post_only_requested"] is True

    def test_execute_target_stamps_the_real_target(self):
        src = inspect.getsource(CoinbaseSleeve.execute_target)
        assert '_octx["target_contracts"] = int(round(target_signed_contracts))' in src
        assert "classify_intent_type(" in src


class TestWatchdogReduceCause:
    def test_reduce_branch_stamps_and_clears(self):
        import main as m
        calls = []

        class FakeSleeve:
            _reconcile_ok = True
            def reconcile_positions(self): pass
            def signed_contracts(self, a): return -2
            def set_order_context(self, a, **k): calls.append(("set", k.get("cause")))
            def clear_order_context(self, a): calls.append(("clear", None))
            async def execute_target(self, a, t, urgent=False):
                calls.append(("exec", t)); return {"status": "OK"}
            def request_stop_followup(self, *a): pass

        st, _ = asyncio.run(m.sleeve_fast_risk_action(FakeSleeve(), "BTC", "REDUCE_50", True))
        assert calls == [("set", "risk_exit:fast_risk_reduce"), ("exec", -1), ("clear", None)]
        assert st == "REDUCED"

    def test_driver_records_direction_under_its_own_name(self):
        import main as m
        src = inspect.getsource(m.HMATSProductionRunner.run_live)
        assert "target_direction=_m_dir" in src
        assert "target_contracts=_m_dir" not in src


class TestFundingAccrual:
    def test_rows_and_sign(self):
        import main as m
        pos = {"BTC": {"signed_contracts": -2, "current_price": 80000.0},
               "ETH": {"signed_contracts": 0, "current_price": 2400.0}}
        rows = m.funding_accrual_rows(pos, {"BTC": 8e-5}, {"BTC": 0.01}.get, 7200, {})
        assert len(rows) == 1 and rows[0]["asset"] == "BTC"
        # short receives positive funding -> negative "paid"
        assert rows[0]["funding_est_usd"] == pytest.approx(-(1e-5) * 1600.0)
        assert "ESTIMATE" in rows[0]["source"]

    def test_once_per_hour_and_unknown_rate_is_none(self):
        import main as m
        pos = {"BTC": {"signed_contracts": 1, "current_price": 80000.0}}
        assert m.funding_accrual_rows(pos, {"BTC": 8e-5}, {"BTC": 0.01}.get, 3600, {"BTC": 3600}) == []
        r = m.funding_accrual_rows(pos, {}, {"BTC": 0.01}.get, 7200, {"BTC": 3600})
        assert r[0]["funding_est_usd"] is None

    def test_wired_in_run_live_loop(self):
        import main as m
        assert "self._accrue_funding_hourly()" in inspect.getsource(m.HMATSProductionRunner.run_live)


class TestLogRetention:
    def test_backup_count_covers_90_days(self):
        import main as m
        assert m._file_handler.backupCount >= 90
        assert m._file_handler.maxBytes == 10 * 1024 * 1024
