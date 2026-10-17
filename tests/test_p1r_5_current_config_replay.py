"""[P1R-5] The current-config replay: pre-registered rule, verdict logic,
sleeve mechanics (flip/resize persistence, re-entry cooldown), costs."""
import inspect

import pytest

from analytics.backtest import current_config_replay as R


class TestRule:
    def test_rule_is_pre_registered_in_the_module(self):
        t = R.RULE_TEXT
        assert "NO_EDGE_AT_CURRENT_COST" in t and "EDGE_CONFIRMED_BACKTEST" in t
        assert "UNCERTAIN" in t and "HYPOTHESIS_SUPPORTED" in t and "0.3-0.5" in t

    def test_counterfactuals_never_change_parameters(self):
        """The replay READS the live profile and writes only its report."""
        src = inspect.getsource(R)
        assert src.count('open(a.out, "w"') == 1
        assert src.count('"w"') == 1
        assert "live_high_risk.json" in inspect.getsource(R.live_cfg)


def _rt(a, g, c):
    return dict(asset=a, gross_alpha_bps=g, all_in_cost_bps=c, net_bps=g - c)


class TestVerdict:
    def test_all_below_cost_is_no_edge(self):
        rts = [_rt(a, -10, 20) for a in R.ASSETS for _ in range(12)]
        overall, per = R.verdict(rts, 365)
        assert overall == "NO_EDGE_AT_CURRENT_COST"
        assert all(v["verdict"] == "NO_EDGE_ASSET" for v in per.values())

    def test_confirmed_needs_ci_lower_bound_above_zero(self):
        rts = [_rt("BTC", 100 + i, 20) for i in range(30)] + [_rt("ETH", -5, 20)] * 12 + [_rt("SOL", -5, 20)] * 12
        overall, per = R.verdict(rts, 365)
        assert per["BTC"]["verdict"] == "EDGE_CONFIRMED_BACKTEST"
        assert overall.startswith("EDGE_CONFIRMED_BACKTEST")

    def test_wide_ci_is_uncertain_with_weeks(self):
        rts = [_rt("BTC", g, 20) for g in (-600, 900, 50, -400, 1200, 60, -500, 800)] + \
              [_rt("ETH", -5, 20)] * 12 + [_rt("SOL", -5, 20)] * 12
        overall, per = R.verdict(rts, 365)
        assert per["BTC"]["verdict"] == "UNCERTAIN" and overall == "UNCERTAIN"


class _Sched:
    contract_size = {"BTC": 0.01, "ETH": 0.1, "SOL": 5.0}

    def fee_usd(self, a, n, px, maker, vol=None):
        return abs(n) * (0.11 + 0.01 * px * (9.5e-4 if maker else 1e-3))


def _cfg():
    return dict(fractions={"BTC": 0.2, "ETH": 0.15, "SOL": 0.095}, flip_persist=2,
                resize_persist=2, cooldown=2)


def _fund():
    return {a: {"daily": {}, "cde": [], "cde_first": float("inf"), "bin_last": None} for a in R.ASSETS}


def _run(btc_dirs, prices=None, **kw):
    t0 = 1_800_000_000.0
    prices = prices or [80000.0] * len(btc_dirs)
    dec = {"BTC": [(t0 + i * R.BAR_S, prices[i], d, "skew_contra", "peace")
                   for i, d in enumerate(btc_dirs)], "ETH": [], "SOL": []}
    return R.simulate(dec, None, _fund(), _Sched(), _cfg(), 0.5,
                      {"BTC": 1.0, "ETH": 1.0, "SOL": 1.0}, 10_800.0, **kw)


class TestSleeveMechanics:
    def test_single_tick_flip_is_deferred(self):
        rts, st = _run([1, 1, -1, 1, 1])
        assert st["open_positions"]["BTC"] > 0 and rts == []

    def test_persistent_flip_executes_as_one_order(self):
        rts, st = _run([1, 1, -1, -1, -1])
        assert st["open_positions"]["BTC"] < 0 and len(rts) == 1 and st["n_fills"] == 2

    def test_reentry_cooldown_blocks_two_ticks(self):
        # flatten at tick 2; ticks 3 and 4 are the two blocked ticks; tick 5 enters
        rts, st = _run([1, 0, 1, 1])
        assert st["n_fills"] == 2 and "BTC" not in st["open_positions"]
        rts2, st2 = _run([1, 0, 1, 1, 1])
        assert st2["n_fills"] == 3 and st2["open_positions"].get("BTC", 0) > 0

    def test_flip_off_counterfactual_only_closes(self):
        rts, st = _run([1, 1, -1, -1, -1], flip_off=True)
        assert "BTC" not in st["open_positions"] and len(rts) == 1

    def test_gross_is_mid_to_mid_and_costs_are_separate(self):
        rts, st = _run([1, 1, 0, 0], prices=[80000.0, 81000.0, 82000.0, 82000.0])
        r = rts[0]
        ct = r["entry_notional_usd"] / (0.01 * 80000.0)
        assert r["gross_usd"] == pytest.approx(ct * 0.01 * 2000.0)
        assert r["fee_bps"] > 0 and r["spread_bps"] > 0
        assert r["net_bps"] == pytest.approx(r["gross_alpha_bps"] - r["all_in_cost_bps"])


class TestImportsTheLiveChain:
    def test_seats_and_book_are_imported_not_restated(self):
        src = inspect.getsource(R)
        for name in ("etf_seat_decision", "skew_seat_decision", "book_target",
                     "regime_label", "causal_funding_z", "donchian_labels",
                     "etf_flow_zscore_direction", "band_direction", "zscore_trailing",
                     "REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP", "skew_contra_alpha_bps"):
            assert name in src

    def test_sleeve_parameters_come_from_the_live_profile(self):
        c = R.live_cfg()
        assert c["flip_persist"] >= 1 and c["cooldown"] >= 0
        assert set(c["fractions"]) == set(R.ASSETS)
