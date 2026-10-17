"""[P1R-7] Netting regression — one asset, one tick, one target, one order.

Phase 0 counted 18 same-4H-bucket opposite-direction order pairs, none after
2026-08-28. Traced fill by fill (reports/HMATS_PHASE1R.md R-7): none was two
intents of ONE tick netted wrongly — they were restart ticks on deploy days
(2026-08-16/18/19/22/25/26; run_live ticks immediately on start, P154) and
one watchdog urgent exit landing in the same bucket as a 4H add (08-28). They
stopped with the deploy freeze plus P416/P420's persisted resize/flip streaks,
not with a netting change. These tests pin the property that makes intra-tick
netting structurally unnecessary, so a future second order path per tick
fails loudly instead of silently re-creating the pattern.
"""
import asyncio
import inspect
import types

import pytest

from exchange.coinbase_sleeve import CoinbaseSleeve


@pytest.fixture(autouse=True)
def _private(tmp_path, monkeypatch):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))


class _Rec:
    def __init__(self):
        self.calls = []

    async def __call__(self, asset, target, **kw):
        self.calls.append((asset, target))
        return {"status": "OK", "asset": asset}


def _sleeve(cur):
    s = object.__new__(CoinbaseSleeve)
    s._flip_persist_ticks = 0
    s._flip_pending = {}
    s._reconcile_ok = True
    s._cur = cur
    s.reconcile_positions = lambda: {}
    s.signed_contracts = lambda a: s._cur
    s.execute_target = _Rec()
    return s


class TestOneTargetOneOrder:
    @pytest.mark.parametrize("cur,d", [(+1, -1.0), (0, 1.0), (+1, 0.0), (-1, 1.0)])
    def test_manage_issues_at_most_one_execute(self, cur, d):
        s = _sleeve(cur)
        asyncio.run(s.manage_to_signal("BTC", d))
        assert len(s.execute_target.calls) <= 1

    def test_a_flip_is_one_net_order_not_close_then_open(self):
        s = _sleeve(+1)
        asyncio.run(s.manage_to_signal("BTC", -1.0))
        assert len(s.execute_target.calls) == 1 and s.execute_target.calls[0][1] < 0

    def test_execute_target_sizes_one_order_from_the_net_delta(self):
        src = inspect.getsource(CoinbaseSleeve.execute_target)
        assert "delta = int(round(target_signed_contracts - cur))" in src


class TestDriverCallsManageOncePerAssetPerTick:
    def test_single_manage_call_site_inside_the_asset_loop(self):
        import main as m
        src = inspect.getsource(m.HMATSProductionRunner.run_live)
        assert src.count("await _sl.manage_to_signal(") == 1
        loop = src.rindex("for _m_a in", 0, src.index("await _sl.manage_to_signal("))
        assert loop > 0


class TestRiskExitBeatsAnAddInTheSameTick:
    @pytest.mark.parametrize("veto", ["[P0_SAFETY] Kill switch active",
                                      "[STRATEGY_SUSPENDED] existence fuse"])
    def test_flatten_intended_veto_overrides_a_long_signal(self, veto):
        import main as m
        it = types.SimpleNamespace(direction=1.0, target_exposure=0.25,
                                   veto_active=True, veto_reason=veto)
        tgt, why = m.sleeve_direction_from_intent(it, 1.0)
        assert tgt == 0.0, why

    def test_watchdog_flatten_between_ticks_arms_reentry_cooldown(self):
        """P418: a position that went flat between ticks (watchdog / stop)
        records a flatten, so the next 4H tick cannot immediately re-add."""
        import main as m
        src = inspect.getsource(m.HMATSProductionRunner.run_live)
        assert "went flat BETWEEN " in src and "re-entry cooldown armed" in src
