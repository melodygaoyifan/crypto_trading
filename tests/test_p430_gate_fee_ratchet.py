"""[P430] The P1-1 venue fee schedule may raise the alpha-gate fee, never
lower it below the P334 table it replaced (SOL 14.51 -> 12.12bps/leg and a
small ETH drop were gate LOOSENINGS on a book the IC audit measured at no
edge)."""
import inspect

import pytest

import main
from core.cde_fees import cde_fee_bps, venue_fee_bps

PX = {"BTC": 81_500.0, "ETH": 2_400.0, "SOL": 104.0}


@pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
@pytest.mark.parametrize("scale", [0.5, 0.8, 1.0, 1.5, 3.0])
def test_ratchet_never_below_table_or_schedule(asset, scale):
    px = PX[asset] * scale
    mk = venue_fee_bps(asset, px, is_maker=True, rolling_30d_volume_usd=0.0)
    tk = venue_fee_bps(asset, px, is_maker=False, rolling_30d_volume_usd=0.0)
    rmk, rtk = main.gate_fee_ratchet(asset, px, mk, tk)
    for got, sched, is_mk in ((rmk, mk, True), (rtk, tk, False)):
        tbl = cde_fee_bps(asset, px, is_maker=is_mk)[0]
        assert got[0] >= tbl - 1e-12
        assert got[0] >= sched[0] - 1e-12
        assert got[0] == pytest.approx(max(tbl, sched[0]))


def test_sol_is_floored_at_the_table():
    px = PX["SOL"]
    tk = venue_fee_bps("SOL", px, is_maker=False, rolling_30d_volume_usd=0.0)
    assert tk[0] < cde_fee_bps("SOL", px, is_maker=False)[0]   # the premise
    _, rtk = main.gate_fee_ratchet("SOL", px, tk, tk)
    assert rtk[0] == pytest.approx(cde_fee_bps("SOL", px, is_maker=False)[0])
    assert rtk[1].startswith("ratchet_floor:")


def test_btc_tightening_passes_through():
    px = PX["BTC"]
    tk = venue_fee_bps("BTC", px, is_maker=False, rolling_30d_volume_usd=0.0)
    assert tk[0] > cde_fee_bps("BTC", px, is_maker=False)[0]
    _, rtk = main.gate_fee_ratchet("BTC", px, tk, tk)
    assert rtk == tk


def test_none_legs_pass_through():
    assert main.gate_fee_ratchet("SOL", 104.0, None, None) == (None, None)


def test_unpriceable_table_leaves_schedule():
    leg = (12.0, "venue_fee_schedule_phase0")
    assert main.gate_fee_ratchet("SOL", 0.0, leg, leg) == (leg, leg)


def test_gate_block_calls_the_ratchet_before_using_the_fee():
    src = inspect.getsource(main)
    i_call = src.index("_mk, _tk = gate_fee_ratchet(asset, _px_fee, _mk, _tk)")
    i_use = src.index("maker_fee_bps = max(maker_fee_bps, _mk[0])")
    i_lookup = src.index("_tk = _cfb(asset, _px_fee, is_maker=False)")
    assert i_lookup < i_call < i_use
