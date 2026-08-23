"""[P382] The Rung-0 edge probe prices its cost bar on the venue that trades.

Until P382 `training/scripts/edge_probe.py` charged `COST_RT_BPS = 6.0` —
the refuted 3bps/side percentage model (P315/P334) — so a 16h candidate
cleared with +7..+20bps "after cost" that the measured CDE fixed fee alone
(19.7-29.0bps RT) would have eaten. The probe now prices each asset at
max(6.0, 2 x the measured CDE taker leg) from `core.cde_fees`, the same bar
the other two P166 gates use (compute_shadow_ic / agent_ic_review), and
says which model it used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import training.scripts.edge_probe as ep  # noqa: E402
from core.cde_fees import CDE_FEE_BPS  # noqa: E402


@pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
def test_the_bar_is_at_least_two_measured_taker_legs(asset):
    bps, src = ep.cost_rt_bps_for(asset)
    assert src == "cde_fees"
    assert bps >= 2.0 * float(CDE_FEE_BPS[asset]["taker"]) - 1e-9
    assert bps >= ep.COST_RT_BPS


def test_the_measured_bar_is_materially_above_the_refuted_model():
    for a in ("BTC", "ETH", "SOL"):
        bps, _ = ep.cost_rt_bps_for(a)
        assert bps > 3.0 * ep.COST_RT_BPS, (a, bps)


def test_the_override_is_honoured_and_labelled(monkeypatch):
    monkeypatch.setattr(ep, "COST_RT_OVERRIDE", 10.0)
    assert ep.cost_rt_bps_for("BTC") == (10.0, "override")


def test_an_unreadable_calibration_falls_back_to_the_floor_and_says_so(
        monkeypatch, capsys):
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "core.cde_fees":
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", _blocked)
    bps, src = ep.cost_rt_bps_for("BTC")
    assert bps == ep.COST_RT_BPS and src == "fallback_refuted_model"
    assert "REFUTED" in capsys.readouterr().err


def test_required_ic_scales_with_the_bar():
    # the same sigma, a 4x bar -> a 4x required IC: the probe cannot clear
    # a 16h candidate at the measured fee by re-using the refuted bar
    assert ep.required_ic(24.0, 500.0) == pytest.approx(4 * ep.required_ic(6.0, 500.0))
