from types import SimpleNamespace

import pytest

from main import HMATSProductionRunner, RunMode


def _make_runner(*, mode=RunMode.PAPER, dry_run=False, paper_positions=None):
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner.config = SimpleNamespace(mode=mode)
    runner.account_sync = SimpleNamespace(dry_run=True) if dry_run else None
    runner._paper_positions = paper_positions or {}
    return runner


def test_effective_position_state_prefers_restored_paper_position_in_paper_mode():
    runner = _make_runner(
        paper_positions={
            "SOL": {
                "exposure": 0.024046343826126266,
                "direction": -1.0,
                "tranche": 1,
                "notional": 84.95562073580894,
            }
        }
    )

    state = runner._get_effective_position_state(
        "SOL",
        {
            "current_exposure": 0.0,
            "position_direction": 0,
            "current_tranche": 0,
        },
    )

    assert state["current_exposure"] == pytest.approx(-0.024046343826126266)
    assert state["direction"] == -1
    assert state["tranche"] == 1
    assert state["current_exposure"] != 0.0


def test_effective_position_state_prefers_restored_position_in_dry_run_mode():
    runner = _make_runner(
        mode=RunMode.LIVE,
        dry_run=True,
        paper_positions={
            "SOL": {
                "exposure": 0.024046343826126266,
                "direction": -1.0,
                "tranche": 2,
                "notional": 84.95562073580894,
            }
        },
    )

    state = runner._get_effective_position_state(
        "SOL",
        {
            "current_exposure": 0.0,
            "position_direction": 0,
            "current_tranche": 0,
        },
    )

    assert state["current_exposure"] == pytest.approx(-0.024046343826126266)
    assert state["direction"] == -1
    assert state["tranche"] == 2


def test_effective_position_state_ignores_inactive_paper_position():
    runner = _make_runner(
        paper_positions={
            "SOL": {
                "exposure": 0.024046343826126266,
                "direction": -1.0,
                "tranche": 1,
                "notional": 0.0,
            }
        }
    )

    state = runner._get_effective_position_state(
        "SOL",
        {
            "current_exposure": 0.0,
            "position_direction": 0,
            "current_tranche": 0,
        },
    )

    assert state["current_exposure"] == pytest.approx(0.0)
    assert state["direction"] == 0
    assert state["tranche"] == 0


def test_reconcile_restored_paper_positions_uses_notional_truth_when_exposure_drifted():
    runner = _make_runner(
        paper_positions={
            "BTC": {
                "exposure": 0.06216585776097214,
                "original_entry_exposure": 0.06216585776097214,
                "direction": 1.0,
                "notional": 258.959106416676,
            }
        }
    )

    reconciled, corrected = runner._reconcile_restored_paper_positions(
        runner._paper_positions,
        9985.234322564444,
    )

    expected_exposure = 258.959106416676 / 9985.234322564444
    assert corrected == 1
    assert reconciled["BTC"]["exposure"] == pytest.approx(expected_exposure)
    assert reconciled["BTC"]["original_entry_exposure"] == pytest.approx(expected_exposure)


def test_reconcile_restored_paper_positions_preserves_distinct_original_entry_exposure():
    runner = _make_runner(
        paper_positions={
            "SOL": {
                "exposure": 0.08,
                "original_entry_exposure": 0.10,
                "direction": -1.0,
                "notional": 500.0,
            }
        }
    )

    reconciled, corrected = runner._reconcile_restored_paper_positions(
        runner._paper_positions,
        10000.0,
    )

    assert corrected == 1
    assert reconciled["SOL"]["exposure"] == pytest.approx(0.05)
    assert reconciled["SOL"]["original_entry_exposure"] == pytest.approx(0.10)


def test_normalize_existing_hold_intent_preserves_current_exposure():
    runner = _make_runner()
    intent = SimpleNamespace(
        tranche_action="HOLD",
        direction=1.0,
        target_exposure=0.0110,
        is_addon=True,
    )

    applied = runner._normalize_existing_hold_intent(
        "BTC",
        intent,
        {"current_exposure": 0.0259},
    )

    assert applied is True
    assert intent.target_exposure == pytest.approx(0.0259)
    assert intent.is_addon is False
    assert intent.hold_existing_position_noop is True


def test_normalize_existing_hold_intent_ignores_opposite_or_flat_cases():
    runner = _make_runner()
    hold_intent = SimpleNamespace(
        tranche_action="HOLD",
        direction=1.0,
        target_exposure=0.0110,
        is_addon=True,
    )

    assert runner._normalize_existing_hold_intent(
        "BTC",
        hold_intent,
        {"current_exposure": -0.0259},
    ) is False
    assert hold_intent.target_exposure == pytest.approx(0.0110)
    assert hold_intent.is_addon is True

    flat_intent = SimpleNamespace(
        tranche_action="HOLD",
        direction=1.0,
        target_exposure=0.0110,
        is_addon=True,
    )
    assert runner._normalize_existing_hold_intent(
        "BTC",
        flat_intent,
        {"current_exposure": 0.0},
    ) is False
