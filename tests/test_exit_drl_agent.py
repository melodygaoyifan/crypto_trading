"""
Smoke tests for agents/exit_drl_agent.py — runtime inference + SHADOW logging.

Coverage:
- DISABLED mode → predict() returns None.
- SHADOW mode without checkpoints → asset status MISSING, predict returns None.
- SHADOW mode with valid checkpoint → predict returns prediction + writes JSONL.
- build_state row matches generate_expert_trajectories.py shape.
- Zero-direction position short-circuits to None (no spurious inference).
- Shadow log contains the right keys + roundtrips through JSON.
"""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agents.exit_drl_agent import (
    ACTION_DIM, STATE_DIM,
    ExitAction, ExitDRLAgent, ExitDRLMode,
    reset_singleton,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_singleton()
    yield
    reset_singleton()


def _make_dummy_checkpoint(models_dir: Path, asset: str) -> None:
    """Create a minimal valid checkpoint that ExitDRLAgent can load."""
    from training.exit_drl.train_exit_sac import DiscreteActor, DiscreteQCritic
    actor = DiscreteActor(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    critic = DiscreteQCritic(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    asset_dir = models_dir / asset
    asset_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "epoch": 0,
        "val_alignment": 0.5,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_names": ['HOLD', 'PARTIAL_EXIT', 'RELEASE_RUNNER', 'EXIT_ALL'],
    }, asset_dir / "exit_sac_best.pt")


def _make_market_data(price: float = 100.0) -> dict:
    md = {"current_price": price}
    for i in range(8):
        md[f"regime_proba_{i}"] = 0.125
    md["regime_confidence"] = 0.5
    md["regime"] = 0
    # Market + flow defaults — agent should tolerate zeros
    return md


def test_disabled_mode_returns_none(tmp_path):
    agent = ExitDRLAgent(
        models_dir=str(tmp_path / "models"),
        shadow_log_path=str(tmp_path / "shadow.jsonl"),
        mode=ExitDRLMode.DISABLED,
    )
    pos = {"direction": 1.0, "entry_price": 100.0}
    assert agent.predict("BTC", pos, _make_market_data(), bars_held=5) is None


def test_missing_checkpoint_marks_asset_missing(tmp_path):
    agent = ExitDRLAgent(
        models_dir=str(tmp_path / "no_models"),
        shadow_log_path=str(tmp_path / "shadow.jsonl"),
        mode=ExitDRLMode.SHADOW,
    )
    status = agent.status()
    assert status["mode"] == "SHADOW"
    assert all(v == "MISSING" for v in status["assets"].values())

    # predict() must return None when asset is MISSING
    pos = {"direction": 1.0, "entry_price": 100.0}
    assert agent.predict("BTC", pos, _make_market_data(), bars_held=5) is None


def test_shadow_mode_predicts_and_logs(tmp_path):
    models = tmp_path / "models"
    log = tmp_path / "shadow.jsonl"
    _make_dummy_checkpoint(models, "BTC")

    agent = ExitDRLAgent(
        models_dir=str(models),
        shadow_log_path=str(log),
        mode=ExitDRLMode.SHADOW,
        assets=["BTC"],
        extra_allowed_roots=[str(tmp_path)],
    )
    assert agent.status()["assets"]["BTC"] == "READY"

    pos = {
        "direction": 1.0,
        "entry_price": 100.0,
        "max_unrealized_pnl_pct": 0.01,
    }
    pred = agent.predict("BTC", pos, _make_market_data(price=101.5), bars_held=10)
    assert pred is not None
    assert isinstance(pred.action, ExitAction)
    assert len(pred.probabilities) == ACTION_DIM
    assert abs(sum(pred.probabilities) - 1.0) < 1e-5
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.inference_ms >= 0.0
    assert pred.state_summary["direction"] == 1.0
    assert pred.state_summary["bars_held"] == 10.0

    # Shadow log written
    assert log.exists()
    lines = log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["asset"] == "BTC"
    assert record["mode"] == "SHADOW"
    assert record["action"] in ('HOLD', 'PARTIAL_EXIT', 'RELEASE_RUNNER', 'EXIT_ALL')
    assert "probs" in record and len(record["probs"]) == ACTION_DIM
    assert "state" in record


def test_zero_direction_returns_none(tmp_path):
    models = tmp_path / "models"
    _make_dummy_checkpoint(models, "BTC")
    agent = ExitDRLAgent(
        models_dir=str(models),
        shadow_log_path=str(tmp_path / "shadow.jsonl"),
        mode=ExitDRLMode.SHADOW,
        assets=["BTC"],
        extra_allowed_roots=[str(tmp_path)],
    )
    # No active position → no inference, no log line
    flat_pos = {"direction": 0.0, "entry_price": 0.0}
    assert agent.predict("BTC", flat_pos, _make_market_data(), bars_held=0) is None
    assert agent.status()["predictions_logged"] == 0


def test_build_state_shape_matches_training_spec(tmp_path):
    pos = {"direction": -1.0, "entry_price": 100.0, "max_unrealized_pnl_pct": 0.02}
    state = ExitDRLAgent.build_state(pos, _make_market_data(price=98.0), bars_held=5)
    assert state.shape == (STATE_DIM,)
    assert state.dtype == np.float32
    # Position block sanity
    assert state[0] == -1.0                      # direction
    assert state[1] == pytest.approx(5.0 / 48.0) # bars_held normalized
    # short PnL: (98-100)/100 * -1 = +0.02
    assert state[2] == pytest.approx(0.02, abs=1e-6)


def test_dim_mismatch_checkpoint_disables_asset(tmp_path):
    """A future state-dim change should make old checkpoints fail loading, not crash."""
    asset_dir = tmp_path / "models" / "BTC"
    asset_dir.mkdir(parents=True)
    torch.save({
        "actor_state_dict": {},
        "critic_state_dict": {},
        "state_dim": STATE_DIM + 8,    # bogus
        "action_dim": ACTION_DIM,
    }, asset_dir / "exit_sac_best.pt")

    agent = ExitDRLAgent(
        models_dir=str(tmp_path / "models"),
        shadow_log_path=str(tmp_path / "shadow.jsonl"),
        mode=ExitDRLMode.SHADOW,
        assets=["BTC"],
        extra_allowed_roots=[str(tmp_path)],
    )
    assert agent.status()["assets"]["BTC"] == "DIM_MISMATCH"
    pos = {"direction": 1.0, "entry_price": 100.0}
    assert agent.predict("BTC", pos, _make_market_data(), bars_held=1) is None


def test_p81_peak_tracker_ratchets_across_calls(tmp_path):
    """P81 self-contained peak tracker: ratchets up across predict() calls
    on the same position; resets on direction change OR entry-price change.

    This is the regression guard for the half-wired-bug shape that caused
    Exit-SAC to never predict PARTIAL_EXIT in production (zero
    `max_unrealized_pnl_pct` writers anywhere in main.py)."""
    _make_dummy_checkpoint(tmp_path, "BTC")
    agent = ExitDRLAgent(
        models_dir=str(tmp_path),
        shadow_log_path=str(tmp_path / "shadow.jsonl"),
        mode=ExitDRLMode.SHADOW,
        assets=["BTC"],
        extra_allowed_roots=[str(tmp_path)],
    )

    pos = {"direction": 1.0, "entry_price": 100.0}  # NO max_unrealized_pnl_pct

    # Call 1 — pnl = 0% (current==entry).
    agent.predict("BTC", pos, _make_market_data(price=100.0), bars_held=1)
    assert agent._peak_unrealized_per_asset["BTC"] == pytest.approx(0.0)

    # Call 2 — pnl = +5%, peak ratchets up.
    agent.predict("BTC", pos, _make_market_data(price=105.0), bars_held=2)
    assert agent._peak_unrealized_per_asset["BTC"] == pytest.approx(0.05)

    # Call 3 — pnl drops to +2%, peak STAYS at +5% (no ratchet down).
    agent.predict("BTC", pos, _make_market_data(price=102.0), bars_held=3)
    assert agent._peak_unrealized_per_asset["BTC"] == pytest.approx(0.05)

    # Call 4 — peak goes higher to +8%, ratchets up again.
    agent.predict("BTC", pos, _make_market_data(price=108.0), bars_held=4)
    assert agent._peak_unrealized_per_asset["BTC"] == pytest.approx(0.08)

    # Call 5 — direction flips (close + reopen short). Peak resets.
    pos2 = {"direction": -1.0, "entry_price": 108.0}
    agent.predict("BTC", pos2, _make_market_data(price=108.0), bars_held=1)
    assert agent._peak_unrealized_per_asset["BTC"] == pytest.approx(0.0)

    # Call 6 — different entry_price (new long position). Peak resets.
    pos3 = {"direction": 1.0, "entry_price": 110.0}
    agent.predict("BTC", pos3, _make_market_data(price=112.0), bars_held=1)
    assert agent._peak_unrealized_per_asset["BTC"] == pytest.approx(2.0 / 110.0)

    # Call 7 — direction == 0 (position closed). Tracker entry cleared.
    pos_empty = {"direction": 0.0, "entry_price": 0.0}
    agent.predict("BTC", pos_empty, _make_market_data(price=110.0), bars_held=0)
    assert "BTC" not in agent._peak_unrealized_per_asset
    assert "BTC" not in agent._position_id_per_asset


def test_p81_caller_supplied_peak_takes_max(tmp_path):
    """If the caller already tracks `max_unrealized_pnl_pct` (e.g. via a
    different mechanism), the agent honors the higher of (caller, ours)."""
    _make_dummy_checkpoint(tmp_path, "BTC")
    agent = ExitDRLAgent(
        models_dir=str(tmp_path),
        shadow_log_path=str(tmp_path / "shadow.jsonl"),
        mode=ExitDRLMode.SHADOW,
        assets=["BTC"],
        extra_allowed_roots=[str(tmp_path)],
    )

    # Caller-supplied peak HIGHER than what the tracker would see — use
    # caller's. (Caller might be tracking via tick-level price, not
    # bar-level.)
    pos_high = {"direction": 1.0, "entry_price": 100.0,
                "max_unrealized_pnl_pct": 0.20}
    agent.predict("BTC", pos_high, _make_market_data(price=105.0), bars_held=1)
    assert pos_high["max_unrealized_pnl_pct"] == pytest.approx(0.20)

    # Caller-supplied peak LOWER than tracker (caller forgot to update).
    # Use tracker's.
    pos_low = {"direction": 1.0, "entry_price": 100.0,
               "max_unrealized_pnl_pct": 0.01}
    agent.predict("BTC", pos_low, _make_market_data(price=110.0), bars_held=2)
    assert pos_low["max_unrealized_pnl_pct"] == pytest.approx(0.10)


def test_real_btc_checkpoint_loads_if_present():
    """If the real checkpoint exists from a real training run, it should load cleanly."""
    real_path = Path("models/exit_drl/BTC/exit_sac_best.pt")
    if not real_path.exists():
        pytest.skip("No real BTC checkpoint")
    agent = ExitDRLAgent(
        models_dir="models/exit_drl",
        shadow_log_path="data/exit_drl_shadow.jsonl",
        mode=ExitDRLMode.SHADOW,
        assets=["BTC"],
    )
    assert agent.status()["assets"]["BTC"] == "READY"
    pos = {"direction": 1.0, "entry_price": 50000.0, "max_unrealized_pnl_pct": 0.005}
    md = {"current_price": 50250.0}
    for i in range(8):
        md[f"regime_proba_{i}"] = 0.125
    md["regime_confidence"] = 0.6
    md["regime"] = 2
    pred = agent.predict("BTC", pos, md, bars_held=8)
    assert pred is not None
