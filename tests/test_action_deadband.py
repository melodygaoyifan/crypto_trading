"""[CHURN-TIER] action_deadband: sub-threshold position ADJUSTMENTS hold, flips
and full exits always pass.

Motivated by the official_p221b cost diagnosis: SOL's TQC changed position at
~96% of decision points and paid $40-72K/fold in trade costs; fold_2 missed
the B&H bar by $5.5K while paying $51K of churn. The deadband is the lever
that lets forecast wobble hold a position without a trade.

Two-layer pattern (source gates + gymnasium-gated behavioral), same as
tests/test_decision_interval.py.
"""

import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINER = REPO_ROOT / "training" / "train_drl_full.py"


# ---------------------------------------------------------------------------
# Source gates (P152 shape: an unwired knob is not a knob)
# ---------------------------------------------------------------------------

def _src():
    return io.open(TRAINER, encoding="utf-8").read()


def test_env_declares_the_param():
    assert "action_deadband: float = 0.0," in _src(), (
        "default must be 0.0 — a nonzero default silently changes every "
        "existing training run's env contract"
    )


def test_optuna_churn_tier_is_wired():
    src = _src()
    assert "'action_deadband', 0.0, 0.5" in src
    assert '"--optuna-churn"' in src
    assert "churn_tier=args.optuna_churn," in src
    assert "venue=args.venue," in src, (
        "the churn tier must price trials at the venue that will trade the "
        "model — tuning a deadband at the wrong venue's fees selects the "
        "wrong deadband"
    )


def test_churn_study_is_a_distinct_optuna_study():
    assert '"_churn" if churn_tier else ""' in _src(), (
        "mixing the 4-dim churn space into the 12-dim base study breaks "
        "TPE and crash-recovery resume"
    )


# ---------------------------------------------------------------------------
# Behavioral — needs the training stack
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def T():
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("sb3_contrib")
    sys.path.insert(0, str(REPO_ROOT))
    import training.train_drl_full as mod
    return mod


@pytest.fixture
def synth_df():
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(11)
    n = 300
    close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
        "close": close,
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "regime": rng.integers(0, 6, n),
    })


def _env(T, df, **kw):
    return T.TradingEnvFull(df=df, feature_cols=["f1", "f2"],
                            reward_mode="classic", **kw)


def _step(env, a):
    import numpy as np
    return env.step(np.array([a], dtype=np.float32))


def test_deadband_zero_is_byte_identical_to_history(T, synth_df):
    """Default off: every adjustment lands, however small."""
    env = _env(T, synth_df)  # deadband defaults 0.0
    env.reset()
    _step(env, 1.0)
    p1 = env.position
    _step(env, 0.98)
    assert env.position != p1, "a tiny adjustment was suppressed at deadband 0"


def test_small_adjustment_holds_no_trade_no_cost(T, synth_df):
    env = _env(T, synth_df, action_deadband=0.3)
    env.reset()
    _step(env, 1.0)
    p1 = env.position
    trades_before = env._trade_count
    _step(env, 0.85)   # |change| = 0.15 x max_position < 0.3 -> hold
    assert env.position == p1, "sub-deadband adjustment moved the position"
    assert env._trade_count == trades_before, "a held position booked a trade"
    assert env._last_trade_cost_usd == 0.0


def test_large_adjustment_passes(T, synth_df):
    env = _env(T, synth_df, action_deadband=0.3)
    env.reset()
    _step(env, 1.0)
    _step(env, 0.5)    # |change| = 0.5 >= 0.3 -> executes
    assert env.position == pytest.approx(0.5 * env.max_position)


def test_flip_always_passes(T, synth_df):
    """A full flip is |change| ~ 2 x max_position — no legal deadband may
    suppress it. This is the P195 asymmetry: suppress wobble, never an exit."""
    env = _env(T, synth_df, action_deadband=0.5)
    env.reset()
    _step(env, 1.0)
    _step(env, -1.0)
    assert env.position < 0, "deadband suppressed a full flip"


def test_full_exit_passes(T, synth_df):
    env = _env(T, synth_df, action_deadband=0.5)
    env.reset()
    _step(env, 1.0)
    _step(env, 0.0)    # |change| = 1.0 x max_position >= 0.5 -> exit executes
    assert env.position == 0.0, "deadband trapped the position (P195 shape)"


def test_deadband_clamped_below_one(T, synth_df):
    """A deadband >= 1.0 x max_position would make exits from full positions
    unreachable — the ctor must clamp, not trust the caller."""
    env = _env(T, synth_df, action_deadband=5.0)
    env.reset()
    _step(env, 1.0)
    _step(env, 0.0)
    assert env.position == 0.0, (
        "an unclamped deadband made a full exit unreachable"
    )


def test_composes_with_decision_interval(T, synth_df):
    """On hold bars the override already equals the current position, so the
    deadband must be a no-op there — and on decision bars it applies."""
    env = _env(T, synth_df, action_deadband=0.3, decision_interval=4)
    env.reset()
    _step(env, 1.0)            # decision bar: long
    p1 = env.position
    for a in (0.9, -1.0, 0.2):  # hold bars: whatever the action, hold
        _step(env, a)
        assert env.position == p1
    _step(env, 0.9)            # decision bar: |change|=0.1 < 0.3 -> still held
    assert env.position == p1
    for _ in range(3):
        _step(env, 0.0)
    _step(env, -1.0)           # decision bar: flip passes
    assert env.position < 0


def test_deadband_cuts_cost_on_wobbling_policy(T, synth_df):
    """The economic claim itself: a wobbling forecast (0.8/1.0 alternation)
    pays every step at deadband 0 and ~nothing at deadband 0.3."""
    def run(db):
        env = _env(T, synth_df, action_deadband=db)
        env.reset()
        for i in range(60):
            _, _, term, trunc, _ = _step(env, 1.0 if i % 2 == 0 else 0.8)
            if term or trunc:
                break
        return env.cumulative_trade_cost
    assert run(0.3) < run(0.0) / 3
