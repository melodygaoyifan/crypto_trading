"""[P200-LADDER] ridge_16h baseline: the supervised alternative the RL
candidate must beat.

The edge probe found the only robust after-cost signal in this feature set is
linear at the 16h horizon; the literature says supervised forecast-then-trade
routinely beats RL in crypto. So the ridge is now a P182 baseline — an RL
policy that cannot beat the ridge it shares features with is not promotable.
"""

import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _src():
    return io.open(REPO_ROOT / "training" / "train_drl_full.py",
                   encoding="utf-8").read()


# ---------------------------------------------------------------------------
# Source gates
# ---------------------------------------------------------------------------

def test_ridge_baseline_exists_and_is_wired():
    src = _src()
    assert "def _ridge_16h_rule(" in src
    assert '"ridge_16h"' in src
    assert "ridge_ctx=ridge_ctx" in src, "registry no longer receives the ctx"
    assert '"train_df": train_df' in src, (
        "the fold call site no longer passes the train fold — ridge_16h "
        "would be silently absent and the promotion gate weaker (P152 shape)"
    )


def test_baselines_without_ctx_stay_unchanged():
    """Optuna/other callers that pass no ridge_ctx must get exactly the two
    historical baselines — the ridge must never appear fit on nothing."""
    src = _src()
    assert "if ridge_ctx is not None:" in src


# ---------------------------------------------------------------------------
# Behavioral — needs the training stack
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def T():
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("sb3_contrib")
    pytest.importorskip("sklearn")
    sys.path.insert(0, str(REPO_ROOT))
    import training.train_drl_full as mod
    return mod


@pytest.fixture
def planted_df():
    """A df where f1 at 4-bar-aligned bars drives the NEXT 4 bars' return by
    construction (non-overlapping blocks — overlapping windows would smear
    the signal across neighbours and dilute sign agreement below any sharp
    threshold). The ridge must recover it at aligned bars."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(3)
    n = 900
    signal = rng.normal(size=n)
    ret = np.zeros(n)
    for i in range(0, n - 4, 4):
        ret[i + 1:i + 5] += 0.0025 * signal[i]
    ret += rng.normal(0, 0.0005, n)
    close = 100 * np.exp(np.cumsum(ret))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
        "close": close,
        "f1": signal,
        "f2": rng.normal(size=n),
        "regime": rng.integers(0, 6, n),
    })


def test_ridge_recovers_a_planted_linear_signal(T, planted_df):
    import numpy as np
    train = planted_df.iloc[:700]
    fn = T._ridge_16h_rule(train, ["f1", "f2"], decision_interval=1,
                           deadband=0.0)
    env = T.TradingEnvFull(df=planted_df, feature_cols=["f1", "f2"],
                           reward_mode="classic")
    env.reset()
    agree = 0
    checked = 0
    for i in range(720, 880, 4):  # aligned bars, where the block signal lives
        env.current_step = i
        a = fn(env)
        truth = planted_df["close"].iloc[min(i + 4, len(planted_df) - 1)] / \
            planted_df["close"].iloc[i] - 1
        if abs(a) > 0.05 and abs(truth) > 1e-4:
            checked += 1
            agree += (a > 0) == (truth > 0)
    assert checked > 20
    assert agree / checked > 0.7, (
        f"ridge recovered only {agree}/{checked} signs of a PLANTED linear "
        f"signal — the fit or the feature plumbing is broken"
    )


def test_ridge_holds_between_decision_bars(T, planted_df):
    train = planted_df.iloc[:700]
    fn = T._ridge_16h_rule(train, ["f1", "f2"], decision_interval=4,
                           deadband=0.0)
    env = T.TradingEnvFull(df=planted_df, feature_cols=["f1", "f2"],
                           reward_mode="classic")
    env.reset()
    env.current_step = 720
    actions = []
    for k in range(12):
        env.current_step = 720 + k
        actions.append(fn(env))
    # within each 4-bar block the action must be constant
    for b in range(0, 12, 4):
        block = actions[b:b + 4]
        assert len(set(block)) == 1, f"block {b} not held: {block}"


def test_deadband_flattens_weak_forecasts(T, planted_df):
    import numpy as np
    train = planted_df.iloc[:700]
    fn_wide = T._ridge_16h_rule(train, ["f2"], decision_interval=1,
                                deadband=5.0)  # f2 is noise; huge deadband
    env = T.TradingEnvFull(df=planted_df, feature_cols=["f2"],
                           reward_mode="classic")
    env.reset()
    acts = []
    for i in range(720, 760):
        env.current_step = i
        acts.append(fn_wide(env))
    assert all(a == 0.0 for a in acts), "deadband did not flatten noise"


def test_registry_includes_ridge_only_with_ctx(T, planted_df):
    env = T.TradingEnvFull(df=planted_df, feature_cols=["f1", "f2"],
                           reward_mode="classic")
    without = T.baseline_policies(env)
    assert "ridge_16h" not in without
    with_ridge = T.baseline_policies(env, ridge_ctx={
        "train_df": planted_df.iloc[:700], "feature_cols": ["f1", "f2"],
        "decision_interval": 4})
    assert "ridge_16h" in with_ridge
