"""[P414c] Live jump-regime shadow: online-filter step, persistence, vocabulary.

The shadow is the shadow-first step of the GMM->jump swap. These pin: the online
forward-DP filter holds a state under the jump penalty (the churn win), the
filter state survives a restart warm (P301), it never claims a position (Iron
Law 7 observation-only), and the exported artifacts map to control-compatible
regime names (the swap's key blocker check, P217/P267)."""
import json
from pathlib import Path

import pytest

from defense.jump_regime_shadow import JumpRegimeShadow

REPO = Path(__file__).resolve().parent.parent
GMM_VOCAB = {"NEUTRAL_DRIFT", "VOLATILE_CHOP", "EXTREME_VOLATILITY",
             "STEADY_UPTREND", "MOMENTUM_RALLY", "WEAK_CONSOLIDATION",
             "QUIET_ACCUMULATION", "PANIC_SELLOFF"}


def _fake_model(tmp):
    """A 2-state model: centroid A at -1, B at +1 (1 feature), lambda high."""
    cfg = REPO / "configs" / "jumpregime"
    return {
        "asset": "BTC", "lambda": 20.0, "k": 2,
        "scaler_mean": [0.0], "scaler_std": [1.0],
        "centroids": [[-1.0], [1.0]],
        "state_to_name": {"0": "QUIET_ACCUMULATION", "1": "MOMENTUM_RALLY"},
    }


def _shadow_with(tmp_path, model):
    s = JumpRegimeShadow.__new__(JumpRegimeShadow)
    s._models = {"BTC": model}
    s._cost = {}; s._last_label = {}; s._jsw = {}; s._gsw = {}
    s._nseen = {}; s._last_gmm = {}
    s._state_path = tmp_path / "jumpregime_state.json"
    return s


def test_step_picks_nearest_centroid_when_cold():
    s = _shadow_with(Path("."), _fake_model(None))
    label, name, switched = s.step("BTC", [1.0])   # near centroid B (+1)
    assert label == 1 and name == "MOMENTUM_RALLY"


def test_jump_penalty_holds_state_against_a_small_wiggle(tmp_path):
    """The churn win: once in a state, a marginal feature move must NOT flip it
    (the jump penalty), while a decisive move does."""
    s = _shadow_with(tmp_path, _fake_model(None))
    s.step("BTC", [1.0])            # settle in state 1 (+1)
    # a tiny move toward the other centroid must be held by the penalty
    _, _, switched = s.step("BTC", [0.2])
    assert switched is False, "small wiggle must not flip the regime"
    # a decisive move flips it
    for _ in range(3):
        label, _, _ = s.step("BTC", [-1.0])
    assert label == 0


def test_state_persists_warm_across_restart(tmp_path):
    m = _fake_model(None)
    s1 = _shadow_with(tmp_path, m)
    s1.step("BTC", [1.0]); s1._persist_state()
    s2 = _shadow_with(tmp_path, m)
    s2._restore_state()
    assert s2._cost.get("BTC") is not None
    assert s2._last_label.get("BTC") == 1   # warm, not cold


def test_tick_is_observation_only_and_logs_churn(tmp_path):
    s = _shadow_with(tmp_path, _fake_model(None))
    out = s.tick({"BTC": [1.0]}, {"BTC": "MOMENTUM_RALLY"})
    assert len(out) == 1 and "MOMENTUM_RALLY" in out[0]
    assert "jump" in out[0] and "gmm" in out[0]   # churn comparison present
    # no position/direction concept exists on this object at all
    assert not hasattr(s, "target_exposure") and not hasattr(s, "direction")


def test_exported_artifacts_map_to_control_compatible_names():
    """The swap blocker: every jump state must inherit a name the live control
    tables understand (P217/P267)."""
    cfg = REPO / "configs" / "jumpregime"
    files = sorted(cfg.glob("*.json"))
    assert files, "no jumpregime artifacts exported"
    for f in files:
        m = json.loads(f.read_text(encoding="utf-8"))
        names = set(m["state_to_name"].values())
        unknown = names - GMM_VOCAB
        assert not unknown, f"{f.name}: states map to non-control names {unknown}"
        assert len(names) >= 3, f"{f.name}: vocabulary collapsed to <3 regimes"
        assert m["provenance"]["fit_policy"] == "split_aware"
