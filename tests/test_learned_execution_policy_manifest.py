from pathlib import Path

import numpy as np

from execution.learned_execution_policy import (
    ExecutionAction,
    ExecutionState,
    LOBFeatures,
    LearnedExecutionConfig,
    LearnedExecutionPolicy,
    get_learned_execution_policy,
    reset_learned_execution_policy,
)


def _sample_inputs():
    lob = LOBFeatures(
        spread_bps=8.0,
        imbalance=0.1,
        bid_depth_usd=500_000.0,
        ask_depth_usd=450_000.0,
        vpin=0.25,
    )
    embedding = np.zeros(128, dtype=np.float32)
    exec_state = ExecutionState(
        target_quantity=1.0,
        executed_quantity=0.0,
        side="sell",
        arrival_price=100.0,
        current_mid_price=99.5,
        deadline_sec=1200,
    )
    return lob, embedding, exec_state


def test_learned_execution_policy_fails_closed_without_validated_manifest(tmp_path):
    missing_manifest = tmp_path / "missing_manifest.json"
    policy = LearnedExecutionPolicy(
        LearnedExecutionConfig(
            mode="advisory",
            manifest_path=str(missing_manifest),
            require_validated_manifest=True,
        )
    )
    lob, embedding, exec_state = _sample_inputs()

    advice = policy.get_advice(lob, embedding, exec_state)
    status = policy.get_status()

    assert advice.is_fallback is True
    assert status["effective_mode"] == "ADVISORY_HEURISTIC"
    assert status["network_usable"] is False
    assert status["validated_manifest_loaded"] is False
    assert status["validated_manifest_error"] == "manifest_not_found"


def test_learned_execution_policy_uses_network_only_after_validated_manifest(tmp_path):
    weights_path = tmp_path / "policy.npy"
    manifest_path = tmp_path / "manifest.json"
    policy = LearnedExecutionPolicy(
        LearnedExecutionConfig(
            mode="advisory",
            manifest_path=str(manifest_path),
            require_validated_manifest=True,
        )
    )
    weights = {
        "W1": np.zeros((policy._policy.input_dim, policy.config.hidden_dim)),
        "b1": np.zeros(policy.config.hidden_dim),
        "W2": np.zeros((policy.config.hidden_dim, policy.config.output_dim)),
        "b2": np.array([5.0, 0.0, 0.0, 0.0]),
        "W_params": np.zeros((policy.config.hidden_dim, 4)),
        "b_params": np.zeros(4),
    }
    np.save(weights_path, weights, allow_pickle=True)
    sha256 = policy._sha256_file(weights_path)
    manifest_path.write_text(
        (
            "{\n"
            f'  "weights_path": "{weights_path.name}",\n'
            f'  "weights_sha256": "{sha256}",\n'
            f'  "input_dim": {policy._policy.input_dim},\n'
            f'  "hidden_dim": {policy.config.hidden_dim},\n'
            f'  "output_dim": {policy.config.output_dim}\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    assert policy.load_validated_manifest(str(manifest_path)) is True

    lob, embedding, exec_state = _sample_inputs()
    advice = policy.get_advice(lob, embedding, exec_state)
    status = policy.get_status()

    assert advice.is_fallback is False
    assert advice.recommended_action == ExecutionAction.WAIT
    assert status["effective_mode"] == "ADVISORY_LEARNED"
    assert status["network_usable"] is True
    assert status["validated_manifest_loaded"] is True
    assert Path(status["validated_manifest_path"]).name == "manifest.json"


def test_get_learned_execution_policy_refreshes_when_config_changes():
    reset_learned_execution_policy()
    first = get_learned_execution_policy(LearnedExecutionConfig(mode="shadow"))
    second = get_learned_execution_policy(LearnedExecutionConfig(mode="advisory"))

    assert first is not second
    assert second.config.mode == "advisory"
    assert second.get_status()["mode"] == "advisory"

    reset_learned_execution_policy()
