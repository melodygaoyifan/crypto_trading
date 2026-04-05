import json

from training.online_finetuner import OnlineFineTuner, OnlineFineTunerConfig
from training.online_model_promoter import (
    OnlineModelPromoter,
    OnlineModelPromoterConfig,
    cap_online_drl_authority,
)


def test_online_finetuner_builds_shadow_only_candidate(tmp_path):
    exp_dir = tmp_path / "live_experiences"
    exp_dir.mkdir()
    exp_file = exp_dir / "experiences_2026-03-12.jsonl"
    rows = [
        {"type": "observation", "timestamp": "2026-03-12T00:00:00Z"},
        {"type": "observation", "timestamp": "2026-03-12T04:00:00Z"},
        {"type": "outcome", "timestamp": "2026-03-12T08:00:00Z"},
    ]
    exp_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    finetuner = OnlineFineTuner(
        OnlineFineTunerConfig(
            experience_dir=str(exp_dir),
            output_dir=str(tmp_path / "out"),
            min_observations=2,
            min_outcomes=1,
        )
    )

    candidate = finetuner.build_shadow_finetune_candidate()
    on_disk = json.loads((tmp_path / "out" / "online_finetune_candidate.json").read_text(encoding="utf-8"))

    assert candidate["eligible"] is True
    assert candidate["target_authority"] == "SHADOW"
    assert candidate["experience_summary"]["counts"]["observation"] == 2
    assert candidate["experience_summary"]["counts"]["outcome"] == 1
    assert candidate["output_file"].endswith("online_finetune_candidate.json")
    assert on_disk["output_file"] == candidate["output_file"]


def test_online_model_promoter_caps_requested_active_to_exit_only(tmp_path):
    promoter = OnlineModelPromoter(
        OnlineModelPromoterConfig(
            output_dir=str(tmp_path / "out"),
            min_candidate_score=0.50,
            min_observations=2,
            min_outcomes=1,
        )
    )
    candidate = {
        "eligible": True,
        "requested_authority": "ACTIVE",
        "score": 0.90,
        "experience_summary": {
            "counts": {"observation": 2, "outcome": 1, "decision": 1}
        },
    }

    result = promoter.evaluate_candidate(candidate)
    on_disk = json.loads((tmp_path / "out" / "online_promotion_decision.json").read_text(encoding="utf-8"))

    assert cap_online_drl_authority("ACTIVE") == "EXIT_ONLY"
    assert result["decision"] == "PROMOTE"
    assert result["applied_authority"] == "EXIT_ONLY"
    assert result["requested_authority"] == "ACTIVE"
    assert on_disk["output_file"] == result["output_file"]
