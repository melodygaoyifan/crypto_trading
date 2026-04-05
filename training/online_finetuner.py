from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass
class OnlineFineTunerConfig:
    experience_dir: str = "data/live_experiences"
    output_dir: str = "data/drl_online"
    min_observations: int = 100
    min_outcomes: int = 10
    max_recent_files: int = 14


class OnlineFineTuner:
    """Build shadow-only DRL finetune proposals from live experience logs."""

    def __init__(self, config: OnlineFineTunerConfig | None = None):
        self.config = config or OnlineFineTunerConfig()

    def summarize_experiences(self) -> Dict[str, Any]:
        exp_dir = Path(self.config.experience_dir)
        files = sorted(exp_dir.glob("experiences_*.jsonl"))[-max(1, int(self.config.max_recent_files)) :]
        counts = {"observation": 0, "decision": 0, "outcome": 0}
        latest_timestamp = ""
        for path in files:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    row_type = str(row.get("type", "") or "").lower()
                    if row_type in counts:
                        counts[row_type] += 1
                    latest_timestamp = str(row.get("timestamp", latest_timestamp) or latest_timestamp)
        return {
            "experience_dir": str(exp_dir),
            "files_scanned": [str(path) for path in files],
            "counts": counts,
            "latest_timestamp": latest_timestamp,
        }

    def build_shadow_finetune_candidate(self) -> Dict[str, Any]:
        summary = self.summarize_experiences()
        observation_count = int((summary.get("counts", {}) or {}).get("observation", 0) or 0)
        outcome_count = int((summary.get("counts", {}) or {}).get("outcome", 0) or 0)
        eligible = (
            observation_count >= int(self.config.min_observations)
            and outcome_count >= int(self.config.min_outcomes)
        )
        reason = "READY_FOR_SHADOW_FINETUNE" if eligible else (
            f"INSUFFICIENT_SAMPLE obs={observation_count}/{self.config.min_observations} "
            f"outcomes={outcome_count}/{self.config.min_outcomes}"
        )
        candidate = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "eligible": bool(eligible),
            "reason": reason,
            "target_authority": "SHADOW",
            "experience_summary": summary,
        }
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "online_finetune_candidate.json"
        candidate["output_file"] = str(out_file)
        out_file.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        return candidate


__all__ = ["OnlineFineTuner", "OnlineFineTunerConfig"]
