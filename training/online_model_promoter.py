from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


ALLOWED_ONLINE_DRL_AUTHORITIES = ("DISABLED", "SHADOW", "EXIT_ONLY")


def cap_online_drl_authority(level: str) -> str:
    normalized = str(level or "SHADOW").upper()
    if normalized == "ACTIVE":
        return "EXIT_ONLY"
    if normalized not in ALLOWED_ONLINE_DRL_AUTHORITIES:
        return "SHADOW"
    return normalized


@dataclass
class OnlineModelPromoterConfig:
    output_dir: str = "data/drl_online"
    min_candidate_score: float = 0.55
    min_observations: int = 100
    min_outcomes: int = 10


class OnlineModelPromoter:
    """Evaluate online DRL candidates while hard-capping runtime authority to EXIT_ONLY."""

    def __init__(self, config: OnlineModelPromoterConfig | None = None):
        self.config = config or OnlineModelPromoterConfig()

    def evaluate_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        counts = dict((candidate.get("experience_summary", {}) or {}).get("counts", {}) or {})
        observations = int(counts.get("observation", 0) or 0)
        outcomes = int(counts.get("outcome", 0) or 0)
        score = float(candidate.get("score", 0.0) or 0.0)
        requested_authority = str(candidate.get("requested_authority", "SHADOW") or "SHADOW")
        capped_authority = cap_online_drl_authority(requested_authority)

        sample_ready = (
            observations >= int(self.config.min_observations)
            and outcomes >= int(self.config.min_outcomes)
        )
        eligible = bool(candidate.get("eligible", False)) and sample_ready
        if not eligible:
            applied_authority = "SHADOW"
            decision = "HOLD_SHADOW"
            reason = (
                f"sample_not_ready obs={observations}/{self.config.min_observations} "
                f"outcomes={outcomes}/{self.config.min_outcomes}"
            )
        elif score < float(self.config.min_candidate_score):
            applied_authority = "SHADOW"
            decision = "HOLD_SHADOW"
            reason = f"score={score:.3f}<{self.config.min_candidate_score:.3f}"
        else:
            applied_authority = capped_authority
            decision = "PROMOTE"
            reason = (
                f"requested={requested_authority.upper()} capped={capped_authority} "
                f"score={score:.3f}"
            )

        result = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "eligible": bool(eligible),
            "sample_ready": bool(sample_ready),
            "requested_authority": requested_authority.upper(),
            "applied_authority": applied_authority,
            "decision": decision,
            "reason": reason,
            "score": score,
            "experience_counts": {
                "observation": observations,
                "outcome": outcomes,
                "decision": int(counts.get("decision", 0) or 0),
            },
        }
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "online_promotion_decision.json"
        result["output_file"] = str(out_file)
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


__all__ = [
    "ALLOWED_ONLINE_DRL_AUTHORITIES",
    "OnlineModelPromoter",
    "OnlineModelPromoterConfig",
    "cap_online_drl_authority",
]
