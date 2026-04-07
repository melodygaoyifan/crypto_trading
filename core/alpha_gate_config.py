"""
Alpha Gate Configuration Parser - extracted from main.py _init_components (Phase 2).

Parses alpha_gate_config dict into a structured dataclass, eliminating
~180 lines of repetitive self._* assignments in HMATSProductionRunner.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _resolve_by_asset_dict(raw: Any) -> Dict[str, float]:
    """Safely parse a {ASSET: float} dict from config, uppercasing keys."""
    if not isinstance(raw, dict):
        return {}
    result = {}
    for asset_key, value in raw.items():
        try:
            result[str(asset_key).upper()] = max(0.0, float(value or 0.0))
        except Exception:
            continue
    return result


@dataclass
class AlphaGateSettings:
    """Parsed alpha gate configuration settings."""
    min_alpha_bps: float = 0.0
    min_alpha_bps_short: float = 0.0
    alpha_stage_selector: str = ""

    quiet_accum_min_direction: float = 0.15       # [UTIL-2] was 0.35 — unblocks SOL in quiet regimes
    quiet_accum_min_direction_by_asset: Dict[str, float] = field(default_factory=dict)
    quiet_accum_min_direction_short: float = 0.15  # [UTIL-2] was 0.35
    quiet_accum_min_direction_short_by_asset: Dict[str, float] = field(default_factory=dict)

    opportunity_actionable_min_direction_short: float = 0.05
    opportunity_actionable_min_direction_short_by_asset: Dict[str, float] = field(default_factory=dict)
    opportunity_actionable_min_exposure: float = 0.01
    opportunity_actionable_min_exposure_by_asset: Dict[str, float] = field(default_factory=dict)

    normal_actionable_min_direction_short: float = 0.10

    borderline_alpha_pass_epsilon_bps: float = 0.0
    borderline_alpha_pass_epsilon_bps_by_asset: Dict[str, float] = field(default_factory=dict)
    borderline_alpha_pass_epsilon_bps_short: float = 0.0
    borderline_alpha_pass_epsilon_bps_short_by_asset: Dict[str, float] = field(default_factory=dict)

    pre_alpha_hold_volume_breakout_long_in_quiet: bool = False
    pre_alpha_hold_volume_breakout_long_max_direction: float = 0.05

    suppress_model_alpha_long_context_in_quiet: bool = False

    @classmethod
    def from_config(cls, alpha_gate_config: Optional[Dict[str, Any]]) -> "AlphaGateSettings":
        """Parse alpha gate config dict into structured settings."""
        s = cls()
        if not alpha_gate_config:
            return s

        ag = alpha_gate_config
        s.alpha_stage_selector = ag.get("stage_selector", "")

        default_bps = ag.get("min_alpha_bps_default", 0.0)
        default_short_bps = ag.get("min_alpha_bps_short_default", default_bps)
        s.min_alpha_bps = default_bps
        s.min_alpha_bps_short = default_short_bps

        s.quiet_accum_min_direction = float(
            ag.get("quiet_accum_min_direction", s.quiet_accum_min_direction) or 0.0
        )
        s.quiet_accum_min_direction_by_asset = _resolve_by_asset_dict(
            ag.get("quiet_accum_min_direction_by_asset")
        )
        s.quiet_accum_min_direction_short = float(
            ag.get("quiet_accum_min_direction_short", s.quiet_accum_min_direction_short) or 0.0
        )
        s.quiet_accum_min_direction_short_by_asset = _resolve_by_asset_dict(
            ag.get("quiet_accum_min_direction_short_by_asset")
        )
        s.opportunity_actionable_min_direction_short = float(
            ag.get("opportunity_actionable_min_direction_short",
                   s.opportunity_actionable_min_direction_short) or 0.0
        )
        s.opportunity_actionable_min_direction_short_by_asset = _resolve_by_asset_dict(
            ag.get("opportunity_actionable_min_direction_short_by_asset")
        )
        s.opportunity_actionable_min_exposure = float(
            ag.get("opportunity_actionable_min_exposure",
                   s.opportunity_actionable_min_exposure) or 0.0
        )
        s.opportunity_actionable_min_exposure_by_asset = _resolve_by_asset_dict(
            ag.get("opportunity_actionable_min_exposure_by_asset")
        )
        s.normal_actionable_min_direction_short = float(
            ag.get("normal_actionable_min_direction_short",
                   s.normal_actionable_min_direction_short) or 0.0
        )
        s.borderline_alpha_pass_epsilon_bps = float(
            ag.get("borderline_alpha_pass_epsilon_bps",
                   s.borderline_alpha_pass_epsilon_bps) or 0.0
        )
        s.borderline_alpha_pass_epsilon_bps_by_asset = _resolve_by_asset_dict(
            ag.get("borderline_alpha_pass_epsilon_bps_by_asset")
        )
        s.borderline_alpha_pass_epsilon_bps_short = float(
            ag.get("borderline_alpha_pass_epsilon_bps_short",
                   s.borderline_alpha_pass_epsilon_bps_short) or 0.0
        )
        s.borderline_alpha_pass_epsilon_bps_short_by_asset = _resolve_by_asset_dict(
            ag.get("borderline_alpha_pass_epsilon_bps_short_by_asset")
        )
        s.pre_alpha_hold_volume_breakout_long_in_quiet = bool(
            ag.get("pre_alpha_hold_volume_breakout_long_in_quiet",
                   s.pre_alpha_hold_volume_breakout_long_in_quiet)
        )
        s.pre_alpha_hold_volume_breakout_long_max_direction = float(
            ag.get("pre_alpha_hold_volume_breakout_long_max_direction",
                   s.pre_alpha_hold_volume_breakout_long_max_direction) or 0.0
        )
        s.suppress_model_alpha_long_context_in_quiet = bool(
            ag.get("suppress_model_alpha_long_context_in_quiet", False)
        )

        # Resolve from staged_schedule
        for entry in ag.get("staged_schedule", []):
            if entry.get("stage") == s.alpha_stage_selector:
                s.min_alpha_bps = entry.get("min_alpha_bps", default_bps)
                break
        for entry in ag.get("staged_schedule_short", []):
            if entry.get("stage") == s.alpha_stage_selector:
                s.min_alpha_bps_short = entry.get("min_alpha_bps", default_short_bps)
                break

        logger.info(
            f"[ALPHA_GATE] Parsed: stage={s.alpha_stage_selector}, "
            f"min_alpha_bps={s.min_alpha_bps}, "
            f"min_alpha_bps_short={s.min_alpha_bps_short}, "
            f"quiet_dir={s.quiet_accum_min_direction:.2f}, "
            f"quiet_dir_short={s.quiet_accum_min_direction_short:.2f}, "
            f"borderline_eps={s.borderline_alpha_pass_epsilon_bps:.3f}"
        )

        return s

    def apply_to_runner(self, runner: Any) -> None:
        """Apply parsed settings to HMATSProductionRunner self.* attributes.

        Maintains backward compatibility with existing code that reads self._*.
        """
        runner._min_alpha_bps = self.min_alpha_bps
        runner._min_alpha_bps_short = self.min_alpha_bps_short
        runner._alpha_stage_selector = self.alpha_stage_selector
        runner._quiet_accum_min_direction = self.quiet_accum_min_direction
        runner._quiet_accum_min_direction_by_asset = self.quiet_accum_min_direction_by_asset
        runner._quiet_accum_min_direction_short = self.quiet_accum_min_direction_short
        runner._quiet_accum_min_direction_short_by_asset = self.quiet_accum_min_direction_short_by_asset
        runner._opportunity_actionable_min_direction_short = self.opportunity_actionable_min_direction_short
        runner._opportunity_actionable_min_direction_short_by_asset = self.opportunity_actionable_min_direction_short_by_asset
        runner._opportunity_actionable_min_exposure = self.opportunity_actionable_min_exposure
        runner._opportunity_actionable_min_exposure_by_asset = self.opportunity_actionable_min_exposure_by_asset
        runner._normal_actionable_min_direction_short = self.normal_actionable_min_direction_short
        runner._borderline_alpha_pass_epsilon_bps = self.borderline_alpha_pass_epsilon_bps
        runner._borderline_alpha_pass_epsilon_bps_by_asset = self.borderline_alpha_pass_epsilon_bps_by_asset
        runner._borderline_alpha_pass_epsilon_bps_short = self.borderline_alpha_pass_epsilon_bps_short
        runner._borderline_alpha_pass_epsilon_bps_short_by_asset = self.borderline_alpha_pass_epsilon_bps_short_by_asset
        runner._pre_alpha_hold_volume_breakout_long_in_quiet = self.pre_alpha_hold_volume_breakout_long_in_quiet
        runner._pre_alpha_hold_volume_breakout_long_max_direction = self.pre_alpha_hold_volume_breakout_long_max_direction
