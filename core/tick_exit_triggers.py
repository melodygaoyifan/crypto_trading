"""
Exit Trigger Evaluator — extracted from main.py Phase 1A refactoring.

Evaluates 7 exit triggers in priority order:
  T10_SOFT_STOP    — price hits soft stop level
  T9_GAMBLER       — underwater position + structure failure (50% reduce)
  T3_REGIME_EXIT   — regime turned hostile to position direction
  T17_ALPHA_FADE   — signal edge decaying over holding period (gradual reduce)
  T16_TIME_STOP    — position held too long while losing (full close)
  T1_TRAILING_STOP — ATR-based adaptive trailing stop hit
  T3-T12 EXIT_ALPHA — phase-aware scale-out + runner management

Each trigger can OVERRIDE the intent returned by integration_v36.decide().
"""
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.canonical_enums import ExecutionMode

logger = logging.getLogger(__name__)


class ExitTriggerResult:
    """Result of exit trigger evaluation."""
    __slots__ = ('triggered', 'tag', 'direction', 'target_exposure',
                 'force_execution', 'execution_mode', 'urgency', 'adaptive_stop_hit',
                 'exit_alpha_fired')

    def __init__(self):
        self.triggered = False
        self.tag: str = ""
        self.direction: Optional[float] = None
        self.target_exposure: Optional[float] = None
        self.force_execution: bool = False
        self.execution_mode: Optional[ExecutionMode] = None
        self.urgency: float = 1.0
        self.adaptive_stop_hit: bool = False
        self.exit_alpha_fired: bool = False


def evaluate_exit_triggers(
    asset: str,
    intent: Any,
    market_data: Dict[str, Any],
    agent_signals: Dict[str, Any],
    position_state: Dict[str, Any],
    paper_positions: Dict[str, Dict],
    exit_trigger_tags: Dict[str, str],
    # Dependencies (previously self.*)
    stop_authority: Any = None,
    gambler_exit: Any = None,
    drl_authority_level: str = "DISABLED",
    adaptive_stop: Any = None,
    adaptive_stop_regime_mult: Optional[Dict[str, float]] = None,
    exit_alpha: Any = None,
    engine: Any = None,
    regime_exit_fired: Optional[Dict[str, str]] = None,
) -> ExitTriggerResult:
    """
    Evaluate all exit triggers and apply overrides to intent.

    Returns ExitTriggerResult with metadata about what fired.
    Modifies intent IN PLACE (same as original code).
    """
    result = ExitTriggerResult()
    if regime_exit_fired is None:
        regime_exit_fired = {}
    if adaptive_stop_regime_mult is None:
        adaptive_stop_regime_mult = {}

    # =================================================================
    # T10: SOFT STOP CHECK
    # =================================================================
    if stop_authority and position_state.get("has_position"):
        try:
            _curr_price = market_data.get("current_price", 0)
            _soft_hit = stop_authority.check_stops(asset, _curr_price)
            if _soft_hit:
                exit_trigger_tags[asset] = "T10_SOFT_STOP"
                logger.warning(f"[SOFT_STOP] {asset}: triggered -{_soft_hit}")
                _pos_dir = position_state.get("direction", 0)
                if _pos_dir != 0:
                    intent.veto_active = False
                    intent.direction = -_pos_dir
                    intent.target_exposure = 0.0
                    intent.force_execution = True
                    intent.execution_mode = ExecutionMode.AGGRESSIVE
                    intent.urgency = 1.5
                    result.triggered = True
                    result.tag = "T10_SOFT_STOP"
        except Exception as e:
            logger.debug(f"[SOFT_STOP] Check skipped: {e}")

    # =================================================================
    # T9: GAMBLER SOFT EXIT (fail-fast for underwater positions)
    # =================================================================
    if (gambler_exit and position_state.get("has_position")
            and not intent.veto_active):
        try:
            _pos = paper_positions.get(asset, {})
            _pos_mode_at_entry = _pos.get("mode_at_entry", "NORMAL")
            if _pos_mode_at_entry == "NORMAL":
                raise Exception("NORMAL mode: skip gambler exit (wrong thresholds)")
            _entry_t = _pos.get("entry_time", "")
            _bars = int((time.time() - datetime.fromisoformat(_entry_t).timestamp()) / 14400) if _entry_t else 0
            _entry_p = _pos.get("entry_price", 0)
            _curr_p = market_data.get("current_price", 0)
            _pos_dir = _pos.get("direction", 0)
            _unrealized_pct = (_curr_p - _entry_p) / _entry_p * _pos_dir if _entry_p > 0 else 0.0
            _unrealized_bps = _unrealized_pct * 10000
            _g_pos_sign = 1 if _pos_dir > 0 else -1 if _pos_dir < 0 else 0
            _g_intent_sign = 1 if intent.direction > 0 else -1 if intent.direction < 0 else 0
            _g_adverse_momentum = _g_intent_sign != 0 and _g_intent_sign != _g_pos_sign

            _pos_strategy = _pos.get("strategy", "")
            _effective_bars = _bars
            if _pos_strategy == "mean_revert" and _bars <= 2:
                _effective_bars = 0

            _g_exit = gambler_exit.check_soft_exit(
                asset=asset,
                unrealized_pnl_bps=_unrealized_bps,
                vpin=market_data.get("vpin", 0.35),
                structure_confirmed=market_data.get("higher_lows_detected", False),
                adverse_momentum=_g_adverse_momentum,
                bars_since_entry=_effective_bars,
                strategy=_pos_strategy,
            )
            if _g_exit.should_exit:
                _gambler_suppressed = False
                if drl_authority_level == "ACTIVE" and "structure unconfirmed" in _g_exit.reason:
                    _drl_dir = agent_signals.get("drl_direction", 0.0)
                    _drl_conf = agent_signals.get("drl_confidence", 0.0)
                    _drl_aligns = (_drl_dir * _pos_dir) > 0.1 and _drl_conf >= 0.4
                    if _drl_aligns:
                        _gambler_suppressed = True
                        logger.info(
                            f"[GAMBLER_EXIT] {asset}: SUPPRESSED by DRL DECIDE "
                            f"(drl_dir={_drl_dir:+.2f}, pos_dir={_pos_dir:+.0f}, "
                            f"drl_conf={_drl_conf:.2f}). Reason was: {_g_exit.reason}"
                        )

                if not _gambler_suppressed:
                    exit_trigger_tags[asset] = "T9_GAMBLER"
                    _gambler_current_exp = abs(paper_positions.get(asset, {}).get("exposure", 0))
                    _gambler_reduced_exp = _gambler_current_exp * 0.50
                    logger.info(
                        f"[GAMBLER_EXIT] {asset}: REDUCE 50% (not CLOSE) "
                        f"-{_g_exit.reason} (bars={_bars}, pnl={_unrealized_pct:.2%}, "
                        f"exp {_gambler_current_exp:.4f}->{_gambler_reduced_exp:.4f})"
                    )
                    intent.veto_active = False
                    intent.direction = _pos_dir
                    intent.target_exposure = _gambler_reduced_exp
                    intent.force_execution = True
                    intent.execution_mode = ExecutionMode.AGGRESSIVE
                    intent.urgency = 1.2
                    result.triggered = True
                    result.tag = "T9_GAMBLER"
        except Exception as e:
            logger.debug(f"[GAMBLER_EXIT] Check skipped: {e}")

    # =================================================================
    # T3: REGIME EXIT — regime turned hostile to position direction
    # =================================================================
    _REGIME_HOSTILE_LONG = {"BEAR_TREND", "PANIC_SELLOFF", "EXTREME_VOLATILITY"}
    _REGIME_HOSTILE_SHORT = {"STRONG_TREND", "BULL_TREND", "MOMENTUM_RALLY", "STEADY_UPTREND"}

    if (position_state.get("has_position")
            and not getattr(intent, 'force_execution', False)):
        try:
            _re_pos = paper_positions.get(asset, {})
            _re_dir = _re_pos.get("direction", 0)
            _re_regime = market_data.get("regime_state", "UNKNOWN")
            _re_entry_regime = _re_pos.get("regime", "UNKNOWN")

            _re_hostile = False
            if _re_dir > 0 and _re_regime in _REGIME_HOSTILE_LONG:
                _re_hostile = True
            elif _re_dir < 0 and _re_regime in _REGIME_HOSTILE_SHORT:
                _re_hostile = True

            _re_prev_fired = regime_exit_fired.get(asset)

            if _re_hostile and _re_prev_fired != _re_regime and _re_regime != _re_entry_regime:
                regime_exit_fired[asset] = _re_regime
                exit_trigger_tags[asset] = "T3_REGIME_EXIT"
                _re_scale_pct = 0.50
                _re_new_exp = abs(_re_pos.get("exposure", 0)) * (1 - _re_scale_pct)
                intent.veto_active = False
                intent.target_exposure = _re_new_exp
                intent.direction = _re_dir
                intent.execution_mode = ExecutionMode.NORMAL
                intent.urgency = 0.8
                logger.info(
                    f"[REGIME_EXIT] {asset}: regime turned hostile "
                    f"({_re_entry_regime} ->{_re_regime}), "
                    f"dir={'LONG' if _re_dir > 0 else 'SHORT'}, "
                    f"scaling out {_re_scale_pct:.0%}"
                )
                result.triggered = True
                result.tag = "T3_REGIME_EXIT"
            elif not _re_hostile:
                regime_exit_fired.pop(asset, None)
        except Exception as e:
            logger.debug(f"[REGIME_EXIT] Check skipped: {e}")

    # =================================================================
    # T17/T16: TIME-BASED EXIT with ALPHA FADE
    # =================================================================
    if (position_state.get("has_position")
            and not getattr(intent, 'force_execution', False)):
        try:
            _tb_pos = paper_positions.get(asset, {})
            _tb_entry_t = _tb_pos.get("entry_time", "")
            if _tb_entry_t:
                _tb_entry_dt = datetime.fromisoformat(_tb_entry_t)
                _tb_bars_held = int(
                    (datetime.now(timezone.utc) - _tb_entry_dt).total_seconds() / 14400
                )
                _tb_entry_p = _tb_pos.get("entry_price", 0)
                _tb_curr_p = market_data.get("current_price", 0)
                _tb_dir = _tb_pos.get("direction", 0)
                _tb_pnl_pct = (
                    (_tb_curr_p - _tb_entry_p) / _tb_entry_p * _tb_dir
                    if _tb_entry_p > 0 else 0.0
                )
                _tb_is_profitable = _tb_pnl_pct > 0.001
                _tb_strategy = _tb_pos.get("strategy", "")
                try:
                    from configs.high_risk_mode import get_high_risk_config
                    _hrc = get_high_risk_config()
                    _tb_min_bars = _hrc.get_time_stop_bars(
                        _tb_is_profitable, _tb_strategy
                    ) if _hrc else (4 if _tb_is_profitable else 2)
                except Exception:
                    _tb_min_bars = 4 if _tb_is_profitable else 2

                # Alpha Fade: gradual reduction before hard time stop
                _alpha_fade = max(0.0, 1.0 - _tb_bars_held / max(_tb_min_bars, 1))
                _tb_current_exp = abs(_tb_pos.get("exposure", 0))
                if (0.0 < _alpha_fade < 0.5 and _tb_pnl_pct < 0.001
                        and _tb_current_exp > 0.01 and not _tb_is_profitable):
                    _faded_exp = _tb_current_exp * max(_alpha_fade, 0.25)
                    exit_trigger_tags[asset] = "T17_ALPHA_FADE"
                    intent.veto_active = False
                    intent.direction = _tb_dir
                    intent.target_exposure = _faded_exp
                    intent.force_execution = True
                    intent.execution_mode = ExecutionMode.AGGRESSIVE_MAKER
                    intent.urgency = 0.8
                    logger.info(
                        f"[ALPHA_FADE] {asset}: fade={_alpha_fade:.2f} "
                        f"({_tb_bars_held}/{_tb_min_bars} bars), "
                        f"reducing {_tb_current_exp:.4f}→{_faded_exp:.4f}"
                    )
                    result.triggered = True
                    result.tag = "T17_ALPHA_FADE"
                elif _tb_bars_held >= _tb_min_bars and _tb_pnl_pct < -0.001:
                    exit_trigger_tags[asset] = "T16_TIME_STOP"
                    intent.veto_active = False
                    intent.direction = -_tb_dir if _tb_dir != 0 else -1.0
                    intent.target_exposure = 0.0
                    intent.force_execution = True
                    intent.execution_mode = ExecutionMode.AGGRESSIVE
                    intent.urgency = 1.2
                    logger.warning(
                        f"[TIME_STOP] {asset}: held {_tb_bars_held} ticks "
                        f"({_tb_bars_held * 4}h), still losing "
                        f"({_tb_pnl_pct:+.2%}) → force exit"
                    )
                    result.triggered = True
                    result.tag = "T16_TIME_STOP"
        except Exception as _tb_err:
            logger.debug(f"[TIME_STOP] Check skipped: {_tb_err}")

    # =================================================================
    # T1: ADAPTIVE TRAILING STOP (ATR-based)
    # =================================================================
    if (adaptive_stop and position_state.get("has_position")
            and not getattr(intent, 'force_execution', False)):
        try:
            _w2_pos = paper_positions.get(asset, {})
            _w2_dir = _w2_pos.get("direction", 0)
            _w2_price = market_data.get("current_price", 0)
            if _w2_price > 0 and _w2_dir != 0:
                _w2_regime = market_data.get("regime_state", "NEUTRAL")
                _w2_regime_mult = adaptive_stop_regime_mult.get(_w2_regime, 1.0)
                _w2_base_atr_mult = 3.5
                _w2_effective_mult = _w2_base_atr_mult * _w2_regime_mult
                adaptive_stop.config.atr_multiplier = _w2_effective_mult
                adaptive_stop.update_stop(asset, _w2_price)
                _w2_hit, _w2_reason, _w2_state = adaptive_stop.check_stop_hit(
                    asset, _w2_price
                )
                if _w2_hit and _w2_state:
                    result.adaptive_stop_hit = True
                    exit_trigger_tags[asset] = "T1_TRAILING_STOP"
                    logger.warning(
                        f"[WIRE-2] {asset}: ADAPTIVE STOP HIT -"
                        f"reason={_w2_reason.value}, "
                        f"stop=${_w2_state.current_stop:.2f}, "
                        f"price=${_w2_price:.2f}, "
                        f"R={_w2_state.current_r_multiple:.2f}, "
                        f"type={_w2_state.stop_type.value}"
                    )
                    intent.veto_active = False
                    intent.target_exposure = 0.0
                    intent.direction = -_w2_dir if _w2_dir != 0 else -1.0
                    intent.force_execution = True
                    intent.execution_mode = ExecutionMode.AGGRESSIVE
                    intent.urgency = 1.5
                    result.triggered = True
                    result.tag = "T1_TRAILING_STOP"
        except Exception as _w2_err:
            logger.debug(f"[WIRE-2] Adaptive stop check skipped: {_w2_err}")

    # =================================================================
    # EXIT ALPHA: Phase-aware scale-out + runner management
    # =================================================================
    if exit_alpha and position_state.get("has_position"):
        try:
            from execution.exit_alpha import RunnerAction
            _pos = paper_positions.get(asset, {})
            _pos_dir = _pos.get("direction", 0)
            _entry_p = _pos.get("entry_price", 0)
            _curr_p = market_data.get("current_price", 0)

            _bars_held = 1
            _entry_time_str = _pos.get("entry_time", "")
            if _entry_time_str:
                try:
                    _entry_dt = datetime.fromisoformat(_entry_time_str)
                    _bars_held = max(1, int((datetime.now(timezone.utc) - _entry_dt).total_seconds() / 14400))
                except Exception:
                    pass

            # Phase result
            _phase_res = None
            if engine and hasattr(engine, '_last_phase_result'):
                _phase_res = engine._last_phase_result
            if _phase_res is None or not hasattr(_phase_res, 'phase_transition'):
                from execution.exit_alpha import RegimePhaseResult as ExitRegimePhaseResult
                _phase_res = ExitRegimePhaseResult()

            # For SHORT: swap prices for profit calc
            if _pos_dir < 0:
                _ea_curr_p, _ea_entry_p = _entry_p, _curr_p
            else:
                _ea_curr_p, _ea_entry_p = _curr_p, _entry_p

            # DRL exit signal
            _drl_exit_output = None
            if drl_authority_level not in ("DISABLED", "SHADOW"):
                _drl_dir = agent_signals.get('drl_direction', 0.0)
                _drl_vs_pos = _drl_dir * _pos_dir
                if _drl_vs_pos < -0.6:
                    from execution.exit_alpha import DRLOutput, DRLAction
                    _drl_exit_output = DRLOutput(action=DRLAction.PARTIAL_EXIT)
                elif _drl_vs_pos < -0.2:
                    from execution.exit_alpha import DRLOutput, DRLAction
                    _drl_exit_output = DRLOutput(action=DRLAction.HOLD)

            _exit_alpha_signal = exit_alpha.evaluate_scale_out(
                asset=asset,
                current_price=_ea_curr_p,
                entry_price=_ea_entry_p,
                position_size=abs(_pos.get("exposure", 0)),
                bars_held=_bars_held,
                phase_result=_phase_res,
                crack_weight=market_data.get("crack_weight", 0.0),
                momentum=agent_signals.get("quant_direction", 0.0),
                drl_output=_drl_exit_output,
            )

            if _exit_alpha_signal and _exit_alpha_signal.should_scale_out:
                result.exit_alpha_fired = True
                _scale_pct = _exit_alpha_signal.scale_out_pct
                _ea_trigger_map = {
                    "PHASE_TRANSITION": "T3_PHASE_EXIT",
                    "CRACK_DECAY": "T12_CRACK_DECAY",
                    "MOMENTUM_STALL": "T4_MOMENTUM_STALL",
                    "DRL_ACTION": "T6_DRL_ACTION",
                    "DRAWDOWN_FROM_PEAK": "T5_PROFIT_DRAWDOWN",
                }
                _ea_tval = _exit_alpha_signal.trigger.value if hasattr(_exit_alpha_signal.trigger, 'value') else str(_exit_alpha_signal.trigger)
                exit_trigger_tags[asset] = _ea_trigger_map.get(_ea_tval, f"T3_{_ea_tval}")
                logger.info(
                    f"[EXIT_ALPHA] {asset}: SCALE-OUT {_scale_pct:.0%} triggered by "
                    f"{_exit_alpha_signal.trigger.value} -{_exit_alpha_signal.reason} "
                    f"(profit={_exit_alpha_signal.profit_bps:.0f}bps, "
                    f"peak={_exit_alpha_signal.peak_profit_bps:.0f}bps)"
                )

                _new_exposure = abs(_pos.get("exposure", 0)) * (1 - _scale_pct)
                intent.veto_active = False
                intent.force_execution = False
                intent.target_exposure = _new_exposure
                intent.direction = _pos_dir

                exit_alpha.create_runner(
                    asset=asset,
                    remaining_size_pct=1.0 - _scale_pct,
                    entry_price=_entry_p,
                    current_price=_curr_p,
                    direction=_pos_dir,
                    strategy=_pos.get("strategy", ""),
                    regime=market_data.get("regime_state", ""),
                )
                result.triggered = True
                result.tag = exit_trigger_tags.get(asset, "EXIT_ALPHA")
        except Exception as e:
            logger.debug(f"[EXIT_ALPHA] Evaluation skipped: {e}")

    # Runner management (only when scale-out didn't just fire)
    if exit_alpha and not result.exit_alpha_fired:
        try:
            from execution.exit_alpha import RunnerAction
            _runner = exit_alpha.get_runner(asset)
            if _runner and _runner.active:
                _pos = paper_positions.get(asset, {})
                _pos_dir = _pos.get("direction", 0)
                _curr_p = market_data.get("current_price", 0)

                _phase_res = None
                if engine and hasattr(engine, '_last_phase_result'):
                    _phase_res = engine._last_phase_result
                if _phase_res is None or not hasattr(_phase_res, 'phase_transition'):
                    from execution.exit_alpha import RegimePhaseResult as ExitRegimePhaseResult
                    _phase_res = ExitRegimePhaseResult()

                _drl_runner_output = None
                if drl_authority_level not in ("DISABLED", "SHADOW"):
                    _drl_dir = agent_signals.get('drl_direction', 0.0)
                    _drl_conf = agent_signals.get('drl_confidence', 0.0)
                    _drl_vs_pos = _drl_dir * _pos_dir
                    if _drl_vs_pos < -0.8 and _drl_conf >= 0.6:
                        from execution.exit_alpha import DRLOutput, DRLAction
                        _drl_runner_output = DRLOutput(action=DRLAction.RELEASE_RUNNER)
                    elif _drl_vs_pos < -0.2:
                        from execution.exit_alpha import DRLOutput
                        try:
                            from agents.drl_agent import DRLAction as _RealDRLAction
                            _drl_runner_output = DRLOutput(action=_RealDRLAction.INCREASE_EXIT_PRESSURE)
                        except ImportError:
                            pass

                _runner_action, _runner_state = exit_alpha.manage_runner(
                    asset=asset,
                    current_price=_curr_p,
                    direction=_pos_dir,
                    phase_result=_phase_res,
                    drl_output=_drl_runner_output,
                    strategy=paper_positions.get(asset, {}).get("strategy", ""),
                    regime=market_data.get("regime_state", ""),
                )

                if _runner_action == RunnerAction.RELEASE:
                    exit_trigger_tags[asset] = "T11_RUNNER_RELEASE"
                    logger.info(f"[EXIT_ALPHA] {asset}: RUNNER RELEASED -closing remaining position")
                    intent.veto_active = False
                    intent.target_exposure = 0.0
                    intent.direction = -_pos_dir
                    intent.force_execution = True
                    intent.execution_mode = ExecutionMode.AGGRESSIVE
                    intent.urgency = 1.5
                    result.triggered = True
                    result.tag = "T11_RUNNER_RELEASE"
                elif _runner_action == RunnerAction.TIGHTEN:
                    logger.info(
                        f"[EXIT_ALPHA] {asset}: Runner trail tightened "
                        f"to {_runner_state.trail_pct:.1%}"
                    )
        except Exception as e:
            logger.debug(f"[EXIT_ALPHA] Runner management skipped: {e}")

    return result
