"""
================================================================================
EXECUTION SERVICE - Extracted from main.py _execute_intent (Phase 5)
================================================================================
Version: 1.0.0
Purpose: Execute trade intents through the P0 safety pipeline.

This is a MECHANICAL EXTRACTION of HMATSProductionRunner._execute_intent().
All `self.*` references have been replaced with `ctx.*` (ExecutionContext).
Business logic is UNCHANGED — this is identical behavior in a new location.

SHADOW MODE:
    main.py runs BOTH the old _execute_intent AND this execute_intent_v2.
    Results are compared. When identical for 48h, old path is removed.

================================================================================
"""

import asyncio
import json
import logging
import math
import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.canonical_enums import RunMode
from core.constants import get_rule

logger = logging.getLogger(__name__)

# Conditional imports matching main.py pattern
try:
    from core.unit_system import exposure_to_quantity, validate_exposure_fraction
except ImportError:
    exposure_to_quantity = None
    validate_exposure_fraction = None

try:
    from risk.unified_position_sizer import TrancheLevel as UPSTrancheLevel
except ImportError:
    UPSTrancheLevel = None

try:
    from configs.canonical_config import get_drawdown_multiplier as _get_drawdown_multiplier
except ImportError:
    def _get_drawdown_multiplier(dd: float) -> float:
        if dd >= 0.25: return 0.0
        if dd >= 0.15: return 0.25
        if dd >= 0.08: return 0.50
        if dd >= 0.05: return 0.75
        return 1.0

try:
    from execution.execution_manager import OrderType, OrderSide, OrderResult
except ImportError:
    OrderType = OrderSide = OrderResult = None

try:
    from defense.execution_guards import ExecutionMode as GuardMode
except ImportError:
    GuardMode = None

try:
    from execution.execution_quality_logger import ExecutionQualityRecord, ExecutionSource as EQSource
except ImportError:
    ExecutionQualityRecord = EQSource = None

try:
    from execution.impact_calibration import CalibrationSample
except ImportError:
    CalibrationSample = None

try:
    from infra.persistence import TradeExecution as AuditTradeExecution
except ImportError:
    AuditTradeExecution = None

try:
    from configs.canonical_config import (
        EXPOSURE_CAPS, POST_LEVERAGE_CAPS, MIN_REGIME_CONFIDENCE,
        DATA_DIR,
    )
except ImportError:
    EXPOSURE_CAPS = {"BTC": 0.40, "ETH": 0.40, "SOL": 0.50}
    POST_LEVERAGE_CAPS = {"BTC": 0.25, "ETH": 0.25, "SOL": 0.20}
    MIN_REGIME_CONFIDENCE = 0.30
    DATA_DIR = Path("data")

# [FIX-SHADOW] Missing imports that caused NameError in shadow path
try:
    from core.market_data_helpers import effective_volume_ratio
except ImportError:
    def effective_volume_ratio(md):
        return float(md.get("volume_ratio_effective", md.get("volume_ratio", 1.0)) or 1.0)

# P0_MODULES_AVAILABLE: in shadow context, P0 modules are always available
# (shadow only runs when main path succeeded, which requires P0)
P0_MODULES_AVAILABLE = True

try:
    from execution.fees.kraken_plus_fee_blender import get_monthly_volume
    FEE_BLENDING_AVAILABLE = True
except ImportError:
    FEE_BLENDING_AVAILABLE = False
    def get_monthly_volume():
        return 0.0


async def execute_intent_v2(
    ctx,
    asset: str,
    intent: Any,  # TradeIntentV36
    market_data: Dict[str, Any],
    agent_signals: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Execute a trade intent with P0 safety guards.

    P0 FIX (v5.4.1):
    1. Get real account equity (fail-closed)
    2. Convert exposure fraction to base quantity
    3. Validate through authority chain
    4. Check leverage limits
    5. Execute with correct units
    """

    if ctx.config.mode == RunMode.VERIFY:
        return {"status": "SKIPPED", "reason": "Verify mode"}

    # Some forced-exit paths call execution without a full signal bundle.
    agent_signals = agent_signals or {}

    current_price = market_data.get("current_price", 0.0)
    if current_price <= 0:
        return {"status": "REJECTED", "reason": "Invalid price"}

    _existing_position = ctx.paper_positions.get(asset, {})
    _has_active_position = ctx.fn_is_active_paper_position(_existing_position)
    _is_full_exit_request = (intent.target_exposure == 0 or intent.direction == 0)
    if _is_full_exit_request and not _has_active_position:
        return {"status": "SKIPPED", "reason": "No active position to close"}

    _execution_direction = float(getattr(intent, "direction", 0.0) or 0.0)
    if _is_full_exit_request and abs(_execution_direction) < 1e-9 and _has_active_position:
        _position_dir = float(_existing_position.get("direction", 0.0) or 0.0)
        if _position_dir > 0:
            _execution_direction = -1.0
        elif _position_dir < 0:
            _execution_direction = 1.0

    # [BUGFIX AUDIT-C1] Refresh dead-man switch timer before execution
    # Prevents Kraken auto-cancel if tick processing took >60s
    if ctx.dead_man_switch:
        try:
            ctx.dead_man_switch.refresh()
        except Exception as _dms_err:
            logger.warning(f"[DEAD_MAN] Pre-execution refresh failed: {_dms_err}")

    # T19: Pre-execution safety gate (stale data, DVOL, SOL risk, DRL constraint)
    if ctx.execution_guard:
        try:
            _side_str = "long" if intent.direction > 0 else "short"
            _drl_conf = market_data.get("drl_confidence", 0.5)
            # [BUGFIX M7] Fallback=0.0 (not 0.5) -DRL gets zero weight when disabled/unavailable
            _drl_weight = ctx.fn_get_drl_weight() if ctx.fn_get_drl_weight else 0.0
            _guard_ok, _guard_mode, _guard_details = ctx.execution_guard.check_execution(
                asset=asset,
                side=_side_str,
                drl_signal=0.0,
                drl_confidence=_drl_conf,
                drl_weight=_drl_weight,
            )
            if not _guard_ok:
                _block_reasons = ctx.execution_guard.block_reasons
                logger.warning(f"[EXEC_GUARD] {asset} BLOCKED: {_block_reasons}")
                return {"status": "REJECTED", "reason": f"ExecutionGuard: {_block_reasons}"}
            elif _guard_mode != GuardMode.NORMAL:
                logger.info(f"[EXEC_GUARD] {asset}: mode={_guard_mode.name} (non-blocking)")
        except Exception as e:
            logger.debug(f"[EXEC_GUARD] Check failed (non-fatal, allowing): {e}")

    # =====================================================================
    # P0 FIX STEP 1: Get real account equity (FAIL-CLOSED)
    # =====================================================================

    account_equity = 0.0
    if ctx.account_sync:
        try:
            await ctx.account_sync.refresh()
            account_equity = ctx.account_sync.get_equity()  # Raises if unavailable
        except Exception as e:
            logger.critical(f"[P0_FAIL_CLOSED] Account equity unavailable: {e}")
            return {"status": "REJECTED", "reason": f"[P0_FAIL_CLOSED] Equity unavailable: {e}"}
    else:
        # Fallback for backtest mode (no account_sync)
        _fallback_equity = getattr(ctx.config, 'initial_capital', 10_000.0)
        account_equity = market_data.get("account_equity", _fallback_equity)

    # =====================================================================
    # P0 FIX STEP 2: Convert exposure fraction to base quantity
    # =====================================================================

    exposure_fraction = intent.target_exposure

    # [FIX-ABORT-SIZE] For tranche abort (force_execution=True), use the exact
    # abort target bypassing all veto-chain sizing modifiers (REGIME_POWER,
    # WIRE-3, G6-SIZING, WARMUP_SIZE) that inflate/deflate it unpredictably.
    # Those multipliers are designed for new entries, not risk management closes.
    _abort_target = getattr(intent, '_abort_target_exposure', None)
    if getattr(intent, 'force_execution', False) and _abort_target is not None:
        exposure_fraction = _abort_target
        logger.info(
            f"[ABORT_SIZE] {asset}: using abort target={_abort_target:.4f} "
            f"(bypassing sizing modifiers; intent.target_exposure={intent.target_exposure:.4f})"
        )

    _exec_is_warmup = (
        ctx.warmup_tracker.get_tick_count(asset) > 0
        and ctx.warmup_tracker.get_tick_count(asset) <= ctx.warmup_tracker.threshold
    )

    # =====================================================================
    # REGIME LEVERAGE: Apply BEFORE unit conversion (Bug #1 fix)
    # Regime leverage scales the exposure fraction, not the asset quantity.
    # Applying after conversion would break quantity accounting.
    # =====================================================================
    regime_leverage = getattr(intent, 'regime_leverage', 1.0)
    # [FIX-C2] Skip regime leverage for force_execution (abort/close).
    # Abort targets are exact risk-management sizes — multiplying by 2x-3x
    # regime leverage would enlarge a close into a new position.
    _skip_regime_lev = getattr(intent, 'force_execution', False)
    # [REGIME-LEV] SOFT veto ->force 1x (PATCH-4 sets veto_type=SOFT upstream)
    if regime_leverage > 1.0 and market_data.get('veto_type') == 'SOFT':
        logger.info(
            f"[REGIME-LEV] {asset}: SOFT veto active ->leverage "
            f"{regime_leverage:.1f}x ->1x"
        )
        regime_leverage = 1.0
    if regime_leverage > 1.0 and not _skip_regime_lev:
        exposure_fraction *= regime_leverage
        logger.info(
            f"[REGIME_LEVERAGE] Applied {regime_leverage:.1f}x to exposure: "
            f"{intent.target_exposure:.4f} ->{exposure_fraction:.4f}"
        )
    elif regime_leverage > 1.0 and _skip_regime_lev:
        logger.info(
            f"[REGIME_LEVERAGE] Skipped {regime_leverage:.1f}x for force_execution "
            f"(abort target={exposure_fraction:.4f})"
        )

    # P1-1D: Dynamic gross exposure hard check (before per-asset cap)
    if ctx.dynamic_limits_result is not None and ctx.dynamic_limits_result.dynamic_active:
        _gross_existing = sum(
            abs(pos.get("exposure", 0.0))
            for pos in ctx.paper_positions.values()
        )
        _max_gross = ctx.dynamic_limits_result.max_gross_exposure
        _gross_after = _gross_existing + exposure_fraction
        if _gross_after > _max_gross:
            _clamped = max(0.0, _max_gross - _gross_existing)
            logger.warning(
                f"[DYN_GROSS_CAP] {asset}: gross={_gross_existing:.4f}+new={exposure_fraction:.4f}"
                f"={_gross_after:.4f} > max={_max_gross:.2f} ->clamped to {_clamped:.4f}"
            )
            exposure_fraction = _clamped
            if exposure_fraction < 0.01:
                return {"status": "REJECTED", "reason": f"[DYN_GROSS_CAP] gross limit {_max_gross:.2f} reached"}

    # === V10 WIRING: GlobalExposureCap -hard gross limit enforcement ===
    try:
        from risk.global_exposure_cap import get_exposure_cap_manager
        _cap_mgr = get_exposure_cap_manager()

        # Update current positions from paper tracker
        # [FIX-1] _paper_positions uses bare keys ("BTC"), not "BTC/USD"
        _btc_exp = abs(ctx.paper_positions.get("BTC", {}).get("exposure", 0.0))
        _eth_exp = abs(ctx.paper_positions.get("ETH", {}).get("exposure", 0.0))
        _sol_exp = abs(ctx.paper_positions.get("SOL", {}).get("exposure", 0.0))
        _cap_mgr.update_positions(_btc_exp, _eth_exp, _sol_exp)

        _corr_score = market_data.get("cross_asset_correlation", 0.0)  # 0.0 = unknown
        _is_crisis = market_data.get("is_crisis", False)

        _cap_result = _cap_mgr.validate_new_exposure(
            asset=asset.replace("/USD", "").replace("USD", ""),
            requested_exposure_delta=exposure_fraction,
            correlation_score=_corr_score,
            is_crisis=_is_crisis,
        )

        if not _cap_result.is_allowed:
            if _cap_result.adjusted_size > 0.01:
                logger.warning(
                    f"[V10] GlobalExposureCap {asset}: clamped {exposure_fraction:.4f}"
                    f" ->{_cap_result.adjusted_size:.4f} ({_cap_result.adjustment_reason})"
                )
                exposure_fraction = _cap_result.adjusted_size
            else:
                logger.warning(
                    f"[V10] GlobalExposureCap BLOCKED {asset}: gross={_cap_result.current_gross:.2f},"
                    f" cap={_cap_result.cap_in_effect:.2f} ({_cap_result.adjustment_reason})"
                )
                return {"status": "REJECTED", "reason": f"[V10] GlobalExposureCap: {_cap_result.adjustment_reason}"}
    except Exception as e:
        logger.debug(f"[V10] GlobalExposureCap skipped: {e}")
    # === END V10 WIRING ===

    # =====================================================================
    # POST-LEVERAGE SIZE CAP: The SIZE_CAP in process_4h_tick runs BEFORE
    # regime_leverage, so leveraged exposure can exceed per-asset caps.
    # This second cap ensures the final leveraged exposure stays within
    # max_correlated_exposure (60%) budget for a single asset.
    # =====================================================================
    # P1-01: post-leverage caps from config (overridable via --risk-profile)
    # [BUGFIX C4] Hard ceiling: no single asset can exceed 60% regardless of config
    if exposure_fraction > 0.60:
        logger.warning(f"[BUGFIX C4] {asset}: exposure {exposure_fraction:.4f} exceeds 60% hard ceiling -clamped")
        exposure_fraction = 0.60
    lev_cap = ctx.config.post_leverage_caps.get(asset, 0.25)
    if exposure_fraction > lev_cap:
        logger.warning(
            f"[POST_LEV_CAP] {asset}: leveraged exposure clamped "
            f"{exposure_fraction:.4f} ->{lev_cap:.4f} "
            f"(regime_lev={regime_leverage:.1f}x)"
        )
        exposure_fraction = lev_cap

    # L4-17: Reduce position size during warmup (first 2 ticks per asset)
    # Skip for force_execution (abort/risk management) - abort target must not be reduced
    if _exec_is_warmup and exposure_fraction > 0 and not getattr(intent, 'force_execution', False):
        logger.info(f"[WARMUP_SIZE] {asset}: exposure {exposure_fraction:.4f} x0.75")
        exposure_fraction *= 0.75

    # === V10A: CascadeExhaustionGovernor -tranche guard ===
    try:
        from risk.cascade_exhaustion_governor import get_cascade_exhaustion_governor
        _cascade_gov = get_cascade_exhaustion_governor()
        _pos_cascade = ctx.paper_positions.get(asset, {})
        _cur_tranche = _pos_cascade.get("tranche", 0) + 1
        if _cur_tranche >= 2:
            _dir_str = "long" if intent.direction > 0 else "short"
            _tranche_check = _cascade_gov.check_tranche(
                symbol=asset, tranche=_cur_tranche, direction=_dir_str,
            )
            if not _tranche_check.allowed:
                logger.warning(
                    f"[V10A] CascadeGovernor BLOCKED {asset} T{_cur_tranche}: "
                    f"{_tranche_check.reason} (phase={_tranche_check.phase.value})"
                )
                return {
                    "status": "REJECTED",
                    "reason": f"[V10A] CascadeExhaustion: {_tranche_check.reason}",
                }
    except Exception as _e:
        logger.debug(f"[V10A] CascadeGovernor tranche check skipped: {_e}")
    # === END V10A ===

    # [FIX-6] _is_opportunity was only defined in _process_4h_tick_inner scope,
    # causing NameError here that silently disabled UnifiedPositionSizer.
    _is_opportunity = getattr(intent, 'system_mode', '') == 'OPPORTUNITY'

    # W3: UnifiedPositionSizer cap (tranche + drawdown + per-asset)
    if ctx.unified_sizer:
        try:
            _pos = ctx.paper_positions.get(asset, {})
            _existing_tranche = _pos.get("tranche", 0)
            _next_tranche = min(_existing_tranche + 1, 4)
            _tranche_level = UPSTrancheLevel(_next_tranche)
            _gross_exp = sum(
                abs(p.get("exposure", 0)) for p in ctx.paper_positions.values()
            )
            _asset_exp = abs(_pos.get("exposure", 0))
            _dd_status = "NORMAL"
            if ctx.drawdown_tracker is not None:
                _dd = getattr(ctx.drawdown_tracker, 'current_drawdown_pct', 0)
                if _dd > get_rule("hard_drawdown_halt", _is_opportunity):  # [RULETABLE] HARD=0.20
                    _dd_status = "CRITICAL"
                elif _dd > get_rule("reduce_at_drawdown", _is_opportunity):  # [RULETABLE] HARD=0.08
                    _dd_status = "REDUCED"
            # [P2-7] Margin constraint check (constraint-first pattern):
            # Compute available margin BEFORE sizing to prevent margin exhaustion.
            _margin_req = 0.50  # Kraken 2x leverage = 50% margin requirement
            _margin_used = _gross_exp * account_equity if account_equity > 0 else 0.0
            _margin_available = max(0.0, (account_equity / _margin_req) - _margin_used)
            if _margin_available <= 0 and exposure_fraction > 0:
                logger.info(f"[MARGIN_LIMIT] {asset}: margin exhausted "
                            f"(used={_margin_used:.0f}, avail={_margin_available:.0f})")
                exposure_fraction = 0.0

            # [P1-FIX] Dynamic stop_loss_pct for sizing -matches actual stop
            # placement in StopLossAuthority (SOL=3% soft, BTC/ETH=2% soft)
            _sizing_stop_pct = 0.03 if 'SOL' in asset else 0.02
            # When position already has tranches, use get_max_position_for_risk
            # as cap (sizer incremental slice is too small for existing positions)
            if _existing_tranche > 0:
                _risk_max = ctx.unified_sizer.get_max_position_for_risk(
                    account_balance=account_equity,
                    stop_loss_pct=_sizing_stop_pct,
                )
                _sizer_exp = _risk_max / account_equity if account_equity > 0 else 0.0
                # Still apply drawdown multiplier
                _dd_mult = ctx.unified_sizer.config.drawdown_size_multipliers.get(_dd_status, 1.0)
                _sizer_exp *= _dd_mult
            else:
                _sizing = ctx.unified_sizer.calculate_position_size(
                    asset=asset,
                    account_balance=account_equity,
                    stop_loss_pct=_sizing_stop_pct,
                    tranche_level=_tranche_level,
                    current_gross_exposure=_gross_exp,
                    current_asset_exposure=_asset_exp,
                    drawdown_status=_dd_status,
                    price_usd=current_price,
                    is_opportunity=_is_opportunity,  # [RULETABLE]
                    confidence=getattr(intent, 'quant_confidence', 0.5),
                    cross_asset_correlation=market_data.get('correlation_btc_eth_sol', 0.5),
                    annualized_vol=market_data.get('annualized_volatility', 0.6),
                    beta_to_btc=market_data.get('beta_to_btc', 1.0),
                )
                _sizer_exp = _sizing.final_exposure_pct
            if _sizer_exp < exposure_fraction:
                logger.info(
                    f"[SIZER] {asset}: capped {exposure_fraction:.4f} ->"
                    f"{_sizer_exp:.4f} ({_sizing.cap_applied})"
                )
                exposure_fraction = _sizer_exp
        except Exception as e:
            logger.debug(f"[SIZER] Failed: {e}")

    # [CFG-7] Drawdown gradient -continuous size multiplier
    try:
        _dd_pct = getattr(ctx, 'current_drawdown_pct', 0.0)
        _dd_mult = _get_drawdown_multiplier(_dd_pct)
        if _dd_mult < 1.0:
            _dd_old_exp = exposure_fraction
            exposure_fraction *= _dd_mult
            logger.info(
                f"[DD_GRADIENT] {asset}: dd={_dd_pct:.1%} ->mult={_dd_mult:.2f} "
                f"({_dd_old_exp:.4f}->{exposure_fraction:.4f})"
            )
        if _dd_mult <= 0.0:
            return {"status": "KILL_SWITCH", "reason": f"[DD_GRADIENT] drawdown {_dd_pct:.1%} >=kill switch"}
    except Exception as _dd_err:
        logger.debug(f"[DD_GRADIENT] Skipped: {_dd_err}")

    # [P3-5] High Position Low Volume Filter (high-position + low-volume selling/buying exhaustion)
    if ctx.hplv_filter is not None and exposure_fraction > 0:
        try:
            import numpy as _np_hplv
            _hplv_prices = market_data.get("close_history", [])
            _hplv_vol_ratio = effective_volume_ratio(market_data)
            _hplv_dir = int(intent.direction) if hasattr(intent, 'direction') else 0
            if hasattr(_hplv_prices, '__len__') and len(_hplv_prices) >= 20:
                _hplv_result = ctx.hplv_filter.check(
                    current_price=current_price,
                    price_history=_np_hplv.array(_hplv_prices[-100:]),
                    volume_ratio=_hplv_vol_ratio,
                    direction=_hplv_dir,
                    current_exposure=exposure_fraction,
                )
                if _hplv_result.triggered:
                    _old_exp = exposure_fraction
                    exposure_fraction *= _hplv_result.exposure_multiplier
                    logger.info(
                        f"[P3-5 ACTIVE] {asset}: {_hplv_result.pattern} "
                        f"pctile={_hplv_result.price_percentile:.2f} "
                        f"vol_ratio={_hplv_result.volume_ratio:.2f} "
                        f"exposure {_old_exp:.4f}->{exposure_fraction:.4f} "
                        f"(x{_hplv_result.exposure_multiplier:.2f})"
                    )
        except Exception as _hplv_err:
            logger.debug(f"[P3-5] HPLV filter skipped: {_hplv_err}")

    # [G6-lite] Per-asset alpha tilt: Sortino-based exposure adjustment
    if ctx.alpha_tilt:
        _tilt_asset = asset.upper().replace("/USD", "").replace("USD", "")
        _tilt_mult = ctx.alpha_tilt.get_multiplier(_tilt_asset)
        if abs(_tilt_mult - 1.0) > 0.01:
            _old_exp = exposure_fraction
            exposure_fraction *= _tilt_mult
            logger.info(
                f"[G6-lite] {asset}: alpha tilt x{_tilt_mult:.2f} "
                f"({_old_exp:.4f}->{exposure_fraction:.4f})"
            )

    # [FIX-SQUEEZE] SqueezeProtection: reduce short exposure on squeeze risk
    # squeeze_action: NONE / REDUCE_30 / REDUCE_50 / FLATTEN
    _squeeze_action = agent_signals.get('squeeze_action', 'NONE')
    _pos_dir = intent.direction if hasattr(intent, 'direction') else 0
    if _squeeze_action != 'NONE' and _pos_dir < 0 and exposure_fraction > 0:
        _squeeze_mults = {
            'REDUCE_30': 0.70,
            'REDUCE_50': 0.50,
            'FLATTEN': 0.0,
        }
        _sq_mult = _squeeze_mults.get(_squeeze_action, 1.0)
        _old_exp = exposure_fraction
        exposure_fraction *= _sq_mult
        logger.warning(
            f"[FIX-SQUEEZE] {asset}: squeeze_action={_squeeze_action} "
            f"score={agent_signals.get('squeeze_score', 0):.2f} "
            f"SHORT exposure {_old_exp:.4f}->{exposure_fraction:.4f} "
            f"(x{_sq_mult:.2f})"
        )

    # [FIX-H3] Minimum exposure floor after all multipliers.
    # If 10+ stacked multipliers reduce exposure below 0.5% ($50 on $10K),
    # the trade is uneconomical (fees > potential profit). Reject it.
    _MIN_VIABLE_EXPOSURE = 0.005  # 0.5% of equity
    if (exposure_fraction > 0
            and exposure_fraction < _MIN_VIABLE_EXPOSURE
            and not getattr(intent, 'force_execution', False)
            and not _is_full_exit_request):
        logger.info(
            f"[FIX-H3] {asset}: exposure {exposure_fraction:.4f} < "
            f"{_MIN_VIABLE_EXPOSURE} after all multipliers — rejecting "
            f"(uneconomical position size)"
        )
        intent.veto_active = True
        intent.veto_reason = "EXPOSURE_BELOW_MINIMUM_VIABLE"
        ctx.exit_trigger_tag.pop(asset, None)

    if _is_full_exit_request and _has_active_position:
        _close_notional = abs(float(_existing_position.get("notional", 0.0) or 0.0))
        _close_exposure = abs(float(_existing_position.get("exposure", 0.0) or 0.0))
        if _close_exposure > 0:
            exposure_fraction = _close_exposure
        if _close_notional > 0 and current_price > 0:
            base_quantity = _close_notional / current_price
        elif P0_MODULES_AVAILABLE and _close_exposure > 0:
            try:
                base_quantity = float(exposure_to_quantity(
                    exposure_fraction=_close_exposure,
                    account_equity=account_equity,
                    price=current_price,
                    asset=asset,
                ) or 0)
            except Exception as e:
                logger.error(f"[P0_UNIT_SYSTEM] Full-exit conversion failed: {e}")
                return {"status": "REJECTED", "reason": f"[P0_UNIT_SYSTEM] {e}"}
        else:
            return {"status": "REJECTED", "reason": "No active position quantity available"}
    elif P0_MODULES_AVAILABLE:
        try:
            validate_exposure_fraction(exposure_fraction)
            base_quantity = float(exposure_to_quantity(
                exposure_fraction=exposure_fraction,
                account_equity=account_equity,
                price=current_price,
                asset=asset,
            ) or 0)
        except Exception as e:
            logger.error(f"[P0_UNIT_SYSTEM] Conversion failed: {e}")
            return {"status": "REJECTED", "reason": f"[P0_UNIT_SYSTEM] {e}"}
    else:
        # Fallback (should not happen in PAPER/LIVE)
        base_quantity = exposure_fraction  # WARNING: This is wrong but matches old behavior
        logger.warning("[P0_WARNING] Using exposure as quantity (P0 modules unavailable)")

    # [BUGFIX H3] None/zero guard after exposure_to_quantity
    if base_quantity is None or base_quantity <= 0:
        logger.warning(
            f"[BUGFIX H3] Invalid base_quantity={base_quantity} for {asset} "
            f"(exposure={exposure_fraction}, price={current_price}, equity={account_equity})"
        )
        return {"status": "REJECTED", "reason": f"[BUGFIX H3] Invalid quantity: {base_quantity}"}

    notional_usd = base_quantity * current_price

    # =====================================================================
    # Phase 0.5: Rebuild Cooldown Check
    # Prevents churning after recent close -must precede execution
    # Only blocks ENTRY (new position) and ADD-ON, not EXIT or HOLD
    # =====================================================================
    is_new_entry = (
        asset not in ctx.paper_positions
        or ctx.paper_positions[asset].get("notional", 0) < 1.0
    )
    is_adding = (
        not is_new_entry
        and notional_usd > ctx.paper_positions.get(asset, {}).get("notional", 0)
    )

    if (is_new_entry or is_adding) and asset in ctx.rebuild_cooldown:
        _cd_entry = ctx.rebuild_cooldown[asset]
        cooldown_until = _cd_entry[0]
        cooldown_reason = _cd_entry[1]
        _cooldown_closed_dir = _cd_entry[2] if len(_cd_entry) > 2 else 0
        _is_addon = getattr(intent, 'is_addon', False)
        if _is_addon and REBUILD_COOLDOWN_EXEMPT_ADDON:
            # [CFG-9] Pyramid add-on to winning position -exempt from cooldown
            logger.info(
                f"[REBUILD_COOLDOWN] {asset}: add-on EXEMPT (pyramid into winning position)"
            )
        elif (_cooldown_closed_dir != 0 and intent.direction != 0
              and _cooldown_closed_dir * intent.direction < 0):
            # Direction flip -new entry is opposite to closed position.
            # Cooldown is designed to prevent same-direction re-entry churn.
            # Opposite-direction entry is a signal-aligned reversal, allow it.
            del ctx.rebuild_cooldown[asset]
            logger.info(
                f"[REBUILD_COOLDOWN] {asset}: DIRECTION-FLIP EXEMPT "
                f"(closed={'LONG' if _cooldown_closed_dir > 0 else 'SHORT'} "
                f"new={'LONG' if intent.direction > 0 else 'SHORT'}, {cooldown_reason})"
            )
        elif datetime.now(timezone.utc) < cooldown_until:
            remaining = (cooldown_until - datetime.now(timezone.utc)).total_seconds() / 3600
            logger.info(
                f"[REBUILD_COOLDOWN] {asset}: BLOCKED entry/add "
                f"(cooldown {remaining:.1f}h remaining, reason={cooldown_reason})"
            )
            return {
                "status": "COOLDOWN_BLOCKED",
                "reason": f"[REBUILD_COOLDOWN] {cooldown_reason}, {remaining:.1f}h remaining",
                "cooldown_remaining_hours": remaining,
            }
        else:
            # Cooldown expired -clear it
            del ctx.rebuild_cooldown[asset]
            logger.info(f"[REBUILD_COOLDOWN] {asset}: cooldown expired, entry/add allowed")

    # =====================================================================
    # [AC-0] Post-restart new-entry guard
    # First tick after restart: allow exits/management but block NEW entries
    # =====================================================================
    _ac0_reason = ctx.fn_get_ac0_entry_block_reason(asset, is_new_entry)
    if _ac0_reason:
        logger.info(
            f"[AC-0] {asset}: NEW ENTRY BLOCKED -post-restart cooldown "
            f"({_ac0_reason})"
        )
        return {
            "status": "AC0_RESTART_BLOCKED",
            "reason": _ac0_reason,
        }

    # =====================================================================
    # [AC-1] Minimum hold time -block voluntary exit if held < MIN_HOLD_TICKS
    # Safety exits (stop_loss, drawdown_halt, max_hold_timeout) are exempt
    # =====================================================================
    if not is_new_entry and not is_adding:
        # This is an exit or position management
        _ac1_entry_tick = ctx.paper_positions.get(asset, {}).get("entry_tick", 0)
        _ac1_ticks_held = ctx.tick_count - _ac1_entry_tick
        _ac1_exit_reason = getattr(intent, 'veto_reason', '') or ''
        _ac1_safety_exits = {"stop_loss", "drawdown_halt", "p0_safety",
                             "max_hold_timeout", "dead_man_switch", "leverage_guard",
                             "FRICTION_EXCEEDS_EDGE"}
        _ac1_is_safety = any(s in _ac1_exit_reason for s in _ac1_safety_exits)
        if (_ac1_ticks_held < ctx.AC1_MIN_HOLD_TICKS
                and _ac1_entry_tick > 0 and not _ac1_is_safety):
            # Check if this is actually reducing position (exit intent)
            _ac1_target = abs(getattr(intent, 'target_exposure', 0))
            _ac1_current = abs(ctx.paper_positions.get(asset, {}).get("exposure", 0))
            if _ac1_target < _ac1_current * 0.95:  # reducing by >5%
                logger.info(
                    f"[AC-1] {asset}: EXIT BLOCKED -held {_ac1_ticks_held}/{ctx.AC1_MIN_HOLD_TICKS} ticks "
                    f"(entry_tick={_ac1_entry_tick}, current_tick={ctx.tick_count})"
                )
                return {
                    "status": "AC1_MIN_HOLD_BLOCKED",
                    "reason": f"min hold: {_ac1_ticks_held}/{ctx.AC1_MIN_HOLD_TICKS} ticks",
                }

    # =====================================================================
    # [AC-5] Fill budget hard cap
    # =====================================================================
    _ac5_today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if ctx.ac5_fills_date != _ac5_today:
        ctx.ac5_fills_today = 0
        ctx.ac5_fills_date = _ac5_today
    if ctx.ac5_fills_today >= ctx.AC5_MAX_FILLS_PER_DAY:
        logger.warning(
            f"[AC-5] {asset}: FILL BUDGET EXHAUSTED -"
            f"{ctx.ac5_fills_today}/{ctx.AC5_MAX_FILLS_PER_DAY} today"
        )
        return {
            "status": "AC5_BUDGET_EXHAUSTED",
            "reason": f"daily fill budget: {ctx.ac5_fills_today}/{ctx.AC5_MAX_FILLS_PER_DAY}",
        }

    # =====================================================================
    # [AC-2] Anti-churn rate limiter (new entries only)
    # =====================================================================
    if is_new_entry:
        # Per-asset: max 2 fills in last 6 ticks (24h)
        _ac2_recent = ctx.ac2_fill_ticks.get(asset, [])
        _ac2_in_window = sum(1 for t in _ac2_recent if ctx.tick_count - t <= ctx.AC2_WINDOW_TICKS)
        if _ac2_in_window >= ctx.AC2_MAX_FILLS_PER_ASSET:
            logger.info(
                f"[AC-2] {asset}: ENTRY BLOCKED -{_ac2_in_window} fills "
                f"in last {ctx.AC2_WINDOW_TICKS} ticks (limit={ctx.AC2_MAX_FILLS_PER_ASSET})"
            )
            return {
                "status": "AC2_RATE_LIMITED",
                "reason": f"per-asset rate limit: {_ac2_in_window}/{ctx.AC2_MAX_FILLS_PER_ASSET}",
            }
        # Global: max 6 fills in last 6 ticks
        _ac2_global = sum(
            sum(1 for t in fills if ctx.tick_count - t <= ctx.AC2_WINDOW_TICKS)
            for fills in ctx.ac2_fill_ticks.values()
        )
        if _ac2_global >= ctx.AC2_MAX_FILLS_GLOBAL:
            logger.info(
                f"[AC-2] {asset}: ENTRY BLOCKED -global rate limit "
                f"{_ac2_global}/{ctx.AC2_MAX_FILLS_GLOBAL}"
            )
            return {
                "status": "AC2_GLOBAL_RATE_LIMITED",
                "reason": f"global rate limit: {_ac2_global}/{ctx.AC2_MAX_FILLS_GLOBAL}",
            }

    # =====================================================================
    # P0 FIX STEP 3: Authority chain validation
    # Skip for force_execution (tranche abort / risk management / MAX_HOLD_TIMEOUT)
    # These are safety-driven exits that must not be blocked by edge checks.
    # Pattern: same as SOTA-ACT Bug 5 fix and WARMUP_SIZE force_execution bypass.
    # =====================================================================

    if ctx.authority_chain and not getattr(intent, 'force_execution', False):
        chain_result = ctx.authority_chain.evaluate(
            asset=asset,
            direction=intent.direction,
            exposure_fraction=exposure_fraction,
            base_quantity=base_quantity,
            price=current_price,
            account_equity=account_equity,
            regime=market_data.get("regime_state", "UNKNOWN"),
            confidence=getattr(intent, 'quant_confidence', 0.5),
            drl_weight=ctx.fn_get_drl_weight(),
            regime_confidence=market_data.get("regime_confidence", 0.5),
            expected_alpha_bps=getattr(intent, 'alpha_estimated_bps', 60.0),
            alpha_gate_passed=bool(getattr(intent, 'alpha_gate_passed', False)),
            volume_ratio=effective_volume_ratio(market_data),
            market_data_age_seconds=market_data.get("data_age_seconds", market_data.get("data_age_sec", 0.0)),
            market_data_fetched_at_ts=market_data.get("market_data_fetched_at_ts"),
            orderbook_stale=market_data.get("orderbook_stale", False),
            orderbook_fallback_reason=market_data.get("orderbook_fallback_reason", ""),
            orderbook_cache_age_seconds=market_data.get("orderbook_cache_age_seconds"),
        )

        if not chain_result.approved:
            logger.warning(
                f"[AUTHORITY_CHAIN] VETOED: {chain_result.veto_stage.name if chain_result.veto_stage else 'UNKNOWN'} - "
                f"{chain_result.veto_reason}"
            )
            return {
                "status": "REJECTED",
                "reason": f"[AUTHORITY_CHAIN] {chain_result.veto_reason}",
                "veto_stage": chain_result.veto_stage.name if chain_result.veto_stage else None,
            }

    # =====================================================================
    # P0 FIX STEP 4: Leverage guard validation
    # =====================================================================

    if ctx.leverage_guard:
        approved, reason = ctx.leverage_guard.validate_order(
            position_notional_usd=notional_usd,
            account_equity=account_equity,
            current_price=current_price,
            is_short=(intent.direction < 0),
            asset=asset,
        )

        if not approved:
            logger.warning(f"[LEVERAGE_GUARD] REJECTED: {reason}")
            return {
                "status": "REJECTED",
                "reason": f"[LEVERAGE_GUARD] {reason}",
            }

    # =====================================================================
    # [WIRE-4] allow_addon enforcement -block scale-in BEFORE order placement
    # Prevents already-filled scale-ins from being rewritten as ADDON_BLOCKED.
    # =====================================================================
    _w4_pre_pos = ctx.paper_positions.get(asset, {}) or {}
    _w4_pre_same_side = bool(
        _w4_pre_pos
        and float(_w4_pre_pos.get("direction", 0) or 0.0) * float(_execution_direction) > 0
    )
    _w4_pre_old_notional = float(_w4_pre_pos.get("notional", 0.0) or 0.0)
    _w4_pre_is_addon = bool(
        _w4_pre_same_side
        and _w4_pre_old_notional > 0.0
        and notional_usd > (_w4_pre_old_notional + 1e-6)
    )
    if _w4_pre_is_addon:
        _w4_allow_intent = getattr(intent, 'allow_add', True)
        _w4_allow_mce = market_data.get('_macro_crowd_effects', {}).get('allow_addon', True)
        if not _w4_allow_intent or not _w4_allow_mce:
            logger.info(
                f"[WIRE-4] {asset}: SCALE-IN BLOCKED pre-execution -"
                f"intent.allow_add={_w4_allow_intent}, "
                f"mce.allow_addon={_w4_allow_mce}"
            )
            return {
                "status": "skipped",
                "reason": "ADDON_BLOCKED",
                "asset": asset,
            }

    # =====================================================================
    # [IMPACT_ADVISORY] Slicing recommendation for large orders (log-only)
    # Calls production_market_impact.compute_optimal_slicing() for notional >$5K
    # =====================================================================
    if ctx.impact_model and notional_usd > 5000:
        try:
            from execution.production_market_impact import UrgencyLevel as _ImpUrgency
            _side_str = "buy" if intent.direction > 0 else "sell"
            _slicing = ctx.impact_model.compute_optimal_slicing(
                symbol=asset,
                side=_side_str,
                total_size_usd=notional_usd,
                urgency=_ImpUrgency.MEDIUM,
            )
            if _slicing.num_slices > 1:
                logger.info(
                    f"[IMPACT_ADVISORY] {asset}: ${notional_usd:,.0f} ->"
                    f"{_slicing.num_slices} slices, "
                    f"est_impact={getattr(_slicing, 'estimated_total_impact_bps', 0):.1f}bps, "
                    f"order_type={getattr(_slicing, 'order_type', '?')}"
                )
        except Exception as _imp_err:
            logger.debug(f"[IMPACT_ADVISORY] Skipped: {_imp_err}")

    # =====================================================================
    # [FILLSLOPE] Pre-execution guidance from fill quality history
    # Reads recent toxicity rate ->adjusts execution mode/price/size.
    # Runs BEFORE PA executor so guidance can override PA decision.
    # =====================================================================
    _fs_guidance = None
    if ctx.fill_slope_monitor is not None:
        try:
            _fs_guidance = ctx.fill_slope_monitor.get_execution_guidance(asset)
            if _fs_guidance.get('reason'):
                logger.info(
                    f"[FILLSLOPE] {asset}: {_fs_guidance['reason']} "
                    f"(force_passive={_fs_guidance['force_passive']}, "
                    f"extra_improve={_fs_guidance['extra_improve_bps']:.1f}bps, "
                    f"size_reduce={_fs_guidance['size_reduce_pct']:.0%})"
                )
                # [FILLSLOPE] Size reduction: shrink base_quantity pre-PA
                if _fs_guidance['size_reduce_pct'] > 0:
                    _fs_orig = base_quantity
                    base_quantity *= (1.0 - _fs_guidance['size_reduce_pct'])
                    notional_usd = base_quantity * current_price
                    logger.info(
                        f"[FILLSLOPE] {asset}: size reduced {_fs_orig:.6f} ->{base_quantity:.6f} "
                        f"(-{_fs_guidance['size_reduce_pct']:.0%})"
                    )
        except Exception as _fs_err:
            logger.debug(f"[FILLSLOPE] guidance skipped: {_fs_err}")

    # =====================================================================
    # [Section C] PassiveAggressiveExecutor -ACTIVE on hot path
    # Drives order type/price selection. Falls back to legacy logic on failure.
    # =====================================================================
    pa_decision = None
    if ctx.pa_executor:
        try:
            _urgency_map = {0.5: "LOW", 1.0: "NORMAL", 1.2: "HIGH", 1.5: "CRITICAL"}
            _urg_val = getattr(intent, 'execution_urgency', 1.0)
            _urg_str = "NORMAL"
            for threshold, label in sorted(_urgency_map.items()):
                if _urg_val >= threshold:
                    _urg_str = label
            # Extract timing engine output from intent (CHECK-051)
            _timing_score = getattr(intent, 'timing_score', 0.5)
            _timing_exec_mode = getattr(intent, 'execution_mode', None)
            _timing_mode_str = _timing_exec_mode.value if hasattr(_timing_exec_mode, 'value') else ""
            pa_decision = ctx.pa_executor.decide(
                asset=asset,
                signal=intent.direction,
                confidence=getattr(intent, 'quant_confidence', 0.5),
                vpin=market_data.get('vpin', 0.35),
                urgency=_urg_str,
                predicted_alpha_bps=getattr(intent, 'alpha_estimated_bps', 0.0),
                current_price=current_price,
                target_qty=base_quantity,
                timing_score=_timing_score,
                timing_mode=_timing_mode_str,
            )
            _pa_tag = "PA_ACTIVE" if not ctx.pa_executor.shadow_mode else "PA_SHADOW"
            logger.info(
                f"[{_pa_tag}] {asset}: mode={pa_decision.execution_mode.value}, "
                f"improve={pa_decision.price_improvement_bps:.1f}bps, "
                f"aggr={pa_decision.max_aggressive_pct:.0%}, "
                f"reason={pa_decision.reason}"
            )
        except Exception as e:
            logger.warning(f"[PA_EXEC] {asset}: decide() failed, using legacy logic: {e}")
            pa_decision = None

    # =====================================================================
    # P0 FIX STEP 5: Execute with correct base quantity
    # V6.4.1: Order type selection -PA executor drives when active
    # =====================================================================

    side = "BUY" if _execution_direction > 0 else "SELL"

    # PA executor drives order type when active (not shadow) and decision available
    _pa_active = (
        pa_decision is not None
        and ctx.pa_executor is not None
        and not ctx.pa_executor.shadow_mode
    )
    if _pa_active:
        # Map PA execution mode ->order type
        _pa_mode = pa_decision.execution_mode
        from execution.passive_aggressive import ExecutionMode as PAMode
        if _pa_mode == PAMode.ABORT:
            logger.warning(f"[PA_ABORT] {asset}: PA executor says ABORT -skipping execution")
            return {"status": "REJECTED", "reason": "[PA_ABORT] Insufficient edge"}
        elif _pa_mode == PAMode.AGGRESSIVE:
            order_type = "MARKET"
            execution_price = current_price
        elif _pa_mode in (PAMode.PASSIVE, PAMode.HIDDEN):
            order_type = "LIMIT"
            _improve = pa_decision.price_improvement_bps
            if _improve > 0 and _execution_direction > 0:
                execution_price = current_price * (1 - _improve / 10000)
            elif _improve > 0:
                execution_price = current_price * (1 + _improve / 10000)
            else:
                execution_price = current_price
        else:
            # HYBRID/TWAP ->limit at current (simple)
            order_type = "LIMIT"
            execution_price = current_price
    else:
        # Legacy fallback: profit-max parameters
        prefer_limit = getattr(intent, 'prefer_limit_order', False)
        limit_offset_bps = getattr(intent, 'limit_offset_bps', 0.0)
        execution_urgency = getattr(intent, 'execution_urgency', 1.0)

        if prefer_limit or execution_urgency < 0.8:
            order_type = "LIMIT"
            if limit_offset_bps > 0:
                if _execution_direction > 0:
                    execution_price = current_price * (1 - limit_offset_bps / 10000)
                else:
                    execution_price = current_price * (1 + limit_offset_bps / 10000)
            else:
                execution_price = current_price
        elif execution_urgency > 1.2 or intent.execution_mode.value == "AGGRESSIVE":
            order_type = "MARKET"
            execution_price = current_price
        else:
            order_type = "LIMIT" if intent.execution_mode.value != "AGGRESSIVE" else "MARKET"
            execution_price = current_price

    # [P1-PASSIVE] Force LIMIT orders in NORMAL mode for non-exit trades.
    # Paper run: 60% taker execution, fees = 162% of gross profit.
    # NORMAL mode entries should NEVER use MARKET orders — the alpha is
    # not urgent enough to justify 10bps taker premium.
    # Safety exits (TAKER_TRIGGERS) will override this below at L3-06.
    _p1_mode = getattr(intent, 'system_mode', 'NORMAL')
    _p1_is_entry = intent.target_exposure > 0 and not getattr(intent, 'force_execution', False)
    if _p1_mode == 'NORMAL' and _p1_is_entry and order_type == "MARKET":
        order_type = "LIMIT"
        execution_price = current_price
        logger.info(f"[P1-PASSIVE] {asset}: MARKET→LIMIT (NORMAL mode entry, save 10bps)")

    _fast_market_decision = ctx.fn_get_fast_market_execution_decision(intent)
    ctx.dashboard_asset_runtime.setdefault(asset, {}).update({
        "fast_market_execution_applied": False,
        "fast_market_execution_reason": str(
            _fast_market_decision.get("reason", "") or ""
        ),
    })
    if (
        order_type == "LIMIT"
        and not _pa_active
        and bool(_fast_market_decision.get("apply", False))
    ):
        order_type = "MARKET"
        execution_price = current_price
        ctx.dashboard_asset_runtime.setdefault(asset, {}).update({
            "fast_market_execution_applied": True,
            "fast_market_execution_reason": str(
                _fast_market_decision.get("reason", "") or ""
            ),
        })
        logger.info(
            f"[FAST_MARKET_EXEC] {asset}: LIMIT->MARKET | "
            f"{_fast_market_decision.get('reason', 'n/a')}"
        )

    # [L3-06] Exit execution mode: emergency exits use TAKER (market order)
    # Safety-critical exits must not risk hanging as unfilled limit orders.
    _L3_TAKER_TRIGGERS = {
        "T7_MAX_HOLD", "T10_SOFT_STOP", "T1_TRAILING_STOP",
        "T14_P0_FORCE", "FORCE_FLAT", "EMERGENCY",
        "CIRCUIT_BREAKER", "DEAD_MAN_SWITCH",
    }
    _L3_AGGRESSIVE_MAKER_TRIGGERS = {
        "T9_GAMBLER", "T11_RUNNER_RELEASE", "T3_REGIME_EXIT",
        "T3_TAKE_PROFIT", "T3_REGIME_SHIFT",
    }
    _l3_exit_trigger = ctx.exit_trigger_tag.get(asset, "")
    _l3_is_exit = (intent.target_exposure == 0 or intent.direction == 0)
    if _l3_is_exit and _l3_exit_trigger:
        if _l3_exit_trigger in _L3_TAKER_TRIGGERS:
            order_type = "MARKET"
            execution_price = current_price
            logger.info(
                f"[EXIT_MODE] {asset}: TAKER override for safety exit "
                f"(trigger={_l3_exit_trigger})"
            )
        elif _l3_exit_trigger in _L3_AGGRESSIVE_MAKER_TRIGGERS and order_type == "LIMIT":
            logger.info(
                f"[EXIT_MODE] {asset}: AGGRESSIVE_MAKER for {_l3_exit_trigger} "
                f"(LIMIT with market fallback)"
            )

    # [FILLSLOPE] Post-decision override: force LIMIT when toxicity high
    # [L3-06] Emergency exits bypass fillslope passive override
    if _fs_guidance and _fs_guidance.get('force_passive') and order_type == "MARKET":
        if not (_l3_is_exit and _l3_exit_trigger in _L3_TAKER_TRIGGERS):
            order_type = "LIMIT"
        # Add extra improvement to pull away from toxic flow
        _fs_extra = _fs_guidance.get('extra_improve_bps', 0.0)
        if _fs_extra > 0 and _execution_direction > 0:
            execution_price = current_price * (1 - _fs_extra / 10000)
        elif _fs_extra > 0:
            execution_price = current_price * (1 + _fs_extra / 10000)
        logger.info(
            f"[FILLSLOPE] {asset}: MARKET ->LIMIT override "
            f"(improve={_fs_extra:.1f}bps, price=${execution_price:.2f})"
        )

    _execution_trade_side = ctx.fn_resolve_execution_trade_side(asset, intent)
    _exec_advisory_guard = ctx.fn_get_execution_advisory_guard(
        asset, _execution_trade_side
    )
    ctx.dashboard_asset_runtime.setdefault(asset, {}).update(
        {
            "execution_advisory_guard_enabled": bool(
                _exec_advisory_guard.get("enabled", False)
            ),
            "execution_advisory_guard_open": bool(
                _exec_advisory_guard.get("open", False)
            ),
            "execution_advisory_guard_reason": str(
                _exec_advisory_guard.get("reason", "")
            ),
            "execution_advisory_guard_side": str(
                _exec_advisory_guard.get("trade_side", "flat")
            ),
            "execution_advisory_evidence_quality_open": bool(
                _exec_advisory_guard.get("evidence_quality_open", False)
            ),
            "execution_advisory_evidence_quality_reason": str(
                _exec_advisory_guard.get("evidence_quality_reason", "")
            ),
            "execution_advisory_fill_telemetry_count": int(
                _exec_advisory_guard.get("fill_telemetry_count", 0) or 0
            ),
            "execution_advisory_fill_telemetry_verified": bool(
                _exec_advisory_guard.get("telemetry_verified", False)
            ),
            "execution_advisory_session_fills": int(
                _exec_advisory_guard.get("session_fills", 0) or 0
            ),
            "execution_advisory_asset_fills": int(
                _exec_advisory_guard.get("asset_fills", 0) or 0
            ),
            "execution_advisory_side_closed_trades": int(
                _exec_advisory_guard.get("side_closed_trades", 0) or 0
            ),
            "execution_advisory_side_total_realized_pnl_usd": float(
                _exec_advisory_guard.get("side_total_realized_pnl_usd", 0.0) or 0.0
            ),
            "execution_advisory_side_avg_realized_pnl_bps": float(
                _exec_advisory_guard.get("side_avg_realized_pnl_bps", 0.0) or 0.0
            )
            if _exec_advisory_guard.get("side_avg_realized_pnl_bps") is not None
            else 0.0,
            "execution_advisory_side_pnl_sample_ready": bool(
                _exec_advisory_guard.get("side_pnl_sample_ready", False)
            ),
        }
    )

    # [P3-1] Composite Toxicity check (EXECUTION advisory only, no veto)
    if ctx.composite_toxicity is not None:
        try:
            # [CORR-4/VPIN-C] Use neutral 0.35 when VPIN is synthetic
            _ct_vpin_src = market_data.get("vpin_source", "synthetic") if market_data else "synthetic"
            _ct_vpin = market_data.get("vpin", 0.35) if (market_data and _ct_vpin_src == "computed") else 0.35
            _ct_obi = abs(market_data.get("order_book_imbalance", 0.0)) if market_data else 0.0
            _ct_fill_metrics = {}
            if ctx.fill_slope_monitor is not None:
                try:
                    _ct_fill_metrics = ctx.fill_slope_monitor.get_asset_fill_metrics(asset)
                except Exception:
                    _ct_fill_metrics = {}
            _ct_as = float(_ct_fill_metrics.get("adverse_selection_proxy", 0.0) or 0.0)
            _ct_latest_slip = float(_ct_fill_metrics.get("latest_slippage_bps", 0.0) or 0.0)
            _ct_tox_rate = float(_ct_fill_metrics.get("toxicity_rate", 0.0) or 0.0)

            _ct_pr = 0.0
            _ct_last_fill = getattr(ctx, "recent_fill_state", {}).get(asset, {}) or {}
            if _ct_last_fill and current_price > 0:
                try:
                    _ct_fill_ts = float(_ct_last_fill.get("timestamp", 0.0) or 0.0)
                    _ct_fill_price = float(_ct_last_fill.get("fill_price", 0.0) or 0.0)
                    _ct_fill_side = str(_ct_last_fill.get("side", "") or "").lower()
                    _ct_age_sec = time.time() - _ct_fill_ts if _ct_fill_ts > 0 else float("inf")
                    if _ct_fill_price > 0 and _ct_age_sec <= 6 * 3600:
                        if _ct_fill_side == "buy":
                            _ct_adverse_bps = max(
                                0.0,
                                ((_ct_fill_price - current_price) / _ct_fill_price) * 10000,
                            )
                        elif _ct_fill_side == "sell":
                            _ct_adverse_bps = max(
                                0.0,
                                ((current_price - _ct_fill_price) / _ct_fill_price) * 10000,
                            )
                        else:
                            _ct_adverse_bps = 0.0
                        _ct_pr_scale = max(
                            75.0,
                            float(market_data.get("spread_bps", 10.0) or 10.0) * 6.0,
                        )
                        _ct_pr = min(1.0, _ct_adverse_bps / _ct_pr_scale)
                except Exception:
                    _ct_pr = 0.0
            _ct_result = ctx.composite_toxicity.compute(
                adverse_selection=_ct_as,
                vpin=_ct_vpin,
                flow_imbalance=_ct_obi,
                price_reversal=_ct_pr,
                asset=asset,
                side=_execution_trade_side,
            )
            ctx.dashboard_asset_runtime.setdefault(asset, {}).update({
                "composite_toxicity_authority": "EXECUTION",
                "composite_toxicity_score": float(_ct_result.score or 0.0),
                "composite_toxicity_warn": bool(_ct_result.should_warn),
                "composite_toxicity_veto": bool(_ct_result.should_veto),
                "composite_toxicity_dominant": str(_ct_result.dominant_dimension or "none"),
                "composite_toxicity_profile": str(_ct_result.profile_key or "DEFAULT:FLAT"),
                "composite_toxicity_warn_threshold": float(_ct_result.warn_threshold or 0.0),
                "composite_toxicity_veto_threshold": float(_ct_result.veto_threshold or 0.0),
                "composite_toxicity_applied": False,
                "composite_toxicity_adverse_selection": float(_ct_as),
                "composite_toxicity_price_reversal": float(_ct_pr),
                "composite_toxicity_toxicity_rate": float(_ct_tox_rate),
                "composite_toxicity_latest_slippage_bps": float(_ct_latest_slip),
            })
            if _ct_result.should_warn and bool(_exec_advisory_guard.get("open", False)):
                _ct_old_urg = float(getattr(intent, 'execution_urgency', 1.0) or 1.0)
                intent.execution_urgency = max(0.6, _ct_old_urg * 0.75)
                intent.prefer_limit_order = True
                ctx.dashboard_asset_runtime.setdefault(asset, {}).update({
                    "composite_toxicity_applied": True,
                })
                logger.info(
                    f"[P3-1 EXECUTION] {asset}: toxicity={_ct_result.score:.3f} "
                    f"dominant={_ct_result.dominant_dimension} "
                    f"urgency {_ct_old_urg:.2f}->{intent.execution_urgency:.2f} "
                    f"prefer_limit=True"
                )
            elif _ct_result.should_warn:
                logger.info(
                    f"[P3-1 EXECUTION] {asset}: toxicity warn observed but guard closed "
                    f"({str(_exec_advisory_guard.get('reason', ''))})"
                )
        except Exception as _ct_err:
            logger.debug(f"[P3-1] Toxicity check skipped: {_ct_err}")

    # P2-02: Determine taker_allowed from fee tier context
    _taker_allowed = True
    _fc_local = market_data.get('fee_context') if market_data else None
    if _fc_local and not getattr(intent, 'force_execution', False):
        _in_free = _fc_local.get("in_free_tier", True)
        _tc = _fc_local.get("tier_config", {})
        if _tc.get("enable_tier_aware_timing", False) and not _in_free:
            _taker_allowed = False

    # T26: Learned Execution Policy advisory (MARKET->LIMIT downgrade only)
    if ctx.learned_exec_policy:
        try:
            _lep_direction_before = float(getattr(intent, "direction", 0.0) or 0.0)
            _lep_exposure_before = float(getattr(intent, "target_exposure", 0.0) or 0.0)
            _lep_veto_before = bool(getattr(intent, "veto_active", False))
            _lep_veto_reason_before = getattr(intent, "veto_reason", None)
            _lep_lob = LEPLOBFeatures(
                spread_bps=market_data.get("spread_bps", 0.0),
                imbalance=market_data.get("order_book_imbalance", 0.0),
                bid_depth_usd=market_data.get("orderbook_depth_1pct_usd", 500000),
                ask_depth_usd=market_data.get("orderbook_depth_1pct_usd", 500000),
                vpin=market_data.get("vpin", 0.35),
            )
            _lep_side = "buy" if _execution_direction > 0 else "sell"
            _lep_state = LEPExecState(
                target_quantity=base_quantity,
                executed_quantity=0.0,
                side=_lep_side,
                arrival_price=current_price,
                current_mid_price=current_price,
                deadline_sec=14400,
            )
            _lep_advice = ctx.learned_exec_policy.get_advice(
                _lep_lob, None, _lep_state,
            )
            _lep_status = ctx.fn_get_learned_exec_runtime_snapshot()
            ctx.dashboard_asset_runtime.setdefault(asset, {}).update({
                "learned_exec_authority": "EXECUTION",
                "learned_exec_mode": str(ctx.learned_exec_policy.config.mode or "advisory").upper(),
                "learned_exec_effective_mode": str(_lep_status.get("effective_mode", "DISABLED") or "DISABLED"),
                "learned_exec_action": str(_lep_advice.recommended_action.value),
                "learned_exec_confidence": float(_lep_advice.confidence or 0.0),
                "learned_exec_aggressiveness": float(_lep_advice.aggressiveness or 0.0),
                "learned_exec_limit_offset_bps": float(_lep_advice.limit_offset_bps or 0.0),
                "learned_exec_applied": False,
                "learned_exec_weights_loaded": bool(_lep_status.get("weights_loaded", False)),
                "learned_exec_heuristic_only": bool(_lep_status.get("heuristic_only", False)),
                "learned_exec_network_usable": bool(_lep_status.get("network_usable", False)),
                "learned_exec_manifest_required": bool(_lep_status.get("validated_manifest_required", True)),
                "learned_exec_manifest_loaded": bool(_lep_status.get("validated_manifest_loaded", False)),
                "learned_exec_manifest_path": str(_lep_status.get("validated_manifest_path", "") or ""),
                "learned_exec_manifest_error": str(_lep_status.get("validated_manifest_error", "") or ""),
                "learned_exec_last_advice_source": str(_lep_status.get("last_advice_source", "") or ""),
                "learned_exec_last_advice_reason": str(_lep_status.get("last_advice_reason", "") or ""),
            })
            # Execution authority only: may alter execution style, never entry direction/veto.
            # [L3-06] Never downgrade emergency taker exits
            _l3_lep_blocked = (_l3_is_exit and _l3_exit_trigger in _L3_TAKER_TRIGGERS)
            if not _l3_lep_blocked and bool(_exec_advisory_guard.get("open", False)):
                if _lep_advice.recommended_action == LEPAction.WAIT:
                    _lep_old_urg = float(getattr(intent, 'execution_urgency', 1.0) or 1.0)
                    intent.execution_urgency = max(0.6, _lep_old_urg * 0.75)
                    intent.prefer_limit_order = True
                    ctx.dashboard_asset_runtime.setdefault(asset, {}).update({
                        "learned_exec_applied": True,
                    })
                elif _lep_advice.recommended_action in (LEPAction.LIMIT_PASSIVE, LEPAction.LIMIT_AGGRESSIVE):
                    intent.prefer_limit_order = True
                    _lep_offset = float(_lep_advice.limit_offset_bps or 0.0)
                    if _lep_offset > 0:
                        _curr_offset = float(getattr(intent, 'limit_offset_bps', 0.0) or 0.0)
                        intent.limit_offset_bps = max(_curr_offset, _lep_offset)
                    if _lep_advice.recommended_action == LEPAction.LIMIT_AGGRESSIVE:
                        _lep_old_urg = float(getattr(intent, 'execution_urgency', 1.0) or 1.0)
                        intent.execution_urgency = max(_lep_old_urg, 1.1)
                    else:
                        _lep_old_urg = float(getattr(intent, 'execution_urgency', 1.0) or 1.0)
                        intent.execution_urgency = min(_lep_old_urg, 0.9)
                    ctx.dashboard_asset_runtime.setdefault(asset, {}).update({
                        "learned_exec_applied": True,
                    })
            elif not _l3_lep_blocked and _lep_advice.recommended_action != LEPAction.MARKET:
                logger.info(
                    f"[EXEC_POLICY] {asset}: advice observed but guard closed "
                    f"({str(_exec_advisory_guard.get('reason', ''))})"
                )
            if (
                order_type == "MARKET"
                and not _l3_lep_blocked
                and bool(_exec_advisory_guard.get("open", False))
                and _lep_advice.recommended_action in (
                LEPAction.WAIT, LEPAction.LIMIT_PASSIVE, LEPAction.LIMIT_AGGRESSIVE,
                )
            ):
                order_type = "LIMIT"
                logger.info(
                    f"[EXEC_POLICY] {asset}: MARKET->LIMIT "
                    f"(spread={market_data.get('spread_bps', 0):.1f}bps, "
                    f"advice={_lep_advice.recommended_action.value})"
                )
            else:
                logger.debug(
                    f"[EXEC_POLICY] {asset}: {_lep_advice.recommended_action.value} "
                    f"(order_type={order_type} unchanged)"
                )
            if (
                abs(float(getattr(intent, "direction", 0.0) or 0.0) - _lep_direction_before) > 1e-9
                or abs(float(getattr(intent, "target_exposure", 0.0) or 0.0) - _lep_exposure_before) > 1e-9
                or bool(getattr(intent, "veto_active", False)) != _lep_veto_before
            ):
                logger.error(
                    f"[EXEC_POLICY] {asset}: advisory contract violation "
                    f"(dir={getattr(intent, 'direction', 0.0)} exp={getattr(intent, 'target_exposure', 0.0)} "
                    f"veto={getattr(intent, 'veto_active', False)}) -> restoring pre-LEP state"
                )
                intent.direction = _lep_direction_before
                intent.target_exposure = _lep_exposure_before
                intent.veto_active = _lep_veto_before
                intent.veto_reason = _lep_veto_reason_before
        except Exception as e:
            logger.debug(f"[EXEC_POLICY] Advisory failed: {e}")

    # Orderbook execution recommendation (advisory log)
    if ctx.orderbook_analyzer:
        try:
            _ob_sym = f"{asset}/USD"
            _ob_side = "buy" if _execution_direction > 0 else "sell"
            _ob_size = abs(notional_usd)
            _ob_rec = ctx.orderbook_analyzer.get_execution_recommendation(
                symbol=_ob_sym, side=_ob_side, size_usd=_ob_size,
                urgency="high" if getattr(intent, 'execution_urgency', 0.5) > 0.7 else "medium",
            )
            if _ob_rec.warnings:
                logger.info(f"[ORDERBOOK] {asset}: mode={_ob_rec.execution_mode} slippage~{_ob_rec.expected_slippage_bps:.1f}bps warnings={_ob_rec.warnings}")
            if not _ob_rec.should_execute:
                logger.warning(f"[ORDERBOOK] {asset}: NOT recommended -{_ob_rec.reasoning}")
        except Exception:
            pass

    # T25: Level2 orderbook analysis -thin liquidity size cut (new entries only)
    if ctx.level2_analyzer and ctx.integrity_shield:
        _existing_pos = ctx.paper_positions.get(asset)
        _is_new_entry = not _existing_pos or _existing_pos.get("exposure", 0.0) == 0.0
        if _is_new_entry:
            try:
                _canonical_pair = ctx.fn_normalize_kraken_pair(asset)
                _ob = ctx.integrity_shield.get_orderbook(_canonical_pair) if _canonical_pair else None
                if _ob and _ob.bids and _ob.asks:
                    _l2_bids = [(float(b.price), float(b.quantity)) for b in _ob.bids[:20]]
                    _l2_asks = [(float(a.price), float(a.quantity)) for a in _ob.asks[:20]]
                    ctx.level2_analyzer.update_orderbook(asset, _l2_bids, _l2_asks)
                    _l2_result = ctx.level2_analyzer.analyze_depth(asset)
                    if _l2_result and _l2_result.liquidity_level.value in ("thin", "dangerous"):
                        logger.warning(
                            f"[L2] {asset}: {_l2_result.liquidity_level.value.upper()} liquidity -"
                            f"depth_bid=${_l2_result.bid_depth_1pct:,.0f} "
                            f"depth_ask=${_l2_result.ask_depth_1pct:,.0f} "
                            f"spread={_l2_result.spread_bps:.1f}bps"
                        )
                        if notional_usd > _l2_result.bid_depth_1pct * 0.5:
                            _old_qty = base_quantity
                            base_quantity *= 0.5
                            notional_usd = base_quantity * current_price
                            logger.warning(
                                f"[L2] {asset}: SIZE CUT 50% due to thin liquidity "
                                f"({_old_qty:.6f} ->{base_quantity:.6f})"
                            )
            except Exception as e:
                logger.debug(f"[L2] Analysis failed: {e}")

    # W7: RL Execution Agent advisory (rule-based action from orderbook)
    if ctx.rl_exec_agent and ctx.integrity_shield:
        try:
            _canonical_pair = ctx.fn_normalize_kraken_pair(asset)
            _ob = ctx.integrity_shield.get_orderbook(_canonical_pair) if _canonical_pair else None
            if _ob and _ob.bids and _ob.asks:
                _agent_ob = AgentOBSnapshot(
                    bid_prices=[float(b.price) for b in _ob.bids[:5]],
                    bid_quantities=[float(b.quantity) for b in _ob.bids[:5]],
                    ask_prices=[float(a.price) for a in _ob.asks[:5]],
                    ask_quantities=[float(a.quantity) for a in _ob.asks[:5]],
                )
                _agent_ob.calculate_metrics()
                _decision = ctx.rl_exec_agent.decide(
                    orderbook=_agent_ob,
                    order_size=base_quantity,
                    side="buy" if _execution_direction > 0 else "sell",
                    urgency=0.5,
                )
                logger.debug(
                    f"[EXEC_AGENT] {asset}: {_decision.action.value} "
                    f"slice={_decision.slice_pct:.0%} "
                    f"offset={_decision.limit_offset_bps:.1f}bps "
                    f"({_decision.reason})"
                )
                # [WIRE-5] Consume RL agent decision: override order params
                # [L3-06] Never downgrade emergency taker exits
                from execution.rl_execution_agent import ExecutionAction as _RLAction
                _l3_rl_blocked = (_l3_is_exit and _l3_exit_trigger in _L3_TAKER_TRIGGERS)
                if _decision.action in (_RLAction.PASSIVE, _RLAction.LIMIT) and order_type == "MARKET" and not _l3_rl_blocked:
                    order_type = "LIMIT"
                    if _decision.limit_offset_bps > 0:
                        if _execution_direction > 0:
                            execution_price = current_price * (1 - _decision.limit_offset_bps / 10000)
                        else:
                            execution_price = current_price * (1 + _decision.limit_offset_bps / 10000)
                    logger.info(
                        f"[WIRE-5] {asset}: RL agent MARKET->LIMIT override "
                        f"offset={_decision.limit_offset_bps:.1f}bps"
                    )
                elif _decision.action == _RLAction.MARKET and order_type == "LIMIT":
                    if getattr(intent, 'execution_urgency', 0.5) > 0.7:
                        order_type = "MARKET"
                        execution_price = current_price
                        logger.info(f"[WIRE-5] {asset}: RL agent LIMIT->MARKET (urgency)")
        except Exception as e:
            logger.debug(f"[EXEC_AGENT] Failed: {e}")

    # [SOTA] FillRateKPI -adjust size for low-fill-rate strategies
    try:
        if ctx.fill_rate_kpi:
            _sota_strat = getattr(intent, 'quant_strategy_id', None) or "unknown"
            _sota_advice = ctx.fill_rate_kpi.get_advice(_sota_strat)
            if _sota_advice.action == "REDUCE_SIZE":
                _sota_old_qty = base_quantity
                base_quantity *= 0.70
                notional_usd *= 0.70
                logger.info(
                    f"[SOTA] FillRate REDUCE_SIZE {_sota_strat}/{asset}: "
                    f"rate={_sota_advice.fill_rate:.1%}, qty {_sota_old_qty:.6f}->{base_quantity:.6f}"
                )
            elif _sota_advice.action == "WIDEN_PRICE":
                logger.info(
                    f"[SOTA] FillRate WIDEN_PRICE {_sota_strat}/{asset}: "
                    f"rate={_sota_advice.fill_rate:.1%} -PA executor may widen"
                )
    except Exception as _sota_err:
        logger.debug(f"[SOTA] FillRateKPI advice skipped: {_sota_err}")

    # [AUDIT M3] Kraken minimum order size floor -reject if reductions pushed below min
    _m3_min_sizes = {"BTC": 0.0001, "ETH": 0.004, "SOL": 0.02}
    _m3_min = _m3_min_sizes.get(asset, 0.01)
    if base_quantity < _m3_min:
        logger.warning(
            f"[AUDIT M3] {asset}: quantity {base_quantity:.8f} below Kraken min {_m3_min} -skipping"
        )
        return {"status": "REJECTED", "reason": f"[AUDIT M3] Below Kraken min: {base_quantity:.8f} < {_m3_min}"}

    _micro_rebalance_block = ctx.fn_maybe_block_micro_rebalance(
        asset=asset,
        intent=intent,
        current_price=current_price,
        target_notional_usd=notional_usd,
        order_type=order_type,
        market_data=market_data,
    )
    if _micro_rebalance_block is not None:
        return _micro_rebalance_block

    # DynamicSlicer: ATR-based slice count for execution quality
    _atr_ratio = market_data.get('atr_ratio', 1.0)
    _vpin_val = market_data.get('vpin', 0.35)
    _vpin_src_slicer = market_data.get('vpin_source', 'synthetic')
    # === V10S: ATR extreme volatility + toxic flow guard ===
    # [VPIN-C] Only apply when VPIN is computed (not synthetic)
    if _vpin_src_slicer == 'computed' and _atr_ratio > 4.0 and _vpin_val > 0.80:
        logger.warning(
            f"[V10S] DynamicSlicer ABORT {asset}: ATR={_atr_ratio:.2f} + VPIN={_vpin_val:.2f} "
            f"-extreme volatility + toxic flow, refusing execution"
        )
        return {"status": "REJECTED", "reason": f"[V10S] DynamicSlicer: ATR={_atr_ratio:.1f}+VPIN={_vpin_val:.2f}"}
    # === END V10S guard ===
    if _atr_ratio < 0.8:
        _num_slices = 3
    elif _atr_ratio < 1.5:
        _num_slices = 5
    elif _atr_ratio < 2.5:
        _num_slices = 10
    elif _atr_ratio < 4.0:
        _num_slices = 15
    else:
        _num_slices = 3  # Extreme vol: fewer larger slices
    if _vpin_src_slicer == 'computed' and _vpin_val > 0.70:
        _num_slices = max(_num_slices, 10)
    # V10S: Asset liquidity adjustment (SOL less liquid)
    _asset_key = asset.upper().replace("/USD", "").replace("USD", "")
    _liq_mult = {"BTC": 0.8, "ETH": 1.0, "SOL": 1.3}.get(_asset_key, 1.0)
    _num_slices = max(1, int(_num_slices * _liq_mult))
    # [SOTA] ImpactCalibration ->DynamicSlicer: high impact ->more slices
    try:
        if ctx.impact_cal_table:
            _ic_hour = datetime.now(timezone.utc).hour
            _ic_session = (
                "ASIA" if _ic_hour < 8 else
                "EUROPE" if _ic_hour < 14 else
                "US" if _ic_hour < 22 else "ASIA"
            )
            _ic_vol_bkt = (
                "LOW" if _atr_ratio < 0.8 else
                "MEDIUM" if _atr_ratio < 1.5 else
                "HIGH" if _atr_ratio < 3.0 else "EXTREME"
            )
            _ic_sprd = market_data.get("spread_bps", 10.0)
            _ic_sprd_bkt = (
                "TIGHT" if _ic_sprd < 5 else
                "NORMAL" if _ic_sprd < 15 else
                "WIDE" if _ic_sprd < 30 else "VERY_WIDE"
            )
            _ic_eta, _ic_gamma, _ic_conf = ctx.impact_cal_table.get_params(
                asset=_asset_key, session=_ic_session,
                volatility_bucket=_ic_vol_bkt, spread_bucket=_ic_sprd_bkt,
            )
            if _ic_conf > 0.3 and _ic_eta > 0.2:
                _ic_extra = max(0, int(_ic_eta * 10))  # eta=0.5 ->+5 slices
                _num_slices = max(_num_slices, _num_slices + _ic_extra)
                logger.info(
                    f"[SOTA] ImpactCal->Slicer {asset}: eta={_ic_eta:.3f} conf={_ic_conf:.2f} "
                    f"->+{_ic_extra} slices (total={_num_slices})"
                )
    except Exception as _ic_err:
        logger.debug(f"[SOTA] ImpactCal->Slicer skipped: {_ic_err}")
    logger.info(
        f"[DYN_SLICER] {asset}: slices={_num_slices} "
        f"atr_ratio={_atr_ratio:.2f} vpin={_vpin_val:.2f}"
    )

    # [P2-FIX] SOL Execution Guard -pre-trade safety check (SHADOW)
    if ctx.sol_exec_guard and 'SOL' in asset:
        try:
            _seg_result = ctx.sol_exec_guard.check_execution_safe(
                order_size_sol=base_quantity,
                orderbook_depth_usd=market_data.get('orderbook_depth_1pct_usd', 0),
                spread_bps=market_data.get('spread_bps', 0),
            )
            if not _seg_result['safe']:
                base_quantity = _seg_result['recommended_size']
                logger.warning(
                    f"[P2-FIX] SOL guard clipped: {_seg_result['reason']}"
                )
        except Exception as _seg_err:
            logger.debug(f"[P2-FIX] SOL guard skipped: {_seg_err}")

    # [BUGFIX H1] Pass _num_slices to execution -slice large orders
    _h1_slice_size = base_quantity / _num_slices if _num_slices > 1 else base_quantity
    if _num_slices > 1 and ctx.config.mode != RunMode.PAPER:
        # Live/non-paper: execute in slices with delay
        import asyncio as _h1_asyncio
        _h1_total_filled = 0.0
        _h1_last_result = None
        _h1_initial_price = execution_price  # [FIX-L2-05] anchor for drift detection
        for _h1_i in range(_num_slices):
            # [FIX-L2-05] Price drift check before each slice (except first)
            if _h1_i > 0:
                try:
                    _h1_cur_md = await ctx.fn_prepare_market_data(asset)
                    _h1_cur_price = _h1_cur_md.get('current_price', execution_price)
                    if _h1_initial_price > 0 and _h1_cur_price > 0:
                        _h1_drift_bps = abs(_h1_cur_price - _h1_initial_price) / _h1_initial_price * 10000
                        if _h1_drift_bps > 30:
                            logger.warning(
                                f"[SLICER_ABORT] {asset}: price drifted {_h1_drift_bps:.0f}bps "
                                f"({_h1_initial_price:.2f} -> {_h1_cur_price:.2f}) "
                                f"-- aborting remaining {_num_slices - _h1_i}/{_num_slices} slices"
                            )
                            break
                        if _h1_drift_bps > 10:
                            _h1_remaining_qty = base_quantity - _h1_total_filled
                            _h1_remaining_slices = _num_slices - _h1_i
                            _h1_slice_size = _h1_remaining_qty / _h1_remaining_slices if _h1_remaining_slices > 0 else _h1_remaining_qty
                            execution_price = _h1_cur_price
                            logger.info(
                                f"[SLICER_DRIFT] {asset}: {_h1_drift_bps:.0f}bps drift -- "
                                f"recalculated slice_size={_h1_slice_size:.6f} at price={_h1_cur_price:.2f}"
                            )
                except Exception as _h1_drift_err:
                    logger.debug(f"[SLICER_DRIFT] {asset}: drift check skipped: {_h1_drift_err}")

            _h1_last_result = ctx.execution_manager.execute_order(
                symbol=f"{asset}/USD",
                side=side,
                size=_h1_slice_size,
                price=execution_price,
                order_type=order_type,
                leverage=int(round(regime_leverage)) if regime_leverage > 1.0 else None,
                spread_bps=market_data.get('spread_bps', 10.0),
                tick_id=ctx.tick_count,
                taker_allowed=_taker_allowed,
                vpin=market_data.get('vpin', 0.35),
                bid_depth_usd=market_data.get('bid_depth_usd', 50_000.0),
                ask_depth_usd=market_data.get('ask_depth_usd', 50_000.0),
            )
            _h1_filled = getattr(_h1_last_result, 'filled_size', _h1_slice_size)
            _h1_total_filled += _h1_filled
            if not getattr(_h1_last_result, 'success', True):
                logger.warning(f"[BUGFIX H1] Slice {_h1_i+1}/{_num_slices} failed for {asset} -stopping")
                break
            if _h1_i < _num_slices - 1:
                await _h1_asyncio.sleep(0.3)
        result = _h1_last_result
        # [AUDIT H1] Store aggregated fill back to result so downstream sees total
        if result is not None and hasattr(result, 'filled_size'):
            result.filled_size = _h1_total_filled
        logger.info(f"[BUGFIX H1] {asset}: {_num_slices} slices, total filled={_h1_total_filled:.6f}")
    else:
        # Paper mode or single slice: execute as one order
        result = ctx.execution_manager.execute_order(
            symbol=f"{asset}/USD",
            side=side,
            size=base_quantity,
            price=execution_price,
            order_type=order_type,
            leverage=int(round(regime_leverage)) if regime_leverage > 1.0 else None,
            spread_bps=market_data.get('spread_bps', 10.0),
            tick_id=ctx.tick_count,
            taker_allowed=_taker_allowed,
            vpin=market_data.get('vpin', 0.35),
            bid_depth_usd=market_data.get('bid_depth_usd', 50_000.0),
            ask_depth_usd=market_data.get('ask_depth_usd', 50_000.0),
        )

    # Log execution details
    leverage_effective = notional_usd / account_equity if account_equity > 0 else 0
    logger.info(
        f"[P0_EXECUTE] {asset} {side} qty={base_quantity:.6f} @ ${current_price:.2f} "
        f"notional=${notional_usd:,.2f} leverage={leverage_effective:.2f}x "
        f"(regime_leverage={regime_leverage:.1f}x)"
    )

    # P2-02: Proof log for fee-tier-aware execution
    if _fc_local:
        _tier_tag = "FREE" if _fc_local.get("in_free_tier") else "POST"
        _tier_adj = getattr(intent, 'timing_score', None)
        # Check if tier adjustment was applied via timing score object
        _timing_score_obj = getattr(ctx.engine, '_last_timing_score', None) if ctx.engine is not None else None
        _adj_str = ""
        if hasattr(intent, 'execution_mode'):
            _adj_str = f" exec_mode={intent.execution_mode.value if hasattr(intent.execution_mode, 'value') else intent.execution_mode}"
        logger.info(
            f"[FEE_TIER_TIMING] {asset}: tier={_tier_tag} "
            f"vol=${_fc_local.get('monthly_volume_usd', 0):,.0f} "
            f"weight={_fc_local.get('fee_weight', 0):.2f} "
            f"taker_allowed={_taker_allowed} "
            f"order_type={order_type}{_adj_str}"
        )

    # P2-03: PA execution quality proof log
    if _pa_active and pa_decision is not None:
        logger.info(
            f"[PA_PROOF] {asset}: mode={pa_decision.execution_mode.value} "
            f"improve={pa_decision.price_improvement_bps:.1f}bps "
            f"order_type={order_type} taker_allowed={_taker_allowed} "
            f"edge_ok={pa_decision.edge_sufficient} "
            f"friction={pa_decision.total_friction_bps:.1f}bps "
            f"reason={pa_decision.reason}"
        )

    exec_result = result.to_dict() if hasattr(result, 'to_dict') else {"status": "EXECUTED"}
    exec_result["p0_details"] = {
        "account_equity": account_equity,
        "exposure_fraction": exposure_fraction,
        "base_quantity": base_quantity,
        "notional_usd": notional_usd,
        "leverage": leverage_effective,
        "regime_leverage": regime_leverage,
    }
    exec_result["shadow_fill_recorded"] = False
    exec_result["shadow_fill_count_delta"] = 0
    exec_result["paper_fee_cost_pnl_delta"] = 0.0

    def _note_shadow_fill(recorded: bool) -> None:
        if not recorded:
            return
        exec_result["shadow_fill_recorded"] = True
        exec_result["shadow_fill_count_delta"] = int(
            exec_result.get("shadow_fill_count_delta", 0) or 0
        ) + 1

    # [BUGFIX C2] Execution success check -do NOT update positions on failure
    _c2_success = exec_result.get("success", True)  # default True for backward compat
    _c2_status = str(exec_result.get("status", "")).upper()
    if _c2_success is False or _c2_status in ("FAILED", "REJECTED", "EXPIRED", "CANCELLED"):
        logger.warning(
            f"[BUGFIX C2] Execution FAILED for {asset}: "
            f"success={_c2_success}, status={_c2_status}, "
            f"error={exec_result.get('error_message', 'unknown')} "
            f"-skipping position update and fee recording"
        )
        exec_result["p0_details"]["bugfix_c2_blocked"] = True
        # [SOTA] Record failed fill for FillRateKPI
        try:
            if ctx.fill_rate_kpi:
                _sota_strat = getattr(intent, 'quant_strategy_id', None) or "unknown"
                ctx.fill_rate_kpi.record_order(
                    strategy=_sota_strat, placed=True, filled=False, fill_pct=0.0, asset=asset,
                )
        except Exception:
            pass
        return exec_result

    # =====================================================================
    # Phase 1.9b: Update paper position tracker
    # Three-branch logic: FULL EXIT / PARTIAL EXIT / ENTRY+SCALE-IN
    # Branch selection based on exposure change direction, not just target==0
    # =====================================================================
    # Use actual fill price (with slippage) instead of raw market price
    fill_price = exec_result.get("filled_price", current_price)
    if fill_price <= 0:
        fill_price = current_price

    # [BUGFIX C5] Use actual filled_size for position accounting, not requested base_quantity
    _c5_filled_size = exec_result.get("filled_size", base_quantity)
    if _c5_filled_size is not None and _c5_filled_size > 0 and _c5_filled_size != base_quantity:
        _c5_old_notional = notional_usd
        notional_usd = _c5_filled_size * fill_price
        base_quantity = _c5_filled_size
        logger.info(
            f"[BUGFIX C5] {asset}: partial fill -adjusted notional "
            f"${_c5_old_notional:,.0f} ->${notional_usd:,.0f} "
            f"(filled={_c5_filled_size:.6f}/{exec_result.get('requested_size', 0):.6f})"
        )

    # =====================================================================
    # Phase 1.9: Record execution fee truth on actual filled notional
    # Spot fee blending only applies to unlevered non-short paths.
    # Shorts / leveraged paper trades must use margin-style fee semantics.
    # =====================================================================
    if notional_usd > 0:
        _fee_result = ctx.fn_build_execution_fee_result(
            asset=asset,
            executed_notional_usd=notional_usd,
            order_type=order_type,
            execution_direction=_execution_direction,
            regime_leverage=regime_leverage,
            existing_position=_existing_position,
        )
        exec_result["fee_blending"] = _fee_result
        logger.info(
            f"[EXEC_FEE] {asset} fill ${notional_usd:,.0f} | "
            f"spot={_fee_result.get('is_spot', True)} "
            f"margin={_fee_result.get('requires_margin', False)} | "
            f"trade_fee=${float(_fee_result.get('trade_fee_usd', 0.0) or 0.0):.2f} "
            f"({float(_fee_result.get('fee_effective', 0.0) or 0.0) * 10000:.1f}bps) | "
            f"maker={bool(_fee_result.get('is_maker', False))}"
        )

    # [SOTA] Record successful fill for FillRateKPI
    try:
        if ctx.fill_rate_kpi:
            _sota_strat = getattr(intent, 'quant_strategy_id', None) or "unknown"
            _sota_fill_pct = 1.0
            if _c5_filled_size is not None and exec_result.get("requested_size", 0) > 0:
                _sota_fill_pct = min(1.0, _c5_filled_size / exec_result["requested_size"])
            ctx.fill_rate_kpi.record_order(
                strategy=_sota_strat, placed=True, filled=True,
                fill_pct=_sota_fill_pct, asset=asset,
            )
    except Exception:
        pass

    # [G8-lite] Fill quality logging -JSONL for weekly review
    try:
        import json as _fq_json
        _fq_requested = exec_result.get("requested_size", base_quantity)
        _fq_fill_ratio = min(1.0, _c5_filled_size / _fq_requested) if _fq_requested and _fq_requested > 0 else 1.0
        _fq_slippage = ((fill_price - current_price) / current_price * 10000) if current_price > 0 else 0.0
        _fq_direction = "long" if _execution_direction > 0 else "short"
        _fq_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "asset": asset,
            "direction": _fq_direction,
            "order_type": order_type,
            "fill_ratio": round(_fq_fill_ratio, 4),
            "slippage_bps": round(_fq_slippage, 2),
            "fill_price": round(fill_price, 4),
            "expected_price": round(current_price, 4),
            "notional_usd": round(notional_usd, 2),
        }
        with open("logs/fill_quality.jsonl", "a") as _fq_f:
            _fq_f.write(_fq_json.dumps(_fq_entry) + "\n")
        logger.info(
            f"[FILL-QUALITY] {asset} {_fq_direction}: "
            f"ratio={_fq_fill_ratio:.0%}, "
            f"slippage={_fq_slippage:.1f}bps, "
            f"type={order_type}"
        )
    except Exception as _fq_err:
        logger.debug(f"[FILL-QUALITY] Log error: {_fq_err}")

    # [WIRE] W-3b: FillSlopeMonitor -record fill for adverse selection detection
    if ctx.fill_slope_monitor is not None:
        try:
            _w3b_order_id = exec_result.get("order_id", f"{asset}_{int(time.time())}")
            ctx.fill_slope_monitor.ensure_order_registered(
                order_id=_w3b_order_id,
                asset=asset,
                side=side,
                total_qty=base_quantity,
                target_price=execution_price,
                limit_price=execution_price if order_type == "LIMIT" else fill_price,
            )
            _w3b_analysis = ctx.fill_slope_monitor.record_fill(
                order_id=_w3b_order_id,
                fill_qty=base_quantity,
                fill_price=fill_price,
                is_maker=(order_type == "LIMIT"),
            )
            if _w3b_analysis and _w3b_analysis.is_toxic:
                market_data['fill_slope_toxic'] = True
                logger.warning(
                    f"[WIRE] W-3b FillSlope TOXIC: {asset} "
                    f"{_w3b_analysis.toxicity_reason} ->pausing tranche upgrade"
                )
            # Track asset-level toxicity rate
            _w3b_tox_rate = ctx.fill_slope_monitor.get_asset_toxicity_rate(asset)
            if _w3b_tox_rate > 0.3:
                logger.info(f"[WIRE] W-3b {asset} toxicity rate: {_w3b_tox_rate:.0%}")
        except Exception as _w3b_err:
            logger.debug(f"[WIRE] W-3b FillSlope skipped: {_w3b_err}")

    if ctx.config.mode == RunMode.PAPER or (ctx.account_sync and ctx.account_sync.dry_run):
        direction_sign = 1.0 if intent.direction > 0 else -1.0
        old_pos = ctx.paper_positions.get(asset)
        _order_fee_result = dict(exec_result.get("fee_blending", {}) or {})
        _order_trade_fee_usd = float(
            _order_fee_result.get(
                "trade_fee_usd",
                _order_fee_result.get("fee_usd", 0.0),
            )
            or 0.0
        )
        is_full_exit = (intent.target_exposure == 0 or intent.direction == 0)

        # FIX: Detect noise rebalances (price drift causes $0-50 notional
        # deltas that trigger spurious Branch B "partial close" each tick)
        is_noise_rebalance = (
            not is_full_exit
            and old_pos is not None
            and old_pos.get("notional", 0) > 0
            and abs(notional_usd - old_pos.get("notional", 0)) < 50.0
        )

        is_partial_exit = (
            not is_full_exit
            and not is_noise_rebalance
            and old_pos is not None
            and old_pos.get("direction", 0) * direction_sign > 0      # same direction
            and old_pos.get("notional", 0) > 0
            and notional_usd < old_pos.get("notional", 0)            # shrinking
        )

        # =============================================================
        # BRANCH A: FULL EXIT  (target_exposure==0 or direction==0)
        # =============================================================
        if is_full_exit:
            # [v3.3-C7] Cancel protective stop before closing position
            # [BUGFIX M5] Retry cancel once + elevate log on failure (orphaned stop risk)
            try:
                from execution.execution_manager import OrderSide as _C7Side
                _c7_symbol = f"{asset}/USD"
                _m5_cancelled = ctx.execution_manager.cancel_stop_loss(_c7_symbol)
                if not _m5_cancelled:
                    # [AUDIT C2] Non-blocking retry in async context (was time.sleep)
                    import asyncio as _m5_asyncio
                    await _m5_asyncio.sleep(0.2)
                    _m5_cancelled = ctx.execution_manager.cancel_stop_loss(_c7_symbol)
                if _m5_cancelled:
                    logger.info(f"[STOP] {asset}: cancelled protective stop (position closing)")
                else:
                    logger.warning(f"[AUDIT H3] {asset}: stop cancel returned False after retry -tracking as orphaned")
                    ctx.orphaned_stops.add(_c7_symbol)
            except Exception as _c7_err:
                logger.warning(f"[AUDIT H3] {asset}: stop cancel FAILED: {_c7_err} -tracking as orphaned")
                ctx.orphaned_stops.add(f"{asset}/USD")

            old_pos = ctx.paper_positions.pop(asset, None)
            # [FIX-TRANCHE-STALE] Clear stale tranche scheduler state when position is
            # fully closed. Without this, corrupted T2/T3 state from a FLIP_GATE
            # block persists and can trigger oversized entries on the next tick.
            try:
                _tranche_sched = ctx.engine.guarantees.tranche_scheduler
                _tranche_sched.positions.pop(asset, None)
            except Exception:
                pass
            if ctx.gambler_exit:
                try:
                    ctx.gambler_exit.clear_entry(asset)
                except Exception:
                    pass
            if ctx.exit_alpha:
                try:
                    ctx.exit_alpha.reset_for_asset(asset)
                except Exception:
                    pass
            # [WIRE-2] Remove position from adaptive stop
            if ctx.adaptive_stop:
                try:
                    ctx.adaptive_stop.remove_position(asset)
                except Exception:
                    pass
            # T20: Release opportunity budget on position close
            if ctx.opportunity_budget:
                try:
                    released = ctx.opportunity_budget.release_for_symbol(asset)
                    if released > 0:
                        logger.info(f"[OP_BUDGET] Released {released} slot(s) for {asset}")
                except Exception as _op_err:
                    logger.warning(f"[SOTA L1] OpportunityBudget release failed: {_op_err}")
            if old_pos:
                exit_price = fill_price
                entry_price = old_pos.get("entry_price", None)
                # [BUGFIX M1+M2] If entry_price missing, use avg_entry_price or
                # last known fill price as fallback (was: silently returns PnL=0)
                if entry_price is None or entry_price <= 0:
                    _fallback = old_pos.get("avg_entry_price", None) or old_pos.get("last_fill_price", None)
                    if _fallback and _fallback > 0:
                        entry_price = _fallback
                        logger.warning(
                            f"[BUGFIX M2] {asset}: entry_price missing, "
                            f"using fallback={_fallback:.2f} from position record"
                        )
                    else:
                        logger.error(
                            f"[BUGFIX M2] {asset}: entry_price AND fallbacks all missing/invalid "
                            f"in position record -PnL will be 0 for this close"
                        )
                        entry_price = 0  # sentinel: triggers the else branch below
                pos_dir = old_pos.get("direction", 0)
                pos_notional = old_pos.get("notional", 0.0)
                _funding_pnl = old_pos.get("cumulative_funding_pnl", 0.0)  # [FIX-L1-02]
                if entry_price > 0 and pos_notional > 0:
                    pnl_pct = (exit_price - entry_price) / entry_price * pos_dir
                    pnl_usd = pnl_pct * pos_notional
                    # [REGIME-LEV] Track margin costs for leveraged trades
                    _pos_regime_lev = float(old_pos.get("regime_leverage", regime_leverage) or 1.0)
                    if _pos_regime_lev > 1.0:
                        _rl_entry_info = ctx.position_entry_times.get(asset)
                        _rl_hold_bars = 1
                        if _rl_entry_info:
                            _rl_held_s = (datetime.now(timezone.utc) - _rl_entry_info["entry_time"]).total_seconds()
                            _rl_hold_bars = max(1, int(_rl_held_s / 14400))
                        ctx.margin_tracker.record_trade(
                            notional_usd=pos_notional, leverage=_pos_regime_lev,
                            holding_bars=_rl_hold_bars, pnl_usd=pnl_usd,
                        )
                else:
                    pnl_pct = 0.0
                    pnl_usd = 0.0

                _exit_fee_usd = _order_trade_fee_usd
                _realized_outcome = ctx.fn_build_realized_outcome(
                    position=old_pos,
                    gross_pnl_usd=pnl_usd,
                    closed_notional_usd=pos_notional,
                    exit_fee_usd=_exit_fee_usd,
                    funding_pnl_usd=_funding_pnl,
                )
                _net_pnl_usd = _realized_outcome["net_pnl_usd"]
                _net_pnl_bps = _realized_outcome["net_pnl_bps"]

                if ctx.account_sync and ctx.account_sync.dry_run:
                    ctx.account_sync.update_dry_run_pnl(_net_pnl_usd)
                if ctx.risk_manager:
                    ctx.risk_manager.record_realized_pnl(_net_pnl_usd)
                if ctx.existence_fuse:
                    ctx.existence_fuse.on_trade_close(_net_pnl_usd)
                    # [FIX-P0-3] Feed real PnL into fuse rolling window
                    _fuse_equity = ctx.config.initial_capital
                    if ctx.account_sync:
                        _fuse_equity = ctx.account_sync.get_equity()
                    ctx.existence_fuse.record_pnl(
                        realized_pnl=_net_pnl_usd,
                        current_equity=_fuse_equity,
                        trade_count=1,
                    )
                _tilt_key = asset.upper().replace("/USD", "").replace("USD", "")
                if _tilt_key in ctx.asset_trade_pnls:
                    ctx.asset_trade_pnls[_tilt_key].append(_net_pnl_usd)
                ctx.fn_record_realized_pnl_breakdown(
                    asset=asset,
                    side="long" if pos_dir > 0 else "short",
                    strategy=old_pos.get("strategy", intent.quant_strategy_id or "momentum"),
                    pnl_usd=_net_pnl_usd,
                    pnl_bps=_net_pnl_bps,
                    exit_type="FULL",
                )
                logger.info(
                    f"[PAPER_PNL] {asset} CLOSED: entry=${entry_price:.2f} exit=${exit_price:.2f} "
                    f"dir={pos_dir:+.0f} gross=${pnl_usd:+.2f} net=${_net_pnl_usd:+.2f} "
                    f"({pnl_pct*100:+.2f}% gross, {_net_pnl_bps:+.1f}bps net) "
                    f"fees=${(_realized_outcome['entry_fee_allocated_usd'] + _exit_fee_usd):.2f} "
                    f"funding=${_funding_pnl:+.4f}"
                )

                cooldown_until = datetime.now(timezone.utc) + timedelta(
                    hours=4 * ctx.REBUILD_COOLDOWN_TICKS
                )
                ctx.rebuild_cooldown[asset] = (
                    cooldown_until,
                    f"full_close_net=${_net_pnl_usd:+.2f}",
                    pos_dir,  # closed direction -opposite-direction entries are exempt
                )
                logger.info(
                    f"[REBUILD_COOLDOWN] {asset}: cooldown set until "
                    f"{cooldown_until.strftime('%H:%M UTC')} "
                    f"({ctx.REBUILD_COOLDOWN_TICKS} ticks, reason=full_close)"
                )

                # Shadow ledger
                if ctx.p0_integrator is not None and ctx.p0_integrator:
                    try:
                        close_side = "BUY" if pos_dir < 0 else "SELL"
                        order_id = exec_result.get("order_id", exec_result.get("id", f"paper_{asset}_{int(datetime.now(timezone.utc).timestamp())}"))
                        _shadow_fill_ok = ctx.p0_integrator.record_fill(
                            asset=asset, side=close_side,
                            size=pos_notional / exit_price if exit_price > 0 else 0.0,
                            price=exit_price, order_id=str(order_id),
                            fee=_exit_fee_usd, realized_pnl=_net_pnl_usd,
                            extra={
                                "exit_trigger": ctx.exit_trigger_tag.get(asset, "UNKNOWN"),
                                "funding_pnl": _funding_pnl,
                                "trade_fee_usd": _exit_fee_usd,
                                "margin_opening_fee_usd": 0.0,
                                # [FIX 2026-04-22] PnL attribution — who "caused" this trade
                                "primary_agent": getattr(ctx.intent, "primary_agent", "") or "",
                            },
                        )
                        _note_shadow_fill(bool(_shadow_fill_ok))
                    except Exception as e:
                        logger.debug(f"[SHADOW_LEDGER] record_fill (close) failed: {e}")

                # [W10] Trade Attributor -record full exit
                if ctx.trade_attributor:
                    try:
                        ctx.trade_attributor.record_exit(
                            asset=asset, price=exit_price, fee=_exit_fee_usd,
                            notional=pos_notional, gross_pnl=pnl_usd,
                            exit_type="FULL",
                        )
                    except Exception as _ta_err:
                        logger.debug(f"[W10] TradeAttributor record_exit (full) failed: {_ta_err}")

                # [HIT-RATE] Update alpha gate performance factor from trade outcome
                try:
                    _alc = getattr(getattr(getattr(ctx, 'engine', None), 'guarantees', None), 'alpha_calculator', None)
                    if _alc and hasattr(_alc, 'update_hit_rate'):
                        _alc.update_hit_rate(won=_net_pnl_usd > 0)
                except Exception:
                    pass

                # [W11] Signal Quality -record outcome (full exit)
                if ctx.sq_tracker:
                    try:
                        _sq_hold = ctx.tick_count - old_pos.get("entry_tick", 0)
                        ctx.sq_tracker.record_outcome(
                            asset=asset, exit_price=exit_price,
                            gross_pnl=pnl_usd, hold_ticks=_sq_hold,
                            exit_reason="FULL_EXIT",
                        )
                    except Exception:
                        pass

                if ctx.experience_buffer is not None:
                    try:
                        _exp_hold_bars = 1
                        _exp_entry_time = old_pos.get("entry_time")
                        if _exp_entry_time:
                            _exp_entry_dt = datetime.fromisoformat(
                                str(_exp_entry_time).replace("Z", "+00:00")
                            )
                            if _exp_entry_dt.tzinfo is None:
                                _exp_entry_dt = _exp_entry_dt.replace(tzinfo=timezone.utc)
                            _exp_hold_bars = max(
                                1,
                                int(
                                    (
                                        datetime.now(timezone.utc) - _exp_entry_dt
                                    ).total_seconds() / 14400
                                ),
                            )
                        ctx.experience_buffer.record_outcome(
                            asset=asset,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            realized_pnl=_net_pnl_usd,
                            bars_held=_exp_hold_bars,
                            realized_pnl_bps=_net_pnl_bps,
                            exit_reason=intent.veto_reason or "full_exit",
                        )
                    except Exception as e:
                        logger.debug(f"[EXPERIENCE] record_outcome (full exit) failed: {e}")

                # [EA-4a] Exit Alpha Tracker -record full exit with trigger classification
                if ctx.ea_tracker:
                    try:
                        _ea_trigger = ctx.exit_trigger_tag.pop(asset, "UNKNOWN")
                        ctx.ea_tracker.record_exit(
                            asset=asset, exit_price=exit_price,
                            trigger=_ea_trigger, pnl_usd=pnl_usd,
                            tick=ctx.tick_count, exit_type="FULL",
                            regime_at_exit=market_data.get("regime_state", ""),
                        )
                    except Exception:
                        pass

                # Audit: log full close + Discord alert
                if ctx.audit_manager:
                    try:
                        _a_qty = pos_notional / exit_price if exit_price > 0 else 0.0
                        _a_side = "BUY" if pos_dir < 0 else "SELL"
                        _a_fee = _exit_fee_usd
                        _a_oid = exec_result.get("order_id", exec_result.get("id", "")) or ""
                        ctx.audit_manager.log_trade_execution(AuditTradeExecution(
                            execution_id=f"{asset}_{int(time.time())}_close",
                            intent_id=f"{asset}_{ctx.tick_count}",
                            timestamp=time.time(), asset=asset, side=_a_side,
                            quantity=_a_qty, price=exit_price, fee=_a_fee,
                            slippage_bps=market_data.get("spread_bps", 0.0),
                            exchange_order_id=str(_a_oid),
                        ), send_notification=True)
                    except Exception as e:
                        logger.debug(f"[AUDIT] Execution log failed: {e}")

                # [Layer 3] Thesis Budget -record ACTUAL realized PnL
                if ctx.thesis_budget_governor:
                    try:
                        _tb_side = "long" if pos_dir > 0 else "short"
                        _tb_strategy = old_pos.get("strategy", intent.quant_strategy_id or "momentum")
                        _tb_key = f"{asset.upper()}:{_tb_side}:{_tb_strategy}"
                        ctx.thesis_budget_governor.record_fill(_tb_key, _net_pnl_usd)
                        logger.info(
                            f"[THESIS_BUDGET] FULL EXIT {_tb_key}: pnl=${_net_pnl_usd:+.2f} "
                            f"(is_win={_net_pnl_usd >= 0})"
                        )
                    except Exception as e:
                        logger.debug(f"[THESIS_BUDGET] record_fill (full exit) failed: {e}")

                # [Section E] ConfidenceScorer -record outcome at close (feedback loop)
                if ctx.confidence_scorer is not None and asset in ctx.confidence_signal_times:
                    try:
                        sig_ts = ctx.confidence_signal_times.pop(asset)
                        pnl_bps = _net_pnl_bps
                        direction_correct = _net_pnl_usd > 0
                        strategy = old_pos.get("strategy", intent.quant_strategy_id or "momentum")
                        ctx.confidence_scorer.record_outcome(
                            strategy_name=strategy,
                            signal_timestamp=sig_ts,
                            realized_pnl_bps=pnl_bps,
                            direction_correct=direction_correct,
                        )
                        logger.info(
                            f"[CONFIDENCE] Recorded {asset}: strategy={strategy} "
                            f"pnl={pnl_bps:+.1f}bps correct={direction_correct}"
                        )
                    except Exception as e:
                        logger.debug(f"[CONFIDENCE] record_outcome failed: {e}")

                # [v3.3-C10] MetaDecision: record trade outcome
                if ctx.meta_decision is not None:
                    try:
                        _md_strategy = old_pos.get("strategy", intent.quant_strategy_id or "momentum")
                        _md_pnl_bps = _net_pnl_bps
                        _md_regime = market_data.get('regime_state', 'UNKNOWN')
                        ctx.meta_decision.record_outcome(
                            strategy=_md_strategy,
                            pnl_bps=_md_pnl_bps,
                            regime=_md_regime,
                            is_opportunity=(old_pos.get("mode_at_entry") == "OPPORTUNITY"),  # [RULETABLE] was hardcoded False
                        )
                        logger.info(
                            f"[META_DECISION] Recorded {asset}: strategy={_md_strategy} "
                            f"pnl={_md_pnl_bps:+.1f}bps regime={_md_regime}"
                        )
                    except Exception as _md_err:
                        logger.debug(f"[META_DECISION] record_outcome failed: {_md_err}")

                # [v3.3-C11] PnL Attribution: record trade for DRL promotion
                if ctx.pnl_attribution is not None:
                    try:
                        _c11_total_bps = _net_pnl_bps
                        _c11_entry_alpha = old_pos.get("entry_alpha_est", 0.0)
                        _c11_exec_alpha = -(
                            _realized_outcome["entry_fee_bps"] + _realized_outcome["exit_fee_bps"]
                        )
                        _c11_exit_alpha = _c11_total_bps - _c11_entry_alpha - _c11_exec_alpha
                        _c11_trade_id = f"{asset}_{int(datetime.now(timezone.utc).timestamp())}"
                        _c11_drl_dir = float(old_pos.get("latest_drl_direction", 0.0) or 0.0)
                        _c11_drl_conf = float(old_pos.get("latest_drl_confidence", 0.0) or 0.0)
                        _c11_pos_dir = float(old_pos.get("direction", 0.0) or 0.0)
                        _c11_drl_rec = None
                        _c11_drl_hypo = None
                        if _c11_drl_conf >= 0.35 and (_c11_drl_dir * _c11_pos_dir) < -0.15:
                            _c11_drl_rec = "EXIT"
                            # Conservative estimate: shadow DRL would have avoided
                            # negative exit alpha, but does not claim extra profit.
                            _c11_drl_hypo = max(0.0, _c11_exit_alpha)
                        ctx.pnl_attribution.record_trade(
                            trade_id=_c11_trade_id,
                            entry_alpha_bps=_c11_entry_alpha,
                            execution_alpha_bps=_c11_exec_alpha,
                            exit_alpha_bps=_c11_exit_alpha,
                            total_pnl_bps=_c11_total_bps,
                            drl_shadow_recommendation=_c11_drl_rec,
                            drl_shadow_would_have_bps=_c11_drl_hypo,
                            actual_exit_type=ctx.exit_trigger_tag.get(asset, "FULL_EXIT"),
                            actual_exit_bps=_c11_exit_alpha,
                        )
                        logger.info(
                            f"[PNL_ATTRIB] {asset}: entry={_c11_entry_alpha:+.1f}bps "
                            f"exec={_c11_exec_alpha:+.1f}bps exit={_c11_exit_alpha:+.1f}bps "
                            f"total={_c11_total_bps:+.1f}bps"
                        )
                        _c11_signal = ctx.pnl_attribution.get_promotion_signal()
                        if _c11_signal.ready_for_promotion:
                            logger.info(
                                f"[PNL_PROMO] DRL ready for authority upgrade: "
                                f"type={_c11_signal.promotion_type} "
                                f"conf={_c11_signal.confidence:.2f} "
                                f"evidence={_c11_signal.supporting_evidence}"
                            )
                            if ctx.promotion_gate and ctx.drl_models_ready > 0:
                                try:
                                    _promo_target = "EXIT_ONLY" if _c11_signal.promotion_type == "SHADOW_TO_EXIT" else "ACTIVE"
                                    if ctx.drl_authority_level in ("DISABLED", "SHADOW"):
                                        ctx.promotion_gate.promote(_promo_target)
                                        ctx.drl_authority_level = ctx.promotion_gate.get_authority_level()
                                        ctx.fn_sync_drl_authority(ctx.drl_authority_level)
                                        logger.warning(
                                            f"[PNL_PROMO] DRL promoted to {ctx.drl_authority_level} "
                                            f"from attribution signal"
                                        )
                                except Exception as _promo_err:
                                    logger.debug(f"[PNL_PROMO] promotion apply failed: {_promo_err}")
                        elif _c11_signal.blocking_factors:
                            logger.debug(
                                f"[PNL_PROMO] Not ready: {_c11_signal.blocking_factors[0]}"
                            )
                    except Exception as _c11_err:
                        logger.debug(f"[PNL_ATTRIB] record_trade failed: {_c11_err}")

                # [DRL-COUNT] Record trade in promotion gate for shadow trade counting
                if ctx.promotion_gate:
                    try:
                        _drl_contributed = bool(_c11_drl_rec is not None and _c11_drl_conf >= 0.35)
                        ctx.promotion_gate.record_trade(
                            pnl=_net_pnl_usd,
                            drl_contributed=_drl_contributed,
                        )
                        _gate_status = ctx.promotion_gate.get_status()
                        logger.info(
                            f"[DRL_COUNT] {asset}: trade recorded "
                            f"(pnl=${_net_pnl_usd:+.2f}, drl_contributed={_drl_contributed}, "
                            f"total_trades={_gate_status.get('total_trades', '?')})"
                        )
                    except Exception as _drl_count_err:
                        logger.debug(f"[DRL_COUNT] record_trade failed: {_drl_count_err}")

                # [SOTA-G6] Feed trade outcome to adaptive weight manager
                if ctx.sota_integration:
                    try:
                        _g6w = ctx.sota_integration.components.get("adaptive_weighting")
                        if _g6w and hasattr(_g6w, 'record_trade'):
                            _g6_strat = old_pos.get("strategy", intent.quant_strategy_id or "unknown")
                            _g6w.record_trade(
                                strategy_name=_g6_strat,
                                pnl_pct=(_net_pnl_usd / pos_notional) if pos_notional > 0 else 0.0,
                                is_winner=(_net_pnl_usd > 0),
                            )
                    except Exception:
                        pass

                # [v3.3-C12] Strategy Aging: record outcome + weekly weight check
                if ctx.strategy_aging is not None:
                    try:
                        _c12_strategy = old_pos.get("strategy", intent.quant_strategy_id or "momentum")
                        _c12_pnl_bps = _net_pnl_bps
                        _c12_correct = _net_pnl_usd > 0
                        _c12_entry_time_str = old_pos.get("entry_time", "")
                        _c12_sig_ts = datetime.fromisoformat(_c12_entry_time_str) if _c12_entry_time_str else datetime.now(timezone.utc)
                        ctx.strategy_aging.record_outcome(
                            strategy_name=_c12_strategy,
                            signal_timestamp=_c12_sig_ts,
                            pnl_bps=_c12_pnl_bps,
                            was_correct_direction=_c12_correct,
                        )
                        _c12_now = datetime.now(timezone.utc)
                        if ctx.last_aging_check is None or (
                            _c12_now - ctx.last_aging_check
                        ).total_seconds() > 168 * 3600:
                            ctx.last_aging_check = _c12_now
                            _c12_mods = ctx.strategy_aging.get_weight_modifiers()
                            for _c12_sname, _c12_wmod in _c12_mods.items():
                                if _c12_wmod < 0.7:
                                    logger.critical(
                                        f"[STRATEGY_AGING] {_c12_sname} degraded: "
                                        f"weight modifier={_c12_wmod:.2f}. "
                                        f"Consider reviewing strategy parameters."
                                    )
                                elif _c12_wmod > 1.1:
                                    logger.info(
                                        f"[STRATEGY_AGING] {_c12_sname} strong: "
                                        f"weight modifier={_c12_wmod:.2f}"
                                    )
                    except Exception as _c12_err:
                        logger.debug(f"[STRATEGY_AGING] record_outcome failed: {_c12_err}")

                # [FIX-16] Removed duplicate SOTA-ACT GA-1 block -SOTA-G6 (L10200) already records

                # [v3.3-C13] FailureAwareMetaMemory: record OPPORTUNITY outcome
                if ctx.failure_memory is not None and old_pos.get("mode_at_entry") == "OPPORTUNITY":
                    try:
                        _c13_pnl_bps = _net_pnl_bps
                        _c13_entry_time_str = old_pos.get("entry_time", "")
                        _c13_entry_dt = datetime.fromisoformat(_c13_entry_time_str) if _c13_entry_time_str else datetime.now(timezone.utc)
                        _c13_bars = max(1, int((datetime.now(timezone.utc) - _c13_entry_dt).total_seconds() / (4 * 3600)))
                        ctx.failure_memory.record_opportunity(
                            trade_id=f"{asset}_{int(datetime.now(timezone.utc).timestamp())}",
                            asset=asset,
                            entry_time=_c13_entry_dt,
                            phase_at_entry=old_pos.get("phase_at_entry", "UNKNOWN"),
                            opportunity_density_at_entry=0.0,
                            crack_weight_at_entry=old_pos.get("crack_weight", 0.0),
                            pnl_bps=_c13_pnl_bps,
                            bars_to_outcome=_c13_bars,
                            max_drawdown_bps=abs(min(_c13_pnl_bps, 0.0)),
                        )
                        _c13_mods = ctx.failure_memory.get_modifiers()
                        if _c13_mods.in_caution_mode:
                            logger.warning(
                                f"[FAILURE_MEMORY] Caution mode ACTIVE after {asset}: "
                                f"density_boost={_c13_mods.density_boost:.2f}, "
                                f"failures={_c13_mods.consecutive_failures}"
                            )
                    except Exception as _c13_err:
                        logger.debug(f"[FAILURE_MEMORY] record_opportunity failed: {_c13_err}")

                # Sync tranche state after full exit (prevents desync with paper_positions)
                if ctx.config.mode != RunMode.VERIFY:
                    try:
                        ctx.fn_persist_tranche_state()
                    except Exception as e:
                        logger.warning(f"[FIX-27] tranche_state persist failed: {e}")

                # W8: Close position in feedback loop (strategy perf tracking)
                if ctx.feedback_loop:
                    try:
                        _fb_open = ctx.feedback_loop.get_open_positions()
                        for _fb_tid, _fb_pos in _fb_open.items():
                            if _fb_pos.asset == asset:
                                ctx.feedback_loop.close_position(
                                    trade_id=_fb_tid,
                                    exit_price=fill_price,
                                    exit_reason=intent.veto_reason or "full_exit",
                                )
                                break
                    except Exception:
                        pass

        # =============================================================
        # BRANCH D: NOISE REBALANCE (position unchanged, skip)
        # Price drift causes notional delta <$50 -not a real trade.
        # =============================================================
        elif is_noise_rebalance:
            logger.debug(
                f"[PAPER_NOOP] {asset}: position unchanged "
                f"(delta=${abs(notional_usd - old_pos.get('notional', 0)):.2f})"
            )

        # =============================================================
        # BRANCH B: PARTIAL EXIT  (same direction, exposure shrinking)
        # =============================================================
        elif is_partial_exit:
            entry_price = old_pos.get("entry_price", fill_price)
            pos_dir = old_pos.get("direction", direction_sign)
            old_notional = old_pos.get("notional", 0.0)
            closed_notional = old_notional - notional_usd
            # [FIX-L1-02] Pro-rate funding PnL by close fraction
            _close_frac = closed_notional / old_notional if old_notional > 0 else 0.0
            _total_funding = old_pos.get("cumulative_funding_pnl", 0.0)
            _partial_funding_pnl = _total_funding * _close_frac

            # Compute PnL on the closed portion using actual fill price
            pnl_usd = 0.0
            pnl_pct = 0.0
            if entry_price > 0 and closed_notional > 0:
                pnl_pct = (fill_price - entry_price) / entry_price * pos_dir
                pnl_usd = pnl_pct * closed_notional
            _partial_exit_fee_usd = _order_trade_fee_usd
            _partial_outcome = ctx.fn_build_realized_outcome(
                position=old_pos,
                gross_pnl_usd=pnl_usd,
                closed_notional_usd=closed_notional,
                exit_fee_usd=_partial_exit_fee_usd,
                funding_pnl_usd=_partial_funding_pnl,
            )
            _partial_net_pnl_usd = _partial_outcome["net_pnl_usd"]
            _partial_net_pnl_bps = _partial_outcome["net_pnl_bps"]

            if ctx.account_sync and ctx.account_sync.dry_run:
                ctx.account_sync.update_dry_run_pnl(_partial_net_pnl_usd)
            if ctx.risk_manager:
                ctx.risk_manager.record_realized_pnl(_partial_net_pnl_usd)
            if ctx.existence_fuse:
                ctx.existence_fuse.on_trade_close(_partial_net_pnl_usd)
                # [FIX-P0-3] Feed real PnL into fuse rolling window
                _fuse_equity = ctx.config.initial_capital
                if ctx.account_sync:
                    _fuse_equity = ctx.account_sync.get_equity()
                ctx.existence_fuse.record_pnl(
                    realized_pnl=_partial_net_pnl_usd,
                    current_equity=_fuse_equity,
                    trade_count=1,
                )
            _tilt_key = asset.upper().replace("/USD", "").replace("USD", "")
            if _tilt_key in ctx.asset_trade_pnls:
                ctx.asset_trade_pnls[_tilt_key].append(_partial_net_pnl_usd)
            ctx.fn_record_realized_pnl_breakdown(
                asset=asset,
                side="long" if pos_dir > 0 else "short",
                strategy=old_pos.get("strategy", intent.quant_strategy_id or "momentum"),
                pnl_usd=_partial_net_pnl_usd,
                pnl_bps=_partial_net_pnl_bps,
                exit_type="PARTIAL",
            )
            logger.info(
                f"[PAPER_PNL] {asset} PARTIAL CLOSE: entry=${entry_price:.2f} "
                f"exit=${fill_price:.2f} dir={pos_dir:+.0f} "
                f"closed=${closed_notional:,.0f} of ${old_notional:,.0f} "
                f"gross=${pnl_usd:+.2f} net=${_partial_net_pnl_usd:+.2f} "
                f"({pnl_pct*100:+.2f}% gross, {_partial_net_pnl_bps:+.1f}bps net) "
                f"fees=${(_partial_outcome['entry_fee_allocated_usd'] + _partial_exit_fee_usd):.2f} "
                f"funding=${_partial_funding_pnl:+.4f}"
            )

            cooldown_until = datetime.now(timezone.utc) + timedelta(
                hours=4 * ctx.REBUILD_COOLDOWN_TICKS
            )
            ctx.rebuild_cooldown[asset] = (
                cooldown_until,
                f"partial_close_net=${_partial_net_pnl_usd:+.2f}",
                pos_dir,  # closed direction -opposite-direction entries are exempt
            )
            logger.info(
                f"[REBUILD_COOLDOWN] {asset}: cooldown set until "
                f"{cooldown_until.strftime('%H:%M UTC')} "
                f"({ctx.REBUILD_COOLDOWN_TICKS} ticks, reason=partial_close)"
            )

            # [FIX-H1] Notify failure_memory of partial exit PnL.
            # Without this, OPPORTUNITY scale-out losses are invisible to
            # the caution system, so it never learns from partial exits.
            if (ctx.failure_memory
                    and old_pos.get("mode_at_entry") == "OPPORTUNITY"
                    and _partial_net_pnl_usd != 0):
                try:
                    _h1_entry_t_str = old_pos.get("entry_time", "")
                    _h1_entry_dt = (
                        datetime.fromisoformat(_h1_entry_t_str)
                        if _h1_entry_t_str
                        else datetime.now(timezone.utc)
                    )
                    _h1_bars = max(1, int(
                        (datetime.now(timezone.utc) - _h1_entry_dt).total_seconds()
                        / (4 * 3600)
                    ))
                    ctx.failure_memory.record_opportunity(
                        trade_id=f"{asset}_partial_{int(datetime.now(timezone.utc).timestamp())}",
                        asset=asset,
                        entry_time=_h1_entry_dt,
                        phase_at_entry=old_pos.get("phase_at_entry", "UNKNOWN"),
                        opportunity_density_at_entry=0.0,
                        crack_weight_at_entry=old_pos.get("crack_weight", 0.0),
                        pnl_bps=_partial_net_pnl_bps,
                        bars_to_outcome=_h1_bars,  # [FIX-C3] was bars_held (wrong param name)
                    )
                except Exception as _h1_err:
                    logger.debug(f"[FIX-H1] failure_memory partial record failed: {_h1_err}")

            # Update remaining position -preserve original entry_price
            tranche = getattr(intent, 'tranche_target', old_pos.get("tranche", 1))
            _remaining_exposure = abs(intent.target_exposure)
            if account_equity and account_equity > 0:
                _remaining_exposure = max(0.0, float(notional_usd) / float(account_equity))
            ctx.paper_positions[asset] = {
                "exposure": _remaining_exposure,
                "direction": pos_dir,
                "tranche": tranche,
                "entry_price": entry_price,          # PRESERVED from original entry
                "notional": notional_usd,
                "entry_time": old_pos.get("entry_time", datetime.now(timezone.utc).isoformat()),
                "strategy": old_pos.get("strategy", intent.quant_strategy_id or "momentum"),
                "entry_alpha_est": old_pos.get("entry_alpha_est", 0.0),
                "mode_at_entry": old_pos.get("mode_at_entry", "NORMAL"),
                "phase_at_entry": old_pos.get("phase_at_entry", "UNKNOWN"),
                "crack_weight": old_pos.get("crack_weight", 0.0),
                "original_entry_exposure": old_pos.get("original_entry_exposure", old_pos.get("exposure", 0)),  # [FIX-3a] preserve
                "cumulative_funding_pnl": _total_funding - _partial_funding_pnl,  # [FIX-L1-02] remaining funding
                "entry_fee_usd": _partial_outcome["remaining_entry_fee_usd"],
                "cumulative_entry_fees_usd": _partial_outcome["remaining_entry_fee_usd"],
            }

            # Shadow ledger -partial close fill
            if ctx.p0_integrator is not None and ctx.p0_integrator:
                try:
                    close_side = "BUY" if pos_dir < 0 else "SELL"  # [FIX-8] close direction, not intent direction
                    order_id = exec_result.get("order_id", exec_result.get("id", f"paper_{asset}_{int(datetime.now(timezone.utc).timestamp())}"))
                    _shadow_fill_ok = ctx.p0_integrator.record_fill(
                        asset=asset, side=close_side,
                        size=base_quantity, price=fill_price,
                        order_id=str(order_id), fee=_partial_exit_fee_usd,
                        realized_pnl=_partial_net_pnl_usd,
                        extra={
                            "exit_trigger": ctx.exit_trigger_tag.get(asset, "PARTIAL"),
                            "funding_pnl": _partial_funding_pnl,
                            "trade_fee_usd": _partial_exit_fee_usd,
                            "margin_opening_fee_usd": 0.0,
                            # [FIX 2026-04-22] PnL attribution
                            "primary_agent": getattr(ctx.intent, "primary_agent", "") or "",
                        },
                    )
                    _note_shadow_fill(bool(_shadow_fill_ok))
                except Exception as e:
                    logger.debug(f"[SHADOW_LEDGER] record_fill (partial close) failed: {e}")

            # [W10] Trade Attributor -record partial exit
            if ctx.trade_attributor:
                try:
                    ctx.trade_attributor.record_exit(
                        asset=asset, price=fill_price, fee=_partial_exit_fee_usd,
                        notional=closed_notional, gross_pnl=pnl_usd,
                        exit_type="PARTIAL",
                    )
                except Exception as _ta_err:
                    logger.debug(f"[W10] TradeAttributor record_exit (partial) failed: {_ta_err}")

            # [HIT-RATE] Update alpha gate performance factor from partial trade outcome
            try:
                _alc = getattr(getattr(getattr(ctx, 'engine', None), 'guarantees', None), 'alpha_calculator', None)
                if _alc and hasattr(_alc, 'update_hit_rate'):
                    _alc.update_hit_rate(won=_partial_net_pnl_usd > 0)
            except Exception:
                pass

            # [W11] Signal Quality -partial exit (no outcome, signal stays pending)
            # Partial exits don't close the signal -only full exit/flip does

            # Audit: log partial close + Discord alert
            if ctx.audit_manager:
                try:
                    _a_fee = _partial_exit_fee_usd
                    _a_oid = exec_result.get("order_id", exec_result.get("id", ""))
                    ctx.audit_manager.log_trade_execution(AuditTradeExecution(
                        execution_id=f"{asset}_{int(time.time())}_partial",
                        intent_id=f"{asset}_{ctx.tick_count}",
                        timestamp=time.time(), asset=asset, side=side,
                        quantity=base_quantity, price=fill_price, fee=_a_fee,
                        slippage_bps=market_data.get("spread_bps", 0.0),
                        exchange_order_id=str(_a_oid),
                    ), send_notification=True)
                except Exception as e:
                    logger.debug(f"[AUDIT] Execution log failed: {e}")

            # [Layer 3] Thesis Budget -record ACTUAL realized PnL from partial close
            if ctx.thesis_budget_governor and _partial_net_pnl_usd != 0:
                try:
                    _tb_side = "long" if pos_dir > 0 else "short"
                    _tb_strategy = old_pos.get("strategy", intent.quant_strategy_id or "momentum")
                    _tb_key = f"{asset.upper()}:{_tb_side}:{_tb_strategy}"
                    ctx.thesis_budget_governor.record_fill(_tb_key, _partial_net_pnl_usd)
                    logger.info(
                        f"[THESIS_BUDGET] PARTIAL EXIT {_tb_key}: pnl=${_partial_net_pnl_usd:+.2f} "
                        f"(is_win={_partial_net_pnl_usd >= 0})"
                    )
                except Exception as e:
                    logger.debug(f"[THESIS_BUDGET] record_fill (partial exit) failed: {e}")

            # Sync tranche state after partial close (prevents desync with paper_positions)
            if ctx.config.mode != RunMode.VERIFY:
                try:
                    ctx.fn_persist_tranche_state()
                except Exception as e:
                    logger.warning(f"[FIX-27] tranche_state persist failed: {e}")

        # =============================================================
        # BRANCH C: ENTRY or SCALE-IN  (new position / increasing)
        # =============================================================
        else:
            # [WIRE-4 BUGFIX] Any same-side add-on should already have been blocked
            # before order placement. If we reach this branch after a real fill,
            # preserve execution truth and only log the unexpected state.
            if (
                old_pos
                and old_pos.get("direction", 0) * direction_sign > 0
                and notional_usd > float(old_pos.get("notional", 0.0) or 0.0) + 1e-6
            ):
                _w4_allow_intent = getattr(intent, 'allow_add', True)
                _w4_allow_mce = market_data.get('_macro_crowd_effects', {}).get('allow_addon', True)
                if not _w4_allow_intent or not _w4_allow_mce:
                    logger.warning(
                        f"[WIRE-4 BUGFIX] {asset}: post-fill scale-in guard reached -"
                        f"intent.allow_add={_w4_allow_intent}, "
                        f"mce.allow_addon={_w4_allow_mce}; preserving filled execution"
                    )

            tranche = getattr(intent, 'tranche_target', 1)
            _m4_old_pos_backup = None  # [BUGFIX M4] init for non-flip path
            _entry_trade_fee_usd = _order_trade_fee_usd
            _entry_margin_opening_fee_usd = ctx.fn_compute_margin_opening_fee_usd(
                _order_fee_result,
                opening_notional_usd=notional_usd,
            )
            _entry_incremental_fee_usd = _entry_trade_fee_usd + _entry_margin_opening_fee_usd

            # [L3-03] FLIP GATE: direction flip costs 2x (close + open)
            # Check alpha covers double transaction cost before allowing flip
            _is_flip = (old_pos is not None
                        and old_pos.get("direction", 0) != 0
                        and old_pos.get("direction", 0) * direction_sign < 0)
            if _is_flip:
                _fg_alpha = getattr(intent, 'alpha_estimated_bps', 0.0)
                _fg_threshold = getattr(intent, 'alpha_threshold_bps', 0.0)
                # Flip requires 2x cost: the threshold already encodes 1x cost
                _fg_flip_threshold = _fg_threshold * 2.0
                if _fg_alpha < _fg_flip_threshold and _fg_flip_threshold > 0:
                    logger.info(
                        f"[FLIP_GATE] {asset}: alpha={_fg_alpha:.1f}bps < "
                        f"flip_cost={_fg_flip_threshold:.1f}bps (2x{_fg_threshold:.1f}) "
                        f"-flip BLOCKED, closing only"
                    )
                    # Convert flip to flat-only: set direction=0, target_exposure=0
                    intent.target_exposure = 0
                    intent.direction = 0
                    # Re-route to BRANCH A (full exit) by returning and re-entering
                    # Simpler: just don't flip -close position via existing exit logic
                    return {
                        "status": "FLIP_BLOCKED",
                        "reason": f"[FLIP_GATE] alpha={_fg_alpha:.1f} < 2xcost={_fg_flip_threshold:.1f}",
                        "asset": asset,
                    }

            # Direction flip: close old position first (compute full PnL)
            if old_pos and old_pos.get("direction", 0) * direction_sign < 0:
                _flip_entry = old_pos.get("entry_price", fill_price)
                _flip_dir = old_pos.get("direction", 0)
                _flip_notional = old_pos.get("notional", 0.0)
                _flip_open_notional_usd = max(0.0, float(notional_usd or 0.0) - float(_flip_notional or 0.0))
                _flip_fee_split = ctx.fn_split_trade_fee_usd(
                    trade_fee_usd=_order_trade_fee_usd,
                    total_notional_usd=notional_usd,
                    close_notional_usd=_flip_notional,
                )
                _flip_fee = float(_flip_fee_split.get("close_fee_usd", 0.0) or 0.0)
                _entry_trade_fee_usd = float(_flip_fee_split.get("open_fee_usd", 0.0) or 0.0)
                _entry_margin_opening_fee_usd = ctx.fn_compute_margin_opening_fee_usd(
                    _order_fee_result,
                    opening_notional_usd=_flip_open_notional_usd,
                )
                _entry_incremental_fee_usd = _entry_trade_fee_usd + _entry_margin_opening_fee_usd
                if _flip_entry > 0 and _flip_notional > 0:
                    _flip_pct = (fill_price - _flip_entry) / _flip_entry * _flip_dir
                    _flip_pnl = _flip_pct * _flip_notional
                    _flip_funding_pnl = old_pos.get("cumulative_funding_pnl", 0.0)  # [FIX-L1-02]
                    _flip_outcome = ctx.fn_build_realized_outcome(
                        position=old_pos,
                        gross_pnl_usd=_flip_pnl,
                        closed_notional_usd=_flip_notional,
                        exit_fee_usd=_flip_fee,
                        funding_pnl_usd=_flip_funding_pnl,
                    )
                    _flip_net_pnl_usd = _flip_outcome["net_pnl_usd"]
                    _flip_net_pnl_bps = _flip_outcome["net_pnl_bps"]
                    if ctx.account_sync and ctx.account_sync.dry_run:
                        ctx.account_sync.update_dry_run_pnl(_flip_net_pnl_usd)
                    if ctx.risk_manager:
                        ctx.risk_manager.record_realized_pnl(_flip_net_pnl_usd)
                    if ctx.existence_fuse:
                        ctx.existence_fuse.on_trade_close(_flip_net_pnl_usd)
                        # [FIX-P0-3] Feed real PnL into fuse rolling window
                        _fuse_equity = ctx.config.initial_capital
                        if ctx.account_sync:
                            _fuse_equity = ctx.account_sync.get_equity()
                        ctx.existence_fuse.record_pnl(
                            realized_pnl=_flip_net_pnl_usd,
                            current_equity=_fuse_equity,
                            trade_count=1,
                        )
                    _tilt_key = asset.upper().replace("/USD", "").replace("USD", "")
                    if _tilt_key in ctx.asset_trade_pnls:
                        ctx.asset_trade_pnls[_tilt_key].append(_flip_net_pnl_usd)
                    ctx.fn_record_realized_pnl_breakdown(
                        asset=asset,
                        side="long" if _flip_dir > 0 else "short",
                        strategy=old_pos.get("strategy", intent.quant_strategy_id or "momentum"),
                        pnl_usd=_flip_net_pnl_usd,
                        pnl_bps=_flip_net_pnl_bps,
                        exit_type="FLIP",
                    )
                    logger.info(
                        f"[PAPER_PNL] {asset} FLIP CLOSE: entry=${_flip_entry:.2f} "
                        f"exit=${fill_price:.2f} dir={_flip_dir:+.0f} "
                        f"gross=${_flip_pnl:+.2f} net=${_flip_net_pnl_usd:+.2f} "
                        f"({ _flip_pct*100:+.2f}% gross, {_flip_net_pnl_bps:+.1f}bps net) "
                        f"funding=${_flip_funding_pnl:+.4f}"
                    )
                    # [HIT-RATE] Update alpha gate performance factor from flip outcome
                    try:
                        _alc = getattr(getattr(getattr(ctx, 'engine', None), 'guarantees', None), 'alpha_calculator', None)
                        if _alc and hasattr(_alc, 'update_hit_rate'):
                            _alc.update_hit_rate(won=_flip_net_pnl_usd > 0)
                    except Exception:
                        pass
                    # Thesis budget for the flipped-out position
                    if ctx.thesis_budget_governor:
                        try:
                            _tb_side = "long" if _flip_dir > 0 else "short"
                            _tb_strategy = old_pos.get("strategy", "momentum")
                            _tb_key = f"{asset.upper()}:{_tb_side}:{_tb_strategy}"
                            ctx.thesis_budget_governor.record_fill(_tb_key, _flip_net_pnl_usd)
                        except Exception as _tb_err:
                            logger.warning(f"[SOTA L1] ThesisBudget record_fill (flip) failed: {_tb_err}")
                    # [FIX-9] Shadow ledger -record close side of flip
                    if ctx.p0_integrator is not None and ctx.p0_integrator:
                        try:
                            _flip_close_side = "BUY" if _flip_dir < 0 else "SELL"
                            _flip_qty = _flip_notional / fill_price if fill_price > 0 else 0.0
                            _flip_oid = f"paper_{asset}_flip_{int(datetime.now(timezone.utc).timestamp())}"
                            _shadow_fill_ok = ctx.p0_integrator.record_fill(
                                asset=asset, side=_flip_close_side,
                                size=_flip_qty, price=fill_price,
                                order_id=_flip_oid, fee=_flip_fee,  # [FIX-L1-01] was 0.0
                                realized_pnl=_flip_net_pnl_usd,
                                extra={
                                    "exit_trigger": ctx.exit_trigger_tag.get(asset, "T13_FLIP"),
                                    "funding_pnl": _flip_funding_pnl,
                                    "trade_fee_usd": _flip_fee,
                                    "margin_opening_fee_usd": 0.0,
                                    # [FIX 2026-04-22] PnL attribution
                                    "primary_agent": getattr(ctx.intent, "primary_agent", "") or "",
                                },
                            )
                            _note_shadow_fill(bool(_shadow_fill_ok))
                        except Exception as _f9_err:
                            logger.debug(f"[FIX-9] Shadow ledger flip close record failed: {_f9_err}")
                    # [W10] Trade Attributor -record flip exit
                    if ctx.trade_attributor:
                        try:
                            ctx.trade_attributor.record_exit(
                                asset=asset, price=fill_price, fee=_flip_fee,  # [FIX-L1-01] was 0.0
                                notional=_flip_notional, gross_pnl=_flip_pnl,
                                exit_type="FLIP",
                            )
                        except Exception as _ta_err:
                            logger.debug(f"[W10] TradeAttributor record_exit (flip) failed: {_ta_err}")

                    # [W11] Signal Quality -record outcome (flip exit)
                    if ctx.sq_tracker:
                        try:
                            _sq_hold = ctx.tick_count - old_pos.get("entry_tick", 0)
                            ctx.sq_tracker.record_outcome(
                                asset=asset, exit_price=fill_price,
                                gross_pnl=_flip_pnl, hold_ticks=_sq_hold,
                                exit_reason="FLIP_EXIT",
                            )
                        except Exception as e:
                            logger.debug(f"[FIX-47] signal_quality record_outcome failed: {e}")

                    if ctx.experience_buffer is not None:
                        try:
                            _exp_hold_bars = 1
                            _exp_entry_time = old_pos.get("entry_time")
                            if _exp_entry_time:
                                _exp_entry_dt = datetime.fromisoformat(
                                    str(_exp_entry_time).replace("Z", "+00:00")
                                )
                                if _exp_entry_dt.tzinfo is None:
                                    _exp_entry_dt = _exp_entry_dt.replace(tzinfo=timezone.utc)
                                _exp_hold_bars = max(
                                    1,
                                    int(
                                        (
                                            datetime.now(timezone.utc) - _exp_entry_dt
                                        ).total_seconds() / 14400
                                    ),
                                )
                            ctx.experience_buffer.record_outcome(
                                asset=asset,
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                realized_pnl=_flip_net_pnl_usd,
                                bars_held=_exp_hold_bars,
                                realized_pnl_bps=_flip_net_pnl_bps,
                                exit_reason=intent.veto_reason or "flip_exit",
                            )
                        except Exception as e:
                            logger.debug(f"[EXPERIENCE] record_outcome (flip exit) failed: {e}")

                    # [EA-4b] Exit Alpha Tracker -record flip exit
                    if ctx.ea_tracker:
                        try:
                            _ea_trigger = ctx.exit_trigger_tag.pop(asset, "T13_FLIP")
                            ctx.ea_tracker.record_exit(
                                asset=asset, exit_price=fill_price,
                                trigger=_ea_trigger, pnl_usd=_flip_pnl,
                                tick=ctx.tick_count, exit_type="FLIP",
                                regime_at_exit=market_data.get("regime_state", ""),
                            )
                        except Exception:
                            pass

                # [BUGFIX M4] Atomic position flip -save old pos before pop, restore on failure
                _m4_old_pos_backup = ctx.paper_positions.get(asset)
                ctx.paper_positions.pop(asset, None)
                if ctx.gambler_exit:
                    try:
                        ctx.gambler_exit.clear_entry(asset)
                    except Exception as e:
                        logger.debug(f"[FIX-47] gambler_exit.clear_entry failed: {e}")
                if ctx.exit_alpha:
                    try:
                        ctx.exit_alpha.reset_for_asset(asset)
                    except Exception as e:
                        logger.debug(f"[FIX-47] exit_alpha.reset_for_asset failed: {e}")
                # [WIRE-2] Remove flipped position from adaptive stop
                if ctx.adaptive_stop:
                    try:
                        ctx.adaptive_stop.remove_position(asset)
                    except Exception as e:
                        logger.debug(f"[FIX-47] adaptive_stop.remove_position failed: {e}")

            # [BUGFIX M4] Wrap new position creation in try/except -restore old pos on failure
            _entry_fee_usd = float(_entry_incremental_fee_usd or 0.0)
            _carried_entry_fees_usd = 0.0
            if old_pos and old_pos.get("direction", 0) * direction_sign > 0:
                _carried_entry_fees_usd = ctx.fn_get_position_entry_fee_usd(old_pos)
            _effective_entry_exposure = 0.0
            if account_equity and account_equity > 0:
                _effective_entry_exposure = max(0.0, float(notional_usd) / float(account_equity))
            _m4_new_pos = {
                "exposure": _effective_entry_exposure,
                "original_entry_exposure": old_pos.get("original_entry_exposure", _effective_entry_exposure) if old_pos else _effective_entry_exposure,
                "direction": direction_sign,
                "tranche": tranche,
                "entry_price": fill_price,
                "notional": notional_usd,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "strategy": intent.quant_strategy_id or "momentum",
                "entry_alpha_est": getattr(intent, 'alpha_estimated_bps', 0.0),
                "mode_at_entry": getattr(intent, 'system_mode', 'NORMAL'),
                "phase_at_entry": market_data.get('phase', 'UNKNOWN'),
                "crack_weight": market_data.get('crack_weight', 0.0),
                "regime_leverage": getattr(intent, 'regime_leverage', 1.0),  # [AUDIT-D3]
                "entry_tick": ctx.tick_count,  # [AC-1] for min hold time
                "entry_fee_usd": _entry_fee_usd,
                "cumulative_entry_fees_usd": _carried_entry_fees_usd + _entry_fee_usd,
                "entry_trade_fee_usd": _entry_trade_fee_usd,
                "entry_margin_opening_fee_usd": _entry_margin_opening_fee_usd,
                "latest_drl_direction": float(agent_signals.get("drl_direction", 0.0) or 0.0),
                "latest_drl_confidence": float(agent_signals.get("drl_confidence", 0.0) or 0.0),
                "latest_drl_action": float(agent_signals.get("ensemble_action", 0.0) or 0.0),
                "latest_drl_ts": datetime.now(timezone.utc).isoformat(),
            }
            try:
                ctx.paper_positions[asset] = _m4_new_pos
            except Exception as _m4_err:
                logger.error(f"[BUGFIX M4] Position flip FAILED for {asset}: {_m4_err} -restoring old position")
                if _m4_old_pos_backup is not None:
                    ctx.paper_positions[asset] = _m4_old_pos_backup
                raise
            if ctx.gambler_exit:
                try:
                    ctx.gambler_exit.record_entry(asset)
                except Exception:
                    pass
            if ctx.exit_alpha:
                try:
                    ctx.exit_alpha.reset_for_asset(asset)
                except Exception:
                    pass
            # [WIRE-2] Register new position with adaptive stop manager
            if ctx.adaptive_stop:
                try:
                    _w2_dir = StopDirection.LONG if direction_sign > 0 else StopDirection.SHORT
                    # Apply regime multiplier at entry time
                    _w2_regime = market_data.get("regime_state", "NEUTRAL")
                    _w2_rm = ctx.adaptive_stop_regime_mult.get(_w2_regime, 1.0)
                    ctx.adaptive_stop.config.atr_multiplier = 3.5 * _w2_rm
                    ctx.adaptive_stop.register_position(
                        symbol=asset,
                        entry_price=fill_price,
                        position_size=abs(intent.target_exposure),
                        direction=_w2_dir,
                    )
                    # Enforce max stop distance 15% safety net
                    _w2_st = ctx.adaptive_stop.get_stop_state(asset)
                    if _w2_st and fill_price > 0:
                        _w2_dist = abs(_w2_st.current_stop - fill_price) / fill_price
                        if _w2_dist > 0.15:
                            if _w2_dir == StopDirection.LONG:
                                _w2_st.current_stop = fill_price * 0.85
                            else:
                                _w2_st.current_stop = fill_price * 1.15
                            _w2_st.initial_stop = _w2_st.current_stop
                    logger.info(
                        f"[WIRE-2] {asset}: stop registered, regime={_w2_regime} "
                        f"mult={_w2_rm:.1f} ->ATRx{3.5*_w2_rm:.1f}"
                    )
                except Exception as _w2_err:
                    logger.debug(f"[WIRE-2] Adaptive stop register failed: {_w2_err}")
            # W8: Record position open for strategy perf tracking
            if ctx.feedback_loop:
                try:
                    ctx.feedback_loop.open_position(
                        trade_id=f"{asset}_{int(time.time())}",
                        asset=asset,
                        entry_price=fill_price,
                        size=base_quantity,
                        direction=direction_sign,
                        signal_source=getattr(intent, 'quant_strategy_id', 'unknown'),
                        signal_confidence=getattr(intent, 'quant_confidence', 0.5),
                        mode=getattr(intent, 'system_mode', 'NORMAL'),
                        regime=market_data.get("regime_state", "UNKNOWN"),
                    )
                except Exception as _fb_err:
                    logger.warning(f"[SOTA L1] FeedbackLoop open_position failed: {_fb_err}")
            # Shadow ledger -entry fill
            if ctx.p0_integrator is not None and ctx.p0_integrator:
                try:
                    fill_fee = _entry_fee_usd
                    order_id = exec_result.get("order_id", exec_result.get("id", f"paper_{asset}_{int(datetime.now(timezone.utc).timestamp())}"))
                    _shadow_fill_ok = ctx.p0_integrator.record_fill(
                        asset=asset, side=side,
                        size=base_quantity, price=fill_price,
                        order_id=str(order_id), fee=fill_fee,
                        realized_pnl=0.0,
                        extra={
                            "trade_fee_usd": _entry_trade_fee_usd,
                            "margin_opening_fee_usd": _entry_margin_opening_fee_usd,
                            "requires_margin": bool(_order_fee_result.get("requires_margin", False)),
                            # [FIX 2026-04-22] PnL attribution — entry-side stamp
                            "primary_agent": getattr(ctx.intent, "primary_agent", "") or "",
                        },
                    )
                    _note_shadow_fill(bool(_shadow_fill_ok))
                except Exception as e:
                    logger.debug(f"[SHADOW_LEDGER] record_fill (entry) failed: {e}")

            # [W10] Trade Attributor -record entry
            if ctx.trade_attributor:
                try:
                    _ta_fee = _entry_fee_usd
                    ctx.trade_attributor.record_entry(
                        asset=asset, price=fill_price, fee=_ta_fee,
                        notional=notional_usd, direction=direction_sign,
                        strategy=intent.quant_strategy_id or "momentum",
                        regime=market_data.get("regime_state", "UNKNOWN"),
                        mode=getattr(intent, 'system_mode', 'NORMAL'),
                    )
                except Exception as _ta_err:
                    logger.debug(f"[W10] TradeAttributor record_entry failed: {_ta_err}")

            # [W11] Signal Quality -record entry signal
            if ctx.sq_tracker:
                try:
                    ctx.sq_tracker.record_signal(
                        asset=asset,
                        strategy=intent.quant_strategy_id or "momentum",
                        direction=direction_sign,
                        alpha_est_bps=getattr(intent, 'alpha_estimated_bps', 0.0),
                        confidence=getattr(intent, 'quant_confidence', 0.5),
                        regime=market_data.get("regime_state", "UNKNOWN"),
                        signal_strength=abs(market_data.get("quant_direction", 0)),
                        tick=ctx.tick_count,
                        price=fill_price,
                    )
                except Exception as e:
                    logger.debug(f"[FIX-47] signal_quality record_entry failed: {e}")

            # [EA-2] Exit Alpha Tracker -record new entry for peak tracking
            if ctx.ea_tracker:
                try:
                    ctx.ea_tracker.record_entry(
                        asset=asset, price=fill_price, direction=direction_sign,
                        strategy=intent.quant_strategy_id or "momentum",
                        regime=market_data.get("regime_state", "UNKNOWN"),
                        tick=ctx.tick_count, notional=notional_usd,
                        mode=getattr(intent, 'system_mode', 'NORMAL'),
                    )
                except Exception:
                    pass

            # Audit: log entry fill + Discord alert
            if ctx.audit_manager:
                try:
                    _a_fee = _entry_fee_usd
                    _a_oid = exec_result.get("order_id", exec_result.get("id", ""))
                    ctx.audit_manager.log_trade_execution(AuditTradeExecution(
                        execution_id=f"{asset}_{int(time.time())}_entry",
                        intent_id=f"{asset}_{ctx.tick_count}",
                        timestamp=time.time(), asset=asset, side=side,
                        quantity=base_quantity, price=fill_price, fee=_a_fee,
                        slippage_bps=market_data.get("spread_bps", 0.0),
                        exchange_order_id=str(_a_oid),
                    ), send_notification=True)
                except Exception as e:
                    logger.debug(f"[AUDIT] Execution log failed: {e}")

            # [FIX-FEE-DOUBLE-COUNT] Entry fee was being deducted TWICE from equity:
            # 1) Here at entry time via paper_fee_cost_pnl_delta
            # 2) Again at exit time via _build_realized_outcome (net_pnl = gross - entry_fee - exit_fee)
            # The entry fee is stored in position["cumulative_entry_fees_usd"] and properly
            # allocated at close by _build_realized_outcome. Removing this duplicate deduction.
            # Impact: ~$145 equity undercount over 70 trades (avg entry_fee ~$9.68).
            exec_result["paper_fee_cost_pnl_delta"] = 0.0

            # [v3.3-C7] Two-tier authority-aware stops
            try:
                from execution.execution_manager import OrderSide as _C7Side
                _c7_symbol = f"{asset}/USD"
                _c7_dir_str = "long" if intent.direction > 0 else "short"
                _c7_stop_side = _C7Side.SELL if intent.direction > 0 else _C7Side.BUY

                if ctx.stop_authority and STOP_AUTHORITY_AVAILABLE:
                    # Map system mode
                    _sys_mode_str = getattr(intent, 'system_mode', 'NORMAL')
                    try:
                        _stop_mode = StopSystemMode(_sys_mode_str)
                    except (ValueError, KeyError):
                        _stop_mode = StopSystemMode.NORMAL

                    _auth = StopAuthorityState(
                        system_mode=_stop_mode,
                        macro_veto_active=bool(market_data.get("risk_veto", False)),
                    )

                    # L4-09: Regime-adaptive entry stop distance
                    _regime_stop_mult = {
                        'PANIC_SELLOFF': 1.25,       # Wide -big moves normal
                        'EXTREME_VOLATILITY': 1.25,  # Wide -high vol
                        'MOMENTUM_RALLY': 1.15,      # Medium-wide -allow pullbacks
                        'VOLATILE_CHOP': 0.8,        # Tight -cut losses fast
                        'WEAK_CONSOLIDATION': 0.9,   # Tight-ish -low edge
                    }.get(market_data.get("regime_state", ""), 1.0)

                    # SOL: wider stops due to higher vol
                    if 'SOL' in asset:
                        _sol_soft = 0.03 * _regime_stop_mult
                        _sol_hard = 0.06 * _regime_stop_mult
                        _sol_cfg = StopAuthorityConfig(soft_stop_pct=_sol_soft, hard_stop_pct=_sol_hard)
                        _stop_dec = get_stop_authority_manager(_sol_cfg).calculate_stops(
                            asset=asset, entry_price=fill_price,
                            direction=_c7_dir_str, authority_state=_auth,
                        )
                    else:
                        if _regime_stop_mult != 1.0:
                            _btc_soft = 0.02 * _regime_stop_mult
                            _btc_hard = 0.04 * _regime_stop_mult
                            _reg_cfg = StopAuthorityConfig(soft_stop_pct=_btc_soft, hard_stop_pct=_btc_hard)
                            _stop_dec = get_stop_authority_manager(_reg_cfg).calculate_stops(
                                asset=asset, entry_price=fill_price,
                                direction=_c7_dir_str, authority_state=_auth,
                            )
                        else:
                            _stop_dec = ctx.stop_authority.calculate_stops(
                                asset=asset, entry_price=fill_price,
                                direction=_c7_dir_str, authority_state=_auth,
                            )

                    # Always place HARD stop on exchange
                    if _stop_dec.hard_stop_active and _stop_dec.hard_stop_price:
                        _c7_result = ctx.execution_manager.place_stop_loss(
                            symbol=_c7_symbol, side=_c7_stop_side,
                            size=base_quantity,
                            stop_price=round(_stop_dec.hard_stop_price, 2),
                        )
                        _soft_status = "ACTIVE" if _stop_dec.soft_stop_active else f"SUSPENDED({_stop_dec.soft_suspension_reason})"
                        if _c7_result.success:
                            logger.info(
                                f"[STOP] {asset}: HARD=${_stop_dec.hard_stop_price:,.2f} | "
                                f"soft={_soft_status} @${_stop_dec.soft_stop_price:,.2f} | "
                                f"mode={_sys_mode_str}"
                            )
                        else:
                            logger.warning(f"[STOP] {asset}: HARD stop FAILED -{_c7_result.error_message}")
                else:
                    # Fallback: original single-tier behavior
                    _c7_stop_pct = 0.05 if 'SOL' in asset else 0.03
                    if intent.direction > 0:
                        _c7_stop_price = fill_price * (1 - _c7_stop_pct)
                    else:
                        _c7_stop_price = fill_price * (1 + _c7_stop_pct)
                    _c7_result = ctx.execution_manager.place_stop_loss(
                        symbol=_c7_symbol, side=_c7_stop_side,
                        size=base_quantity, stop_price=round(_c7_stop_price, 2),
                    )
                    if _c7_result.success:
                        logger.info(f"[STOP] {asset}: placed at ${_c7_stop_price:,.2f} (fallback {_c7_stop_pct:.0%})")
            except Exception as _c7_err:
                logger.warning(f"[STOP] {asset}: stop placement failed (non-fatal): {_c7_err}")

            # [Section E] ConfidenceScorer -record signal at entry (feedback loop)
            # Singleton shared with v36 engine: outcomes update confidences for next tick
            if ctx.confidence_scorer is not None:
                try:
                    regime = market_data.get("regime_state", "UNKNOWN")
                    sig_ts = ctx.confidence_scorer.record_signal(
                        strategy_name=intent.quant_strategy_id or "momentum",
                        direction=intent.direction,
                        confidence=intent.quant_confidence,
                        expected_pnl_bps=intent.alpha_estimated_bps,
                        regime=regime,
                    )
                    if sig_ts:
                        ctx.confidence_signal_times[asset] = sig_ts
                        logger.info(
                            f"[CONFIDENCE] {asset} signal recorded: "
                            f"strategy={intent.quant_strategy_id}, dir={intent.direction:+.2f}"
                        )
                except Exception as e:
                    logger.debug(f"[CONFIDENCE] record_signal failed: {e}")

            # [FIX-BRANCH-C-PERSIST] Persist paper position + tranche state after
            # new entry. BRANCH A (full exit) and BRANCH B (partial close) both have
            # these calls, but BRANCH C was missing them. Without this:
            # 1. Paper position only saved by periodic debounced save (up to 5s gap)
            # 2. Tranche state (level, last_escalation) never persisted for new entries
            #    because the persist at line 8960 runs BEFORE the paper position exists
            # On crash-restart, missing tranche state causes last_escalation=None
            # → bars_since=999 → premature T2 escalation on first tick.
            ctx.fn_save_paper_positions(force=True)
            if ctx.config.mode != RunMode.VERIFY:
                try:
                    ctx.fn_persist_tranche_state()
                except Exception as e:
                    logger.warning(f"[FIX-BRANCH-C-PERSIST] tranche_state persist failed: {e}")

    # =====================================================================
    # Phase 1.10: Record position entry time for max hold enforcement
    # =====================================================================
    if intent.target_exposure != 0 and asset not in ctx.position_entry_times:
        # New position entry -record time and mode
        mode = market_data.get("volatility_regime", "NORMAL").upper()
        if mode not in ("NORMAL", "OPPORTUNITY"):
            mode = "NORMAL"
        ctx.position_entry_times[asset] = {
            "entry_time": datetime.now(timezone.utc),
            "mode": mode,
        }
        max_h = get_rule("max_hold_hours", mode == "OPPORTUNITY")  # [RULETABLE]
        logger.info(
            f"[MAX_HOLD] {asset} position opened -mode={mode}, max_hold={max_h}h"
        )
    elif intent.target_exposure == 0 and asset in ctx.position_entry_times:
        # Position closed -clear entry time
        entry_info = ctx.position_entry_times.pop(asset)
        held_hours = (datetime.now(timezone.utc) - entry_info["entry_time"]).total_seconds() / 3600
        _hold_limit = get_rule("max_hold_hours", entry_info["mode"] == "OPPORTUNITY")  # [RULETABLE]
        logger.info(
            f"[MAX_HOLD] {asset} position closed after {held_hours:.1f}h "
            f"(limit was {_hold_limit}h)"
        )

    if fill_price > 0:
        _recent_slippage_bps = 0.0
        if current_price > 0:
            if side.lower() == "buy":
                _recent_slippage_bps = ((fill_price - current_price) / current_price) * 10000
            else:
                _recent_slippage_bps = ((current_price - fill_price) / current_price) * 10000
        ctx.recent_fill_state[asset] = {
            "fill_price": float(fill_price),
            "side": str(side).lower(),
            "timestamp": time.time(),
            "order_type": str(order_type).lower(),
            "slippage_bps": float(_recent_slippage_bps),
        }

    # =====================================================================
    # Compounding: Update paper equity with simulated P&L
    # For paper mode, track cumulative fill fees as a proxy for cost drag.
    # Real compounding happens via account_sync.get_equity() on next tick.
    # =====================================================================
    if ctx.account_sync and ctx.account_sync.dry_run:
        try:
            fee_cost_delta = float(exec_result.get("paper_fee_cost_pnl_delta", 0.0) or 0.0)
            if fee_cost_delta:
                ctx.account_sync.update_dry_run_pnl(fee_cost_delta)
        except Exception as _fee_err:
            logger.warning(f"[SOTA L1] Fee cost deduction failed: {_fee_err}")

    # [Phase 4B] Record fill via AntiChurnManager (AC-2 tracking + AC-5 budget)
    ctx.anti_churn.record_fill(asset, ctx.tick_count)

    # =====================================================================
    # Bug #2 fix: Sync RiskManager equity after trade execution
    # RiskManager uses current_balance for drawdown calculations.
    # Without this, it accumulates error over time.
    # =====================================================================
    if ctx.risk_manager and ctx.account_sync:
        try:
            new_equity = ctx.account_sync.get_equity()
            ctx.risk_manager.update_balance(new_equity)
        except Exception as _eq_err:
            logger.warning(f"[SOTA L1] RiskManager balance sync failed: {_eq_err}")

    # T23: Log execution quality (after fill_price is determined)
    if ctx.exec_quality_logger and fill_price > 0:
        try:
            _intended_price = current_price
            _slippage_bps = abs(fill_price - _intended_price) / _intended_price * 10000 if _intended_price > 0 else 0
            if side.lower() == "buy":
                _realized_slip = (fill_price - _intended_price) / _intended_price * 10000
            else:
                _realized_slip = (_intended_price - fill_price) / _intended_price * 10000
            _filled_size = exec_result.get("filled_size", base_quantity)
            _record = ExecutionQualityRecord(
                timestamp=datetime.now(timezone.utc),
                execution_id=f"{asset}_{int(time.time() * 1000)}",
                asset=asset,
                side=side.lower(),
                intended_quantity=base_quantity,
                executed_quantity=_filled_size,
                arrival_price=_intended_price,
                avg_execution_price=fill_price,
                num_slices=_num_slices,
                order_type=order_type.lower(),
                expected_slippage_bps=market_data.get("spread_bps", 10.0) / 2,
                realized_slippage_bps=_realized_slip,
                fill_ratio=_filled_size / base_quantity if base_quantity > 0 else 1.0,
                time_to_fill_sec=exec_result.get("time_to_fill_sec", 0.0),
                cancel_count=0,
                execution_source=EQSource.BASELINE,
                rl_advice_used=False,
                volatility=market_data.get("volatility", 0.0),
                spread_bps=market_data.get("spread_bps", 0.0),
            )
            ctx.exec_quality_logger.log(_record)
            if _slippage_bps > 5.0:
                logger.info(f"[EXEC_QUALITY] {asset}: slippage={_slippage_bps:.1f}bps (high)")
        except Exception as e:
            logger.debug(f"[EXEC_QUALITY] Log failed: {e}")

    # W1: Feed market impact calibration with fill data
    if ctx.impact_cal_table and fill_price > 0 and base_quantity > 0:
        try:
            _realized_impact = abs(fill_price - current_price) / current_price * 10000
            _sample = CalibrationSample(
                asset=asset,
                side=side.lower(),
                size_usd=notional_usd,
                expected_impact_bps=market_data.get("spread_bps", 3.0),
                realized_impact_bps=_realized_impact,
                execution_time_ms=exec_result.get("latency_ms", 0),
                volatility=market_data.get("volatility", 0.02),
                spread_bps=market_data.get("spread_bps", 3.0),
                timestamp=time.time(),
            )
            ctx.impact_cal_table.add_sample(_sample)
            if ctx.impact_cal_table._stats.get("total_samples", 0) % 20 == 0:
                ctx.impact_cal_table.calibrate_online()
                logger.info("[IMPACT_CAL] Online calibration updated")
        except Exception as e:
            logger.debug(f"[IMPACT_CAL] Sample recording failed: {e}")

    if (
        (ctx.config.mode == RunMode.PAPER or (ctx.account_sync and ctx.account_sync.dry_run))
        and not bool(exec_result.get("shadow_fill_recorded", False))
    ):
        exec_result["shadow_fill_missing"] = True
        logger.warning(
            f"[P0-2] {asset}: execution completed without shadow-ledger fill ack "
            f"(status={exec_result.get('status', 'UNKNOWN')}, side={side}, qty={base_quantity:.6f})"
        )

    return exec_result


