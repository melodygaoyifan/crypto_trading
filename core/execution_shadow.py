"""
================================================================================
EXECUTION SHADOW MODE - Dual-Path Comparison for Safe Migration
================================================================================
Version: 1.0.0
Purpose: Compare old (_execute_intent) and new (execute_intent_v2) code paths
         without risking live state.

APPROACH:
    During paper trading, after the OLD _execute_intent runs:
    1. Snapshot the exec_result from old path
    2. Deep-copy the pre-execution state
    3. Run execute_intent_v2 on the COPY
    4. Compare result dicts key-by-key
    5. Log differences to data/shadow_exec_comparison.jsonl

    The OLD result is ALWAYS used for real state changes.
    The NEW result is ONLY used for comparison logging.

ACTIVATION:
    Set ENABLE_EXECUTION_SHADOW_MODE = True in main.py or config.
    Shadow mode adds ~50ms per execution (deep copy + re-run).

ANALYSIS:
    python -c "
    import json
    diffs = [json.loads(l) for l in open('data/shadow_exec_comparison.jsonl')]
    mismatches = [d for d in diffs if d.get('match') == False]
    print(f'{len(mismatches)}/{len(diffs)} mismatches')
    "
================================================================================
"""

import copy
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SHADOW_LOG_FILE = Path("data") / "shadow_exec_comparison.jsonl"

# Keys that are expected to differ (timestamps, latencies, etc.)
IGNORE_KEYS = frozenset({
    "timestamp", "latency_ms", "time_to_fill_sec",
    "execution_id", "_audit_intent_id",
    "shadow_fill_missing",  # timing-dependent
})

# Keys that MUST match for safety
CRITICAL_KEYS = frozenset({
    "status", "success", "side", "filled_price",
    "filled_size", "paper_pnl_usd", "paper_pnl_bps",
    "shadow_fill_recorded", "notional_usd",
})


def compare_exec_results(
    old_result: Dict[str, Any],
    new_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare two execution results key-by-key.

    Returns a comparison report dict.
    """
    all_keys = set(old_result.keys()) | set(new_result.keys())
    diffs = {}
    critical_match = True

    for key in sorted(all_keys):
        if key in IGNORE_KEYS:
            continue
        old_val = old_result.get(key)
        new_val = new_result.get(key)

        # Float comparison with tolerance
        if isinstance(old_val, float) and isinstance(new_val, float):
            if abs(old_val - new_val) > 1e-6:
                diffs[key] = {"old": old_val, "new": new_val}
                if key in CRITICAL_KEYS:
                    critical_match = False
        elif old_val != new_val:
            diffs[key] = {"old": str(old_val)[:100], "new": str(new_val)[:100]}
            if key in CRITICAL_KEYS:
                critical_match = False

    return {
        "match": len(diffs) == 0,
        "critical_match": critical_match,
        "diff_count": len(diffs),
        "diffs": diffs,
    }


def log_shadow_comparison(
    asset: str,
    tick_count: int,
    old_result: Dict[str, Any],
    new_result: Dict[str, Any],
    elapsed_ms: float,
    error: Optional[str] = None,
) -> None:
    """Log a shadow comparison result to JSONL file."""
    comparison = compare_exec_results(old_result, new_result) if new_result else {
        "match": False, "critical_match": False, "diff_count": -1,
        "diffs": {"error": error or "new_path_failed"},
    }

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "asset": asset,
        "tick": tick_count,
        "old_status": old_result.get("status", ""),
        "new_status": new_result.get("status", "") if new_result else "ERROR",
        "match": comparison["match"],
        "critical_match": comparison["critical_match"],
        "diff_count": comparison["diff_count"],
        "shadow_elapsed_ms": round(elapsed_ms, 1),
    }

    if not comparison["match"]:
        record["diffs"] = comparison["diffs"]

    try:
        SHADOW_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SHADOW_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        logger.debug(f"[SHADOW] Log write failed: {e}")

    if not comparison["critical_match"]:
        logger.warning(
            f"[SHADOW] {asset} CRITICAL MISMATCH: "
            f"old_status={old_result.get('status')} "
            f"new_status={new_result.get('status', 'ERROR')} "
            f"diffs={comparison['diff_count']}"
        )
    elif not comparison["match"]:
        logger.info(
            f"[SHADOW] {asset} minor diffs: {comparison['diff_count']} "
            f"(non-critical, elapsed={elapsed_ms:.0f}ms)"
        )


async def run_shadow_execution(
    runner: Any,
    asset: str,
    intent: Any,
    market_data: Dict[str, Any],
    old_result: Dict[str, Any],
    agent_signals: Optional[Dict[str, Any]] = None,
    pre_exec_snapshot: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run the new execution path in shadow mode and compare with old result.

    This function:
    1. Uses pre-execution state snapshot (captured BEFORE old path ran)
    2. Builds ExecutionContext with snapshot state
    3. Runs execute_intent_v2 on the snapshot
    4. Compares with old_result
    5. Logs differences

    The old_result is ALWAYS the authoritative one.
    """
    from core.execution_context import ExecutionContext

    try:
        t0 = time.monotonic()

        # Build ctx from runner (for component refs) then overlay pre-exec snapshot
        ctx = ExecutionContext.build_from_runner(runner)

        if pre_exec_snapshot:
            # Use PRE-EXECUTION state — this is what old path saw when it started
            ctx.paper_positions = pre_exec_snapshot["paper_positions"]
            ctx.position_entry_times = pre_exec_snapshot["position_entry_times"]
            ctx.rebuild_cooldown = pre_exec_snapshot["rebuild_cooldown"]
            ctx.exit_trigger_tag = pre_exec_snapshot["exit_trigger_tag"]
            ctx.ac2_fill_ticks = pre_exec_snapshot.get("ac2_fill_ticks", {})
        else:
            # Fallback: deep copy current (post-execution) state — less accurate
            ctx.paper_positions = copy.deepcopy(runner._paper_positions)
            ctx.position_entry_times = copy.deepcopy(runner._position_entry_times)
            ctx.rebuild_cooldown = copy.deepcopy(runner._rebuild_cooldown)
            ctx.exit_trigger_tag = copy.deepcopy(getattr(runner, '_exit_trigger_tag', {}))

        # These are less critical — post-exec copy is acceptable
        ctx.dashboard_asset_runtime = copy.deepcopy(runner._dashboard_asset_runtime)
        ctx.asset_trade_pnls = {k: copy.copy(v) for k, v in runner._asset_trade_pnls.items()}
        ctx.orphaned_stops = set(runner._orphaned_stops)
        ctx.recent_fill_state = copy.deepcopy(getattr(runner, '_recent_fill_state', {}))

        # Intent and market_data already deep-copied by caller
        shadow_intent = intent
        shadow_market_data = market_data

        # Run new path
        from core.execution_service import execute_intent_v2
        new_result = await execute_intent_v2(
            ctx, asset, shadow_intent, shadow_market_data,
            agent_signals=agent_signals,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000

        log_shadow_comparison(
            asset=asset,
            tick_count=runner._tick_count,
            old_result=old_result,
            new_result=new_result,
            elapsed_ms=elapsed_ms,
        )

    except Exception as e:
        elapsed_ms = (time.monotonic() - t0) * 1000 if 't0' in dir() else 0
        log_shadow_comparison(
            asset=asset,
            tick_count=getattr(runner, '_tick_count', 0),
            old_result=old_result,
            new_result={},
            elapsed_ms=elapsed_ms,
            error=f"{type(e).__name__}: {str(e)[:200]}",
        )
        logger.debug(f"[SHADOW] {asset} new path failed: {e}")
