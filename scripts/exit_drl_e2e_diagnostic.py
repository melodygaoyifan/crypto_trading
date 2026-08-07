#!/usr/bin/env python3
"""
End-to-end Exit-DRL wire-up diagnostic.

Exercises every link in the chain without booting the full runner. Lights up
each stage and reports OK / FAIL / SKIP per asset:

  1. Module imports (agent / ledger / kill switch / promotion gate)
  2. Checkpoint loadability (per asset)
  3. Per-asset mode helpers (mode_for_asset, should_act_on)
  4. Predict + log to data/exit_drl_shadow.jsonl
  5. Outcome ledger record_open + record_prediction + record_close round-trip
  6. Kill switch lifecycle (record_promotion → record_close → should_demote)
  7. Promotion-gate evaluate (current evidence)
  8. Promotion-state JSON has the override stamp
  9. tick_exit_triggers.py has the System-3 PARTIAL_EXIT bridge
 10. main.py initialization block with the per-asset mode override

Usage:
    python -X utf8 scripts/exit_drl_e2e_diagnostic.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _section(title: str) -> None:
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def _skip(msg: str) -> None:
    print(f"  · {msg}  [skipped]")


failures = []


def main() -> int:
    print("=" * 72)
    print("  HMATS Exit-DRL end-to-end wire-up diagnostic")
    print(f"  ts={time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}")
    print("=" * 72)

    # ────────────────────────────────────────────────────────────
    # Stage 1 — module imports
    # ────────────────────────────────────────────────────────────
    _section("STAGE 1 — module imports")
    try:
        from agents.exit_drl_agent import (
            ExitAction, ExitDRLAgent, ExitDRLMode, get_exit_drl_agent,
            reset_singleton as _reset_agent,
        )
        _ok("agents.exit_drl_agent")
    except Exception as e:
        _fail(f"agents.exit_drl_agent: {e}")
        failures.append("imports")
        return _summarize()

    try:
        from agents.exit_drl_outcome_ledger import (
            ExitDRLOutcomeLedger, get_outcome_ledger,
            reset_singleton as _reset_ledger,
        )
        _ok("agents.exit_drl_outcome_ledger")
    except Exception as e:
        _fail(f"agents.exit_drl_outcome_ledger: {e}")
        failures.append("imports")

    try:
        from risk.exit_drl_kill_switch import (
            ExitDRLKillSwitch, get_kill_switch,
            reset_singleton as _reset_ks,
        )
        _ok("risk.exit_drl_kill_switch")
    except Exception as e:
        _fail(f"risk.exit_drl_kill_switch: {e}")
        failures.append("imports")

    try:
        from risk.exit_drl_promotion_gate import ExitDRLPromotionGate
        _ok("risk.exit_drl_promotion_gate")
    except Exception as e:
        _fail(f"risk.exit_drl_promotion_gate: {e}")
        failures.append("imports")

    # Reset singletons so this script is idempotent
    _reset_agent()
    _reset_ledger()
    _reset_ks()

    # ────────────────────────────────────────────────────────────
    # Stage 2 — checkpoint loadability
    # ────────────────────────────────────────────────────────────
    _section("STAGE 2 — checkpoint loadability per asset")
    models_dir = REPO / "models" / "exit_drl_v2"
    if not models_dir.exists():
        models_dir = REPO / "models" / "exit_drl"
        _warn(f"using v1 fallback dir {models_dir}")
    else:
        _ok(f"using v2 models dir {models_dir}")

    per_asset_modes_test = {
        "BTC": ExitDRLMode.SHADOW,    # mirror main.py production config (P200 demotion)
        "ETH": ExitDRLMode.SHADOW,
        "SOL": ExitDRLMode.SHADOW,
    }
    agent = get_exit_drl_agent(
        models_dir=str(models_dir),
        shadow_log_path=str(REPO / "data" / "exit_drl_shadow.jsonl"),
        mode=ExitDRLMode.SHADOW,
        per_asset_modes=per_asset_modes_test,
    )
    status = agent.status()
    for asset, st in status["assets"].items():
        if st == "READY":
            _ok(f"{asset}: checkpoint loaded → {st}")
        else:
            _fail(f"{asset}: checkpoint status={st}")
            failures.append(f"checkpoint:{asset}")

    # ────────────────────────────────────────────────────────────
    # Stage 3 — per-asset mode helpers
    # ────────────────────────────────────────────────────────────
    _section("STAGE 3 — per-asset mode helpers")
    for asset, expected_mode in per_asset_modes_test.items():
        actual = agent.mode_for_asset(asset)
        if actual == expected_mode:
            _ok(f"{asset}: mode_for_asset = {actual.value}")
        else:
            _fail(f"{asset}: expected {expected_mode.value}, got {actual.value}")
            failures.append(f"mode:{asset}")
        should_act = agent.should_act_on(asset)
        expected_act = (expected_mode == ExitDRLMode.EXIT_ONLY)
        if should_act == expected_act:
            _ok(f"{asset}: should_act_on={should_act}")
        else:
            _fail(f"{asset}: should_act_on={should_act}, expected {expected_act}")
            failures.append(f"act:{asset}")

    # ────────────────────────────────────────────────────────────
    # Stage 4 — predict + shadow log write
    # ────────────────────────────────────────────────────────────
    _section("STAGE 4 — predict() + shadow log")
    pos = {
        "direction": +1.0,
        "entry_price": 50000.0,
        "max_unrealized_pnl_pct": 0.005,
    }
    md = {
        "current_price": 50500.0,
        "regime": 2,
        "regime_confidence": 0.6,
    }
    for i in range(8):
        md[f"regime_proba_{i}"] = 0.125
    for asset in ("BTC", "ETH", "SOL"):
        if status["assets"].get(asset) != "READY":
            _skip(f"{asset}: checkpoint not READY")
            continue
        try:
            pred = agent.predict(asset, pos, md, bars_held=8)
            if pred is None:
                _fail(f"{asset}: predict returned None")
                failures.append(f"predict:{asset}")
                continue
            _ok(
                f"{asset}: action={pred.action.name} "
                f"conf={pred.confidence:.3f} "
                f"probs=[{', '.join(f'{p:.2f}' for p in pred.probabilities)}] "
                f"infer={pred.inference_ms:.1f}ms"
            )
        except Exception as e:
            _fail(f"{asset}: predict raised {type(e).__name__}: {e}")
            failures.append(f"predict:{asset}")

    shadow_log = REPO / "data" / "exit_drl_shadow.jsonl"
    if shadow_log.exists():
        n_lines = sum(1 for _ in shadow_log.open(encoding="utf-8"))
        _ok(f"shadow log has {n_lines} lines (this run wrote ≥3)")
    else:
        _fail(f"shadow log {shadow_log} not created")
        failures.append("shadow_log")

    # ────────────────────────────────────────────────────────────
    # Stage 5 — outcome ledger round-trip
    # ────────────────────────────────────────────────────────────
    _section("STAGE 5 — outcome ledger round-trip")
    test_ledger_path = REPO / "data" / "_diag_outcome_ledger.jsonl"
    test_ledger_path.unlink(missing_ok=True)
    ledger = ExitDRLOutcomeLedger(ledger_path=str(test_ledger_path))
    try:
        ledger.record_open("BTC", entry_price=50000.0, direction=+1)
        ledger.record_prediction("BTC", "HOLD", 0.7, 0.005, 1)
        ledger.record_prediction("BTC", "PARTIAL_EXIT", 0.6, 0.012, 4)
        ledger.record_prediction("BTC", "HOLD", 0.55, 0.008, 6)
        rec = ledger.record_close(
            "BTC", exit_price=50400.0, exit_reason="T6_DRL_ACTION",
            realized_pnl_bps=80.0,
        )
        if rec is None:
            _fail("record_close returned None")
            failures.append("ledger:close")
        elif rec.get("predicted_action_at_peak") == "PARTIAL_EXIT":
            _ok(
                f"open → 3 predictions → close: pred_at_close={rec['predicted_action_at_close']}, "
                f"pred_at_peak={rec['predicted_action_at_peak']}, "
                f"realized={rec['realized_pnl_bps']:+.1f}bps"
            )
        else:
            _fail(f"peak prediction wrong: {rec}")
            failures.append("ledger:peak")
    finally:
        test_ledger_path.unlink(missing_ok=True)

    # ────────────────────────────────────────────────────────────
    # Stage 6 — kill switch lifecycle
    # ────────────────────────────────────────────────────────────
    _section("STAGE 6 — kill switch lifecycle")
    test_ks_path = REPO / "data" / "_diag_kill_switch.jsonl"
    test_ks_path.unlink(missing_ok=True)
    ks = ExitDRLKillSwitch(state_path=str(test_ks_path))

    # No promotion → no trip
    if ks.should_demote("BTC") is None:
        _ok("BTC pre-promotion: no demote (correct — only promoted assets are eligible)")
    else:
        _fail("BTC pre-promotion: spurious demote signal")
        failures.append("ks:premature")

    # Promote, then 5 consecutive losses → trip
    ks.record_promotion("BTC")
    for _ in range(4):
        ks.record_close("BTC", -50.0, "EXIT_ALL", was_drl_driven=True)
    if ks.should_demote("BTC") is None:
        _ok("BTC after 4 losses: no demote yet (correct — threshold is 5)")
    else:
        _fail("BTC: tripped on 4 losses (off-by-one)")
        failures.append("ks:off_by_one")
    ks.record_close("BTC", -50.0, "EXIT_ALL", was_drl_driven=True)
    reason = ks.should_demote("BTC")
    if reason and "CONSECUTIVE_LOSSES" in reason:
        _ok(f"BTC after 5 losses: demote signal fired — '{reason}'")
    else:
        _fail(f"BTC after 5 losses: expected CONSECUTIVE_LOSSES trip, got {reason}")
        failures.append("ks:5_losses")

    # ETH still healthy (no promotion)
    if ks.should_demote("ETH") is None:
        _ok("ETH (not promoted): isolation works")
    else:
        _fail(f"ETH: spurious demote {ks.should_demote('ETH')}")
        failures.append("ks:isolation")
    test_ks_path.unlink(missing_ok=True)

    # ────────────────────────────────────────────────────────────
    # Stage 7 — promotion gate evaluate
    # ────────────────────────────────────────────────────────────
    _section("STAGE 7 — promotion gate state per asset")
    gate = ExitDRLPromotionGate()
    for asset in ("BTC", "ETH", "SOL"):
        ev = gate.evaluate(asset)
        block_summary = ", ".join(b.split(":")[0] for b in ev["blockers"]) or "NONE"
        line = (
            f"{asset}: would_promote={ev['would_promote']}, "
            f"blockers={block_summary}, "
            f"sharpe_lift={ev['evidence']['sharpe_lift_pct']}"
        )
        if ev["would_promote"] or "min_shadow_days" in block_summary:
            _ok(line)
        else:
            _warn(line)

    # ────────────────────────────────────────────────────────────
    # Stage 8 — promotion-state JSON has the override stamp
    # ────────────────────────────────────────────────────────────
    _section("STAGE 8 — override audit stamp")
    promo_state_path = REPO / "data" / "exit_drl_promotion_state.json"
    if not promo_state_path.exists():
        _warn(
            f"{promo_state_path} not yet created — runs at runtime startup. "
            f"Will be written on first deploy."
        )
    else:
        try:
            data = json.loads(promo_state_path.read_text(encoding="utf-8"))
            overrides = data.get("overrides", {})
            for asset in ("BTC",):
                if asset in overrides:
                    rec = overrides[asset]
                    _ok(
                        f"{asset}: force_promote_at={rec['force_promote_at']:.0f}, "
                        f"reason='{rec['reason'][:60]}...', "
                        f"blockers_at_override={len(rec['blockers_at_override'])}"
                    )
                else:
                    _warn(f"{asset}: no override stamp (runs at startup)")
        except Exception as e:
            _fail(f"override state read failed: {e}")
            failures.append("override_stamp")

    # ────────────────────────────────────────────────────────────
    # Stage 9 — bridge code in tick_exit_triggers.py
    # ────────────────────────────────────────────────────────────
    _section("STAGE 9 — System-3 PARTIAL_EXIT bridge in tick_exit_triggers.py")
    src = (REPO / "core" / "tick_exit_triggers.py").read_text(encoding="utf-8")
    checks = [
        ("[EXIT-DRL ACCELERATED" in src,            "marker comment present"),
        (".should_act_on(asset)" in src,            "should_act_on consulted"),
        (".should_demote(asset)" in src,            "kill switch consulted"),
        ("set_mode_for_asset(asset, _EM.SHADOW)" in src, "auto-demote on trip"),
        ('latest_exit_drl_action' in src,           "consumes per-tick prediction"),
        ('PARTIAL_EXIT"' in src,                    "PARTIAL_EXIT branch"),
    ]
    for ok, label in checks:
        (_ok if ok else _fail)(label)
        if not ok:
            failures.append(f"bridge:{label}")
    # Confirm bridge does NOT authorize RELEASE_RUNNER or EXIT_ALL
    bridge_block = re.search(
        r"\[EXIT-DRL ACCELERATED.*?_exit_alpha_signal = exit_alpha\.evaluate_scale_out",
        src, re.DOTALL,
    )
    if bridge_block:
        block = bridge_block.group(0)
        partial_assigns = re.findall(r"_drl_exit_output\s*=\s*DRLOutput\(action=DRLAction\.(\w+)\)", block)
        if partial_assigns == ["PARTIAL_EXIT"]:
            _ok(f"bridge limits System-3 to PARTIAL_EXIT only: {partial_assigns}")
        else:
            _fail(f"bridge over-authorized: assigns {partial_assigns}")
            failures.append("bridge:authority_creep")
    else:
        _fail("bridge block markers not found in tick_exit_triggers.py")
        failures.append("bridge:not_found")

    # ────────────────────────────────────────────────────────────
    # Stage 10 — main.py initialization
    # ────────────────────────────────────────────────────────────
    _section("STAGE 10 — main.py initialization block")
    main_src = (REPO / "main.py").read_text(encoding="utf-8-sig")
    init_checks = [
        (re.search(r'"BTC":\s*ExitDRLMode\.EXIT_ONLY', main_src),
         "BTC explicitly set to EXIT_ONLY"),
        (re.search(r'"ETH":\s*ExitDRLMode\.EXIT_ONLY', main_src),
         "ETH explicitly set to EXIT_ONLY"),
        (re.search(r'"SOL":\s*ExitDRLMode\.EXIT_ONLY', main_src),
         "SOL explicitly set to EXIT_ONLY"),
        ("_exit_drl_kill_switch" in main_src,                   "kill switch field on runner"),
        ("record_promotion(_asset)" in main_src,                "promotion timestamp recorded"),
        ("record_override(" in main_src,                        "override audit stamp recorded"),
        ("ACCELERATED_PATH" in main_src,                        "override reason explicit"),
    ]
    for hit, label in init_checks:
        (_ok if hit else _fail)(label)
        if not hit:
            failures.append(f"init:{label}")

    return _summarize()


def _summarize() -> int:
    print()
    print("=" * 72)
    if failures:
        print(f"  RESULT: {len(failures)} FAILURE(S) — {', '.join(failures)}")
        print("=" * 72)
        return 1
    print("  RESULT: ALL STAGES OK — wire-up healthy.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
