"""
Watch Step 15 / Step 16 readiness and automatically enter Live Phase 1.

This script is intentionally separate from the trading runtime. It:
1. Polls the Step 15 gate report.
2. Requires manual Step 16 checks to be marked PASS.
3. Verifies live_phase1.json.
4. Stops paper trading.
5. Launches live trading with configs/live_phase1.json.

Usage:
    python -X utf8 scripts/auto_phase1_transition.py check
    python -X utf8 scripts/auto_phase1_transition.py trigger
    python -X utf8 scripts/auto_phase1_transition.py watch --interval-min 30
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import step15_gate_report


ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
STATE_PATH = ROOT / "data" / "auto_phase1_transition_state.json"
PHASE1_CONFIG = ROOT / "configs" / "live_phase1.json"


def _write_state(payload: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


def _step16_report() -> dict:
    return step15_gate_report.build_report(hours=48)


def _pre_live_checks() -> tuple[bool, str]:
    """Run automated pre-live checks (kill switch, risk controls, reconciler)."""
    result = _run(
        [
            str(PYTHON),
            "-X",
            "utf8",
            "scripts/pre_live_checks.py",
            "--auto-mark",
        ]
    )
    detail = (result.stdout or result.stderr or "").strip()
    if len(detail) > 1200:
        detail = detail[-1200:]
    return result.returncode == 0, detail


def _live_verify() -> tuple[bool, str]:
    result = _run(
        [
            str(PYTHON),
            "-X",
            "utf8",
            "scripts/verify_live_config.py",
            "--config",
            str(PHASE1_CONFIG),
        ]
    )
    detail = (result.stdout or result.stderr or "").strip()
    if len(detail) > 1200:
        detail = detail[-1200:]
    return result.returncode == 0, detail


def _paper_running() -> tuple[bool, str]:
    result = _run([str(PYTHON), "scripts/launch_paper.py", "status"])
    output = (result.stdout or result.stderr or "").strip()
    return "[STATUS] RUNNING" in output, output


def _live_running() -> tuple[bool, str]:
    result = _run([str(PYTHON), "scripts/launch_live.py", "status"])
    output = (result.stdout or result.stderr or "").strip()
    return "[STATUS] RUNNING" in output, output


def evaluate_readiness() -> tuple[bool, dict]:
    report = _step16_report()
    ready = bool(report.get("step16_ready"))
    return ready, report


def trigger_once(*, waiting_is_error: bool = False) -> int:
    ready, report = evaluate_readiness()
    if not ready:
        _write_state(
            {
                "status": "waiting_for_step16",
                "step16_ready": False,
                "latest_session_id": report.get("latest_session_id"),
                "report": report,
            }
        )
        print("Step 16 not ready yet. Waiting for fills / gate completion.")
        print(step15_gate_report.render_text(report))
        return 2 if waiting_is_error else 0

    # Run automated pre-live checks (kill switch, risk, reconciler)
    pre_ok, pre_detail = _pre_live_checks()
    if not pre_ok:
        _write_state(
            {
                "status": "blocked_pre_live_checks",
                "step16_ready": True,
                "latest_session_id": report.get("latest_session_id"),
                "report": report,
                "pre_live_checks_detail": pre_detail,
            }
        )
        print("Step 16 ready, but pre-live checks failed.")
        print(pre_detail)
        return 6

    live_ok, live_detail = _live_verify()
    if not live_ok:
        _write_state(
            {
                "status": "blocked_live_verify",
                "step16_ready": True,
                "latest_session_id": report.get("latest_session_id"),
                "report": report,
                "live_verify_detail": live_detail,
            }
        )
        print("Step 16 ready, but live_phase1 verification failed.")
        print(live_detail)
        return 3

    live_running, live_status = _live_running()
    if live_running:
        _write_state(
            {
                "status": "live_already_running",
                "step16_ready": True,
                "latest_session_id": report.get("latest_session_id"),
                "report": report,
                "live_verify_detail": live_detail,
                "live_status": live_status,
            }
        )
        print("Live Phase 1 already running.")
        print(live_status)
        return 0

    paper_running, paper_status = _paper_running()
    if paper_running:
        stop_result = _run([str(PYTHON), "scripts/launch_paper.py", "stop"])
        if stop_result.returncode != 0:
            _write_state(
                {
                    "status": "blocked_paper_stop",
                    "step16_ready": True,
                    "latest_session_id": report.get("latest_session_id"),
                    "report": report,
                    "paper_status": paper_status,
                    "paper_stop_detail": (stop_result.stdout or stop_result.stderr or "").strip(),
                }
            )
            print("Step 16 ready, but failed to stop paper run.")
            print(stop_result.stdout or stop_result.stderr)
            return 4

    start_result = _run([str(PYTHON), "scripts/launch_live.py", "start", str(PHASE1_CONFIG)])
    if start_result.returncode != 0:
        _write_state(
            {
                "status": "blocked_live_start",
                "step16_ready": True,
                "latest_session_id": report.get("latest_session_id"),
                "report": report,
                "live_verify_detail": live_detail,
                "live_start_detail": (start_result.stdout or start_result.stderr or "").strip(),
            }
        )
        print("Step 16 ready, but failed to start Live Phase 1.")
        print(start_result.stdout or start_result.stderr)
        return 5

    _write_state(
        {
            "status": "phase1_started",
            "step16_ready": True,
            "latest_session_id": report.get("latest_session_id"),
            "report": report,
            "live_verify_detail": live_detail,
            "phase1_config": str(PHASE1_CONFIG),
            "launch_output": (start_result.stdout or "").strip(),
        }
    )
    print("Step 16 reached. Live Phase 1 launched.")
    print(start_result.stdout)
    return 0


def cmd_check() -> int:
    ready, report = evaluate_readiness()
    print(step15_gate_report.render_text(report))
    pre_ok, pre_detail = _pre_live_checks()
    print("")
    print(f"Pre-live checks: {'PASS' if pre_ok else 'FAIL'}")
    if pre_detail:
        print(pre_detail)
    live_ok, live_detail = _live_verify()
    print("")
    print(f"Live phase1 verification: {'PASS' if live_ok else 'FAIL'}")
    if live_detail:
        print(live_detail)
    return 0


def cmd_watch(interval_min: int) -> int:
    print(f"Watching Step 16 readiness every {interval_min} minutes...")
    try:
        while True:
            rc = trigger_once(waiting_is_error=True)
            if rc == 0:
                return 0
            time.sleep(max(interval_min, 1) * 60)
    except KeyboardInterrupt:
        print("Watch stopped by user.")
        return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Auto-transition from Step 15 to Live Phase 1.")
    parser.add_argument("command", choices=["check", "trigger", "watch"], help="Action to run.")
    parser.add_argument("--interval-min", type=int, default=30, help="Watch poll interval in minutes.")
    args = parser.parse_args(argv)

    if args.command == "check":
        return cmd_check()
    if args.command == "trigger":
        return trigger_once(waiting_is_error=False)
    return cmd_watch(args.interval_min)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
