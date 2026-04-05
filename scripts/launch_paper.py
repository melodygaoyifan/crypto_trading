"""
Paper trading launcher with process management.

Usage:
    python scripts/launch_paper.py start   -- launch paper run (detached)
    python scripts/launch_paper.py stop    -- graceful stop via CTRL_BREAK_EVENT
    python scripts/launch_paper.py status  -- check if running + last log lines
    python scripts/launch_paper.py restart -- stop then start
    python scripts/launch_paper.py smoke   -- foreground 1-round health check
    python scripts/launch_paper.py tail    -- tail stderr log (live output)

PID file: data/paper_run.pid
Logs:     logs/paper_run_stdout.log, logs/paper_run_stderr.log
"""
import subprocess
import sys
import os
import signal
import time
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(r"c:\Users\melod\Downloads\hmats")
PID_FILE = ROOT / "data" / "paper_run.pid"
MANUAL_RESTART_MARKER = ROOT / "data" / "manual_restart_marker.json"
STDOUT_LOG = ROOT / "logs" / "paper_run_stdout.log"
STDERR_LOG = ROOT / "logs" / "paper_run_stderr.log"
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
DEFAULT_PROFILE = "HIGH_RISK"
SMOKE_TIMEOUT_SECONDS = 180
PROFILE_ALIASES = {
    "highrisk": "HIGH_RISK",
    "default": "HIGH_RISK",
    "profit": "PAPER_PROFIT_5Y",
    "baseline": "PAPER_BASELINE_5Y",
}


def _read_pid() -> int | None:
    """Read PID from file. Returns None if missing or stale."""
    if not PID_FILE.exists():
        return None
    try:
        data = json.loads(PID_FILE.read_text())
        return data.get("pid")
    except (json.JSONDecodeError, KeyError):
        return None


def _is_alive(pid: int) -> bool:
    """Check if a process with given PID is alive (Windows)."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:
        # Fallback: try tasklist
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            return str(pid) in result.stdout
        except Exception:
            return False


def _resolve_profile(profile: str | None) -> str:
    """Resolve user-friendly profile aliases to actual risk-profile names."""
    if not profile:
        return DEFAULT_PROFILE
    prof = PROFILE_ALIASES.get(profile.lower(), profile)
    return str(prof).upper()


def _write_pid(pid: int, profile: str):
    """Write PID file with metadata."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    PID_FILE.write_text(json.dumps({
        "pid": pid,
        "started_at": started_at,
        "command": f"main.py --mode paper --risk-profile {profile}",
    }, indent=2))


def _mark_manual_restart(action: str):
    """Mark restart intent so runtime can avoid false AC-4 crash-loop counting."""
    try:
        MANUAL_RESTART_MARKER.parent.mkdir(parents=True, exist_ok=True)
        MANUAL_RESTART_MARKER.write_text(
            json.dumps(
                {
                    "ts": time.time(),
                    "action": action,
                },
                indent=2,
            )
        )
    except Exception:
        pass


def _stop_process(proc: subprocess.Popen, label: str = "PROC"):
    """Stop a foreground child process without leaving a stray paper runner."""
    if proc.poll() is not None:
        return

    try:
        os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
        try:
            proc.wait(timeout=8)
            return
        except subprocess.TimeoutExpired:
            pass
    except Exception:
        pass

    try:
        proc.terminate()
        proc.wait(timeout=5)
        return
    except Exception:
        pass

    try:
        subprocess.run(["taskkill", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=5)
    except Exception:
        pass

    try:
        proc.wait(timeout=5)
    except Exception:
        print(f"[{label}] Failed to confirm child shutdown for PID {proc.pid}")


def _safe_print(text: str = ""):
    """Print without crashing on Windows console encoding mismatches."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe)


def _mark_smoke_clean_shutdown():
    """Prevent smoke checks from polluting AC-4 crash-loop session state."""
    session_file = ROOT / "data" / "session_state.json"
    if not session_file.exists():
        return
    try:
        data = json.loads(session_file.read_text(encoding="utf-8"))
        if data.get("manual_restart_consumed"):
            data["clean_shutdown"] = True
            data["last_shutdown"] = time.time()
            data["recent_starts"] = []
            session_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def cmd_start(profile: str = DEFAULT_PROFILE):
    """Launch paper run as detached process."""
    profile = _resolve_profile(profile)
    # Check for existing process
    existing_pid = _read_pid()
    if existing_pid and _is_alive(existing_pid):
        print(f"[ERROR] Paper run already running (PID {existing_pid})")
        print(f"        Use 'stop' first, or 'status' to check")
        sys.exit(1)

    # Clean stale PID file
    if PID_FILE.exists():
        PID_FILE.unlink()

    # Ensure log directory exists
    STDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)

    # [AUDIT-12] Truncate stderr log to last 1000 lines on restart
    # Main structured logs now go through RotatingFileHandler (logs/hmats.log).
    # This file only catches unhandled stderr (tracebacks, warnings).
    for _log_path in (STDOUT_LOG, STDERR_LOG):
        if _log_path.exists():
            try:
                _lines = _log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if len(_lines) > 1000:
                    _log_path.write_text(
                        "\n".join(_lines[-1000:]) + "\n", encoding="utf-8"
                    )
            except Exception:
                pass

    # Open logs in append mode
    stdout_f = open(STDOUT_LOG, "a", encoding="utf-8")
    stderr_f = open(STDERR_LOG, "a", encoding="utf-8")

    # Write separator for this run
    banner_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    banner = f"\n{'='*72}\n  Paper run started at {banner_ts}\n{'='*72}\n"
    stdout_f.write(banner)
    stderr_f.write(banner)
    stdout_f.flush()
    stderr_f.flush()

    # Default to the explicit HIGH_RISK baseline; sample/profit overlays remain available.
    cmd = [str(PYTHON), "-X", "utf8", "-u", "main.py", "--mode", "paper",
           "--risk-profile", profile]
    proc = subprocess.Popen(
        cmd,
        stdout=stdout_f,
        stderr=stderr_f,
        cwd=str(ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    )

    _write_pid(proc.pid, profile)

    print(f"Paper run launched (PID {proc.pid})")
    print(f"  profile : {profile}")
    print(f"  PID file: {PID_FILE}")
    print(f"  stdout  : {STDOUT_LOG}")
    print(f"  stderr  : {STDERR_LOG}")
    print(f"  Stop    : python scripts/launch_paper.py stop")
    print(f"  Tail    : python scripts/launch_paper.py tail")


def cmd_stop():
    """Graceful stop: send CTRL_BREAK_EVENT (triggers signal handler in main.py)."""
    pid = _read_pid()
    _mark_manual_restart("stop")
    if not pid:
        print("[INFO] No PID file found - paper run not managed by this launcher")
        return

    if not _is_alive(pid):
        print(f"[INFO] Process {pid} not running (stale PID file). Cleaning up.")
        PID_FILE.unlink(missing_ok=True)
        return

    print(f"[STOP] Sending CTRL_BREAK_EVENT to PID {pid}...")
    try:
        # CTRL_BREAK_EVENT is the clean way to stop a process group on Windows
        # main.py has signal.signal(signal.SIGINT, signal_handler) which calls runner.stop()
        os.kill(pid, signal.CTRL_BREAK_EVENT)
    except OSError as e:
        print(f"[WARN] os.kill failed: {e}. Trying taskkill...")
        subprocess.run(["taskkill", "/PID", str(pid)], capture_output=True)

    # Wait up to 15s for graceful shutdown
    print("[STOP] Waiting for graceful shutdown (up to 15s)...")
    for i in range(15):
        time.sleep(1)
        if not _is_alive(pid):
            print(f"[STOP] Process exited after {i+1}s")
            PID_FILE.unlink(missing_ok=True)
            return

    # Force kill if still alive
    print(f"[STOP] Process still alive after 15s. Force killing...")
    try:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    except Exception:
        pass
    PID_FILE.unlink(missing_ok=True)
    print("[STOP] Done")


def cmd_status():
    """Show paper run status."""
    pid = _read_pid()
    if not pid:
        print("[STATUS] No PID file - paper run not running")
        _show_last_activity()
        return

    alive = _is_alive(pid)
    if alive:
        # Read PID file metadata
        try:
            data = json.loads(PID_FILE.read_text())
            started = data.get("started_at", "unknown")
            command = data.get("command", "")
        except Exception:
            started = "unknown"
            command = ""
        print(f"[STATUS] RUNNING (PID {pid}, started {started})")
        if command:
            print(f"         {command}")
    else:
        print(f"[STATUS] DEAD (PID {pid} not found, stale PID file)")
        PID_FILE.unlink(missing_ok=True)

    _show_last_activity()

    # Show paper positions if they exist
    pos_file = ROOT / "data" / "paper_positions.json"
    if pos_file.exists():
        try:
            raw = json.loads(pos_file.read_text())
            if isinstance(raw, dict) and isinstance(raw.get("positions"), dict):
                positions = raw.get("positions", {})
            elif isinstance(raw, dict):
                positions = raw
            else:
                positions = {}

            open_positions = {}
            for asset, pos in positions.items():
                if not isinstance(pos, dict):
                    continue
                exposure = float(pos.get("exposure", 0) or 0)
                if abs(exposure) > 1e-6:
                    open_positions[asset] = pos

            if open_positions:
                print(f"\n  Open positions ({len(open_positions)}):")
                for asset, pos in open_positions.items():
                    direction = "LONG" if pos.get("direction", 0) > 0 else "SHORT"
                    exp = float(pos.get("exposure", 0) or 0) * 100
                    entry = float(pos.get("entry_price", 0) or 0)
                    print(f"    {asset}: {direction} {exp:.2f}% @ ${entry:,.2f}")
            else:
                print("\n  No open positions")
        except Exception:
            pass


def _extract_latest_run_activity(log_path: Path, labels: list[str]) -> list[str]:
    """Return recent activity scoped to the latest banner section only."""
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        banner_idx = 0
        for idx, line in enumerate(lines):
            if "Paper run started at" in line:
                banner_idx = idx
        scoped = lines[banner_idx:]
        return [line.strip() for line in scoped if any(label in line for label in labels)][-8:]
    except Exception:
        return []


def _show_last_activity():
    """Show last few lines from the latest run section."""
    stderr_recent = _extract_latest_run_activity(
        STDERR_LOG, ["[PAPER]", "Round", "TRADE", "PNL", "FLATTEN", "ERROR", "Traceback"]
    )
    stdout_recent = _extract_latest_run_activity(
        STDOUT_LOG, ["[PAPER]", "Round", "TRADE", "PNL", "FLATTEN", "ERROR", "[CONFIG]", "[DRL_GATE]"]
    )

    if stderr_recent:
        print(f"\n  Last activity (from stderr):")
        for line in stderr_recent:
            print(f"    {line}")
    elif stdout_recent:
        print(f"\n  Last activity (from stdout):")
        for line in stdout_recent:
            print(f"    {line}")
    elif STDERR_LOG.exists() or STDOUT_LOG.exists():
        print("\n  Last activity:")
        print("    No runtime lines captured after the latest start banner yet")


def cmd_restart(profile: str = DEFAULT_PROFILE):
    """Stop then start."""
    _mark_manual_restart("restart")
    cmd_stop()
    time.sleep(2)
    cmd_start(profile=profile)


def cmd_tail():
    """Tail the stderr log (blocks until Ctrl+C)."""
    if not STDERR_LOG.exists():
        print(f"[TAIL] Log not found: {STDERR_LOG}")
        return

    print(f"[TAIL] Following {STDERR_LOG} (Ctrl+C to stop)\n")
    try:
        # Use PowerShell Get-Content -Wait for live tailing on Windows
        subprocess.run(
            ["powershell", "-Command", f"Get-Content '{STDERR_LOG}' -Tail 30 -Wait -Encoding UTF8"],
            cwd=str(ROOT),
        )
    except KeyboardInterrupt:
        print("\n[TAIL] Stopped")


def cmd_smoke(profile: str = DEFAULT_PROFILE):
    """Run a foreground smoke check and pass only after Round 1 completes."""
    profile = _resolve_profile(profile)
    _mark_manual_restart("smoke")
    cmd = [
        str(PYTHON),
        "-X",
        "utf8",
        "-u",
        "main.py",
        "--mode",
        "paper",
        "--risk-profile",
        profile,
    ]
    print(f"[SMOKE] Starting foreground paper health check")
    print(f"[SMOKE] Profile: {profile}")
    print(f"[SMOKE] Timeout: {SMOKE_TIMEOUT_SECONDS}s")
    print(f"[SMOKE] Command: {' '.join(cmd[1:])}")
    print("")

    success = False
    runtime_error = None
    deadline = time.time() + SMOKE_TIMEOUT_SECONDS
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )

    try:
        while True:
            if time.time() > deadline:
                runtime_error = f"timed out after {SMOKE_TIMEOUT_SECONDS}s before Round 1 completed"
                break

            line = proc.stdout.readline() if proc.stdout is not None else ""
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                continue

            line = line.rstrip("\r\n")
            _safe_print(line)

            if "Runtime error:" in line or "Traceback" in line:
                runtime_error = line
                break
            if "[PAPER] Round 1 done." in line or "[PAPER] ==== Round 2" in line:
                success = True
                break
    finally:
        if proc.poll() is None:
            _stop_process(proc, label="SMOKE")

    if success:
        _mark_smoke_clean_shutdown()
        print("")
        print("[SMOKE] PASS: Round 1 completed without a runtime crash")
        return

    if proc.returncode not in (None, 0) and runtime_error is None:
        runtime_error = f"child exited with code {proc.returncode}"
    elif runtime_error is None:
        runtime_error = "paper process exited before Round 1 completion"

    print("")
    print(f"[SMOKE] FAIL: {runtime_error}")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/launch_paper.py <start|stop|status|restart|smoke|tail> [profit|baseline|PROFILE]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "restart": cmd_restart,
        "smoke": cmd_smoke,
        "tail": cmd_tail,
    }

    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(commands)}")
        sys.exit(1)

    os.chdir(str(ROOT))
    if cmd in ("start", "restart", "smoke"):
        profile = _resolve_profile(sys.argv[2] if len(sys.argv) > 2 else None)
        commands[cmd](profile=profile)
    else:
        commands[cmd]()


if __name__ == "__main__":
    main()
