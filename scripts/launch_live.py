"""
Live trading launcher with process management.

Usage:
    python scripts/launch_live.py start [highrisk|phase1|phase2|CONFIG]
    python scripts/launch_live.py stop
    python scripts/launch_live.py status
    python scripts/launch_live.py restart [highrisk|phase1|phase2|CONFIG]
    python scripts/launch_live.py tail
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
IS_WIN = sys.platform == "win32"
PYTHON = ROOT / ("venv/Scripts/python.exe" if IS_WIN else "venv/bin/python")
PID_FILE = ROOT / "data" / "live_run.pid"
STDOUT_LOG = ROOT / "logs" / "live_run_stdout.log"
STDERR_LOG = ROOT / "logs" / "live_run_stderr.log"
DEFAULT_CONFIG = ROOT / "configs" / "live_high_risk.json"
CONFIG_ALIASES = {
    "highrisk": ROOT / "configs" / "live_high_risk.json",
    "phase1": ROOT / "configs" / "live_phase1.json",
    "phase2": ROOT / "configs" / "live_phase2.json",
}


def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        data = json.loads(PID_FILE.read_text(encoding="utf-8"))
        return int(data.get("pid"))
    except Exception:
        return None


def _is_alive(pid: int) -> bool:
    if not IS_WIN:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False
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
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
            encoding="utf-8")
            return str(pid) in result.stdout
        except Exception:
            return False


def _resolve_config(config: str | None) -> Path:
    if not config:
        return DEFAULT_CONFIG
    resolved = CONFIG_ALIASES.get(config.lower())
    if resolved:
        return resolved
    path = Path(config)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _write_pid(pid: int, config_path: Path) -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    PID_FILE.write_text(
        json.dumps(
            {
                "pid": pid,
                "started_at": started_at,
                "command": f"main.py --mode live --config {config_path} --confirm-live",
                "config": str(config_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _extract_latest_run_activity(log_path: Path, labels: list[str]) -> list[str]:
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        banner_idx = 0
        for idx, line in enumerate(lines):
            if "Live run started at" in line:
                banner_idx = idx
        scoped = lines[banner_idx:]
        return [line.strip() for line in scoped if any(label in line for label in labels)][-8:]
    except Exception:
        return []


def _show_last_activity() -> None:
    stderr_recent = _extract_latest_run_activity(
        STDERR_LOG, ["[LIVE]", "Round", "TRADE", "PNL", "ERROR", "Traceback", "[CONFIG]"]
    )
    stdout_recent = _extract_latest_run_activity(
        STDOUT_LOG, ["[LIVE]", "Round", "TRADE", "PNL", "ERROR", "[CONFIG]", "[DRL_GATE]"]
    )
    if stderr_recent:
        print("\n  Last activity (from stderr):")
        for line in stderr_recent:
            print(f"    {line}")
    elif stdout_recent:
        print("\n  Last activity (from stdout):")
        for line in stdout_recent:
            print(f"    {line}")
    elif STDERR_LOG.exists() or STDOUT_LOG.exists():
        print("\n  Last activity:")
        print("    No runtime lines captured after the latest start banner yet")


def cmd_start(config: str | None = None) -> None:
    config_path = _resolve_config(config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    existing_pid = _read_pid()
    if existing_pid and _is_alive(existing_pid):
        print(f"[ERROR] Live run already running (PID {existing_pid})")
        sys.exit(1)

    if PID_FILE.exists():
        PID_FILE.unlink()

    STDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    stdout_f = open(STDOUT_LOG, "a", encoding="utf-8")
    stderr_f = open(STDERR_LOG, "a", encoding="utf-8")
    banner_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    banner = f"\n{'='*72}\n  Live run started at {banner_ts}\n{'='*72}\n"
    stdout_f.write(banner)
    stderr_f.write(banner)
    stdout_f.flush()
    stderr_f.flush()

    cmd = [
        str(PYTHON),
        "-X",
        "utf8",
        "-u",
        "main.py",
        "--mode",
        "live",
        "--config",
        str(config_path),
        "--confirm-live",
    ]
    _popen_kwargs = {
        "stdout": stdout_f,
        "stderr": stderr_f,
        "cwd": str(ROOT),
    }
    if IS_WIN:
        _popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        _popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **_popen_kwargs)
    _write_pid(proc.pid, config_path)
    print(f"Live run launched (PID {proc.pid})")
    print(f"  config  : {config_path}")
    print(f"  PID file: {PID_FILE}")
    print(f"  stdout  : {STDOUT_LOG}")
    print(f"  stderr  : {STDERR_LOG}")


def cmd_stop() -> None:
    pid = _read_pid()
    if not pid:
        print("[INFO] No PID file found - live run not managed by this launcher")
        return
    if not _is_alive(pid):
        print(f"[INFO] Process {pid} not running (stale PID file). Cleaning up.")
        PID_FILE.unlink(missing_ok=True)
        return

    if IS_WIN:
        print(f"[STOP] Sending CTRL_BREAK_EVENT to PID {pid}...")
        try:
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        except OSError:
            subprocess.run(["taskkill", "/PID", str(pid)], capture_output=True)
    else:
        print(f"[STOP] Sending SIGTERM to PID {pid}...")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            PID_FILE.unlink(missing_ok=True)
            return
    print("[STOP] Waiting for graceful shutdown (up to 15s)...")
    for i in range(15):
        time.sleep(1)
        if not _is_alive(pid):
            print(f"[STOP] Process exited after {i+1}s")
            PID_FILE.unlink(missing_ok=True)
            return
    print("[STOP] Process still alive after 15s. Force killing...")
    try:
        if IS_WIN:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
    PID_FILE.unlink(missing_ok=True)
    print("[STOP] Done")


def cmd_status() -> None:
    pid = _read_pid()
    if not pid:
        print("[STATUS] No PID file - live run not running")
        _show_last_activity()
        return

    alive = _is_alive(pid)
    if alive:
        try:
            data = json.loads(PID_FILE.read_text(encoding="utf-8"))
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


def cmd_restart(config: str | None = None) -> None:
    cmd_stop()
    time.sleep(2)
    cmd_start(config=config)


def cmd_tail() -> None:
    if not STDERR_LOG.exists():
        print(f"[TAIL] Log not found: {STDERR_LOG}")
        return
    print(f"[TAIL] Following {STDERR_LOG} (Ctrl+C to stop)\n")
    try:
        if IS_WIN:
            subprocess.run(
                ["powershell", "-Command", f"Get-Content '{STDERR_LOG}' -Tail 30 -Wait -Encoding UTF8"],
                cwd=str(ROOT),
            )
        else:
            subprocess.run(["tail", "-n", "30", "-f", str(STDERR_LOG)], cwd=str(ROOT))
    except KeyboardInterrupt:
        print("\n[TAIL] Stopped")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/launch_live.py <start|stop|status|restart|tail> [highrisk|phase1|phase2|CONFIG]")
        sys.exit(1)
    cmd = sys.argv[1].lower()
    os.chdir(str(ROOT))
    if cmd == "start":
        cmd_start(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "stop":
        cmd_stop()
    elif cmd == "status":
        cmd_status()
    elif cmd == "restart":
        cmd_restart(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "tail":
        cmd_tail()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
