"""
HMATS Live Watchdog — monitors live process health and auto-restarts on failure.

Checks:
  1. Process alive (PID check)
  2. Log freshness (no new logs in N minutes = stuck)
  3. Data flow (no LIVE_DATA lines in N minutes = API dead)

Usage:
  python scripts/live_watchdog.py              # run once (cron-friendly)
  python scripts/live_watchdog.py --loop 300   # loop every 300s (5min)
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
PID_FILE = ROOT / "data" / "detached_pid.json"
STDERR_LOG = ROOT / "logs" / "live_stderr.log"
STDOUT_LOG = ROOT / "logs" / "live_stdout.log"
WATCHDOG_LOG = ROOT / "logs" / "watchdog.log"
IS_WIN = sys.platform == "win32"
PYTHON = ROOT / ("venv/Scripts/python.exe" if IS_WIN else "venv/bin/python")

# Thresholds
MAX_LOG_STALE_SECONDS = 300       # 5 min no log activity = stuck
MAX_DATA_STALE_SECONDS = 600      # 10 min no LIVE_DATA = API dead
MAX_RESTARTS_PER_HOUR = 3         # avoid restart loops

_restart_times = []


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(WATCHDOG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def is_alive(pid: int) -> bool:
    if not pid:
        return False
    if not IS_WIN:
        # Linux: signal 0 checks existence without affecting the process.
        # PermissionError = process exists but owned by another user (still "alive").
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
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:
        # Fallback: tasklist
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True, timeout=10
            , encoding="utf-8")
            return str(pid) in result.stdout
        except Exception:
            return False


def get_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        data = json.loads(PID_FILE.read_text(encoding="utf-8"))
        return data.get("pid")
    except Exception:
        return None


def get_log_age() -> float | None:
    """Seconds since last log write."""
    if not STDERR_LOG.exists():
        return None
    try:
        mtime = STDERR_LOG.stat().st_mtime
        return time.time() - mtime
    except Exception:
        return None


def get_last_data_age() -> float | None:
    """Seconds since last LIVE_DATA log line."""
    if not STDERR_LOG.exists():
        return None
    try:
        # Read last 50KB of log
        with open(STDERR_LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 50000))
            tail = f.read().decode("utf-8", errors="replace")

        # Find last LIVE_DATA timestamp
        last_ts = None
        for line in tail.split("\n"):
            if "[LIVE_DATA]" in line and line[0:4].isdigit():
                try:
                    ts_str = line[:19]  # "2026-04-10 18:47:25"
                    last_ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue

        if last_ts is None:
            return None
        # Log timestamps are LOCAL time (not UTC)
        now = datetime.now()
        return (now - last_ts).total_seconds()
    except Exception:
        return None


def restart_process():
    """Kill and restart the live trading process."""
    global _restart_times
    now = time.time()
    _restart_times = [t for t in _restart_times if now - t < 3600]
    if len(_restart_times) >= MAX_RESTARTS_PER_HOUR:
        log(f"ABORT: {len(_restart_times)} restarts in last hour, exceeds limit {MAX_RESTARTS_PER_HOUR}. Manual intervention needed.")
        return False

    # Kill ALL existing main.py live processes (not just PID file)
    # This prevents orphan accumulation (Bug #4)
    try:
        import subprocess as _sp
        if IS_WIN:
            _sp.run(
                ["powershell.exe", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                 "Where-Object { $_.CommandLine -match 'main.py.*live' } | "
                 "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
                capture_output=True, text=True, timeout=15
            , encoding="utf-8")
        else:
            # Linux: pkill matches against full command line with -f
            _sp.run(["pkill", "-9", "-f", "main.py.*live"],
                    capture_output=True, text=True, timeout=15, encoding="utf-8")
        log(f"Killed all live processes")
        time.sleep(3)
    except Exception as e:
        # Fallback: kill by PID file
        pid = get_pid()
        if pid and is_alive(pid):
            log(f"Killing PID {pid}...")
            try:
                os.kill(pid, 9)
                time.sleep(2)
            except Exception as e2:
                log(f"Kill failed: {e2}")

    # Find config from PID file
    config = "configs/live_high_risk.json"
    try:
        data = json.loads(PID_FILE.read_text(encoding="utf-8"))
        config = data.get("config", config)
    except Exception:
        pass

    # Start new process
    log(f"Starting new live process with config={config}...")
    try:
        stdout_f = open(STDOUT_LOG, "w", encoding="utf-8")
        stderr_f = open(STDERR_LOG, "w", encoding="utf-8")
        _popen_kwargs = {
            "stdout": stdout_f,
            "stderr": stderr_f,
            "cwd": str(ROOT),
        }
        if IS_WIN:
            _popen_kwargs["creationflags"] = (
                subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
            )
        else:
            # Linux: start_new_session detaches from controlling terminal
            _popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            [str(PYTHON), "-X", "utf8", "-u", "main.py",
             "--mode", "live", "--config", config, "--confirm-live"],
            **_popen_kwargs,
        )
        new_pid = proc.pid
        PID_FILE.write_text(json.dumps({
            "pid": new_pid,
            "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "command": f"main.py --mode live --config {config} --confirm-live",
            "config": config,
            "restart_reason": "watchdog auto-restart",
        }, indent=2), encoding="utf-8")
        _restart_times.append(now)
        log(f"Started PID {new_pid} successfully")
        return True
    except Exception as e:
        log(f"Start FAILED: {e}")
        return False


DRL_STATE_FILE = ROOT / "data" / "drl_promotion_state.json"
_last_known_drl_level = None


def check_orphan_processes() -> tuple[str, str]:
    """W1: Check for orphan main.py live processes."""
    try:
        if IS_WIN:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                 "Where-Object { $_.CommandLine -match 'main.py.*live' }).Count"],
                capture_output=True, text=True, timeout=10
            , encoding="utf-8")
            count = int(result.stdout.strip() or "0")
            threshold = 2  # 2 = parent+child (normal Windows behavior)
        else:
            # Linux: pgrep -c counts matching processes (single process is normal)
            result = subprocess.run(
                ["pgrep", "-cf", "main.py.*live"],
                capture_output=True, text=True, timeout=10
            , encoding="utf-8")
            count = int(result.stdout.strip() or "0")
            threshold = 1
        if count <= threshold:
            return "PASS", f"{count} process(es)"
        return "WARN", f"{count} processes — {count - threshold} potential orphans"
    except Exception as e:
        return "SKIP", f"check failed: {e}"


def check_drl_state_file() -> tuple[str, str]:
    """W2: Check DRL state file hasn't been unexpectedly overwritten."""
    global _last_known_drl_level
    try:
        if not DRL_STATE_FILE.exists():
            return "SKIP", "no state file"
        with open(DRL_STATE_FILE, encoding="utf-8") as f:
            state = json.load(f)
        current = state.get("authority_level", "UNKNOWN")
        if _last_known_drl_level is None:
            _last_known_drl_level = current
            return "PASS", f"initial={current}"
        if current == _last_known_drl_level:
            return "PASS", f"level={current} (unchanged)"
        old = _last_known_drl_level
        _last_known_drl_level = current
        return "WARN", f"CHANGED {old}→{current} — may have been overwritten"
    except Exception as e:
        return "SKIP", f"check failed: {e}"


def check_log_growing() -> tuple[str, str]:
    """W3: Check log file is growing (not frozen from stuck process)."""
    try:
        if not STDERR_LOG.exists():
            return "SKIP", "no log file"
        size = STDERR_LOG.stat().st_size
        mtime = STDERR_LOG.stat().st_mtime
        age = time.time() - mtime
        if age > MAX_LOG_STALE_SECONDS:
            return "WARN", f"log frozen {age:.0f}s (size={size:,})"
        return "PASS", f"age={age:.0f}s, size={size:,}"
    except Exception as e:
        return "SKIP", f"check failed: {e}"


def check_data_rate() -> tuple[str, str]:
    """W4: Check LIVE_DATA log rate (should be ~3 assets per ~34s cycle)."""
    try:
        if not STDERR_LOG.exists():
            return "SKIP", "no log file"
        with open(STDERR_LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 20000))
            tail = f.read().decode("utf-8", errors="replace")

        count = tail.count("[LIVE_DATA]")
        lines = tail.count("\n")
        # In ~20KB of logs, expect at least a few LIVE_DATA entries
        if count == 0:
            return "WARN", f"zero LIVE_DATA in last {lines} lines"
        return "PASS", f"{count} LIVE_DATA entries in last {lines} lines"
    except Exception as e:
        return "SKIP", f"check failed: {e}"


def check_health() -> tuple[bool, str]:
    """Returns (healthy, reason). Runs W1-W4 + existing checks."""
    pid = get_pid()
    if not pid:
        return False, "no PID file"
    if not is_alive(pid):
        return False, f"PID {pid} not alive"

    log_age = get_log_age()
    if log_age is not None and log_age > MAX_LOG_STALE_SECONDS:
        return False, f"log stale ({log_age:.0f}s > {MAX_LOG_STALE_SECONDS}s)"

    data_age = get_last_data_age()
    if data_age is not None and data_age > MAX_DATA_STALE_SECONDS:
        return False, f"no LIVE_DATA for {data_age:.0f}s (> {MAX_DATA_STALE_SECONDS}s)"

    return True, f"PID={pid}, log_age={log_age:.0f}s, data_age={data_age:.0f}s" if log_age and data_age else f"PID={pid} alive"


def run_once():
    healthy, reason = check_health()

    # Layer 3 checks (W1-W4) — always run, even when healthy
    w1_status, w1_detail = check_orphan_processes()
    w2_status, w2_detail = check_drl_state_file()
    w3_status, w3_detail = check_log_growing()
    w4_status, w4_detail = check_data_rate()

    # Log Layer 3 warnings
    for wid, status, detail in [("W1", w1_status, w1_detail), ("W2", w2_status, w2_detail),
                                  ("W3", w3_status, w3_detail), ("W4", w4_status, w4_detail)]:
        if status == "WARN":
            log(f"[HEALTH_{wid}] WARN: {detail}")

    if healthy:
        log(f"HEALTHY: {reason}")
    else:
        log(f"UNHEALTHY: {reason} — restarting...")
        restart_process()


def main():
    os.chdir(str(ROOT))
    if "--loop" in sys.argv:
        idx = sys.argv.index("--loop")
        interval = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 300
        log(f"Watchdog loop started (interval={interval}s)")
        while True:
            try:
                run_once()
            except Exception as e:
                log(f"Watchdog error: {e}")
            time.sleep(interval)
    else:
        run_once()


if __name__ == "__main__":
    main()
