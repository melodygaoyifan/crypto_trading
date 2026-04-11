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
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"

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
            )
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

    # Kill existing
    pid = get_pid()
    if pid and is_alive(pid):
        log(f"Killing PID {pid}...")
        try:
            os.kill(pid, 9)
            time.sleep(2)
        except Exception as e:
            log(f"Kill failed: {e}")

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
        proc = subprocess.Popen(
            [str(PYTHON), "-X", "utf8", "-u", "main.py",
             "--mode", "live", "--config", config, "--confirm-live"],
            stdout=stdout_f,
            stderr=stderr_f,
            cwd=str(ROOT),
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
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


def check_health() -> tuple[bool, str]:
    """Returns (healthy, reason)."""
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
