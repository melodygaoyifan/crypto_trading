"""
Step 15 / Step 16 gate report from local paper-run artifacts.

Usage:
    python -X utf8 scripts/step15_gate_report.py
    python -X utf8 scripts/step15_gate_report.py --json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeVar

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None


ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
DATA_DIR = ROOT / "data"
MANUAL_GATES_DIR = DATA_DIR / "manual_gates"
STEP16_MANUAL_CHECKS = MANUAL_GATES_DIR / "step16_manual_checks.json"
STEP15_STATUS = DATA_DIR / "step15_status.json"
DASHBOARD_STATE = DATA_DIR / "dashboard_state.json"
LOCAL_TZ = ZoneInfo("America/Chicago") if ZoneInfo else timezone.utc
UTC = timezone.utc
STACK_LEN = 8
OBS_DIM = 126
WAVELET_FEATURES = [
    "rsi_14_denoised",
    "macd_12_26_denoised",
    "bb_width_20_denoised",
    "atr_14_denoised",
    "vol_ratio_s_denoised",
]

LOG_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
START_BANNER_RE = re.compile(r"Paper run started at (?P<ts>[\d\-:T\.]+Z)")
GMM_RE = re.compile(r"\[GMM\]\s+(?P<asset>BTC|ETH|SOL):")
OOD_LOAD_RE = re.compile(r"OOD detector loaded:\s+(?P<asset>BTC|ETH|SOL)")
LSTM_RE = re.compile(r"models/retrained/(?P<asset>BTC|ETH|SOL)/.+/LSTM_FILM_A")
PROFILE_RE = re.compile(r"profile=(?P<profile>[A-Z0-9_]+)")
CRASH_MARKERS = (
    "Runtime error:",
    "Traceback",
    "NameError:",
    "UnboundLocalError:",
)


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


T = TypeVar("T")


def parse_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        if text.endswith("Z") and re.search(r"[+-]\d{2}:\d{2}Z$", text):
            text = text[:-1]
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=LOCAL_TZ).astimezone(UTC)
        return dt.astimezone(UTC)
    except Exception:
        pass
    match = LOG_TS_RE.match(text)
    if match:
        dt = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=LOCAL_TZ).astimezone(UTC)
    return None


def fmt_dt(dt: Optional[datetime]) -> str:
    return dt.astimezone(UTC).isoformat() if dt else "n/a"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def safe_ratio(num: float, den: float) -> float:
    return 0.0 if den <= 0 else num / den


def load_text_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []


def iter_log_events() -> list[tuple[datetime, str, str]]:
    events: list[tuple[datetime, str, str]] = []
    log_paths = sorted(LOG_DIR.glob("hmats.log*")) + [
        LOG_DIR / "paper_run_stderr.log",
        LOG_DIR / "paper_run_stdout.log",
    ]
    seen: set[tuple[str, str]] = set()
    for path in log_paths:
        if not path.exists():
            continue
        for line in load_text_lines(path):
            dt = parse_dt(line)
            if not dt:
                continue
            key = (dt.isoformat(), line)
            if key in seen:
                continue
            seen.add(key)
            events.append((dt, str(path.relative_to(ROOT)), line))
    events.sort(key=lambda item: item[0])
    return events


def latest_start_banner() -> tuple[Optional[datetime], int]:
    path = LOG_DIR / "paper_run_stderr.log"
    banner_dt: Optional[datetime] = None
    runtime_lines_after = 0
    active = False
    for line in load_text_lines(path):
        match = START_BANNER_RE.search(line)
        if match:
            banner_dt = parse_dt(match.group("ts"))
            runtime_lines_after = 0
            active = True
            continue
        if not active:
            continue
        if parse_dt(line):
            runtime_lines_after += 1
    return banner_dt, runtime_lines_after


def gather_anchor(events: list[tuple[datetime, str, str]]) -> datetime:
    latest = [dt for dt, _, _ in events]
    for path in sorted((DATA_DIR / "diagnostics").glob("diag_*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            latest.append(parse_dt(str(obj.get("tick_time", ""))) or datetime.fromtimestamp(path.stat().st_mtime, UTC))
        except Exception:
            latest.append(datetime.fromtimestamp(path.stat().st_mtime, UTC))
    for path in sorted((DATA_DIR / "shadow_ledger").glob("ledger_*.jsonl")):
        try:
            for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
                if not line.strip():
                    continue
                obj = json.loads(line)
                dt = parse_dt(str(obj.get("timestamp", "")))
                if dt:
                    latest.append(dt)
                    break
        except Exception:
            latest.append(datetime.fromtimestamp(path.stat().st_mtime, UTC))
    for path in sorted((DATA_DIR / "live_experiences").glob("experiences_*.jsonl")):
        try:
            for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
                if not line.strip():
                    continue
                obj = json.loads(line)
                dt = parse_dt(str(obj.get("timestamp", "")))
                if dt:
                    latest.append(dt)
                    break
        except Exception:
            latest.append(datetime.fromtimestamp(path.stat().st_mtime, UTC))
    if latest:
        return max(latest)
    return datetime.now(UTC)


def window_filter(items: Iterable[T], start: datetime, get_dt: Callable[[T], Optional[datetime]]) -> list[T]:
    return [item for item in items if get_dt(item) and get_dt(item) >= start]


def load_shadow_ledger(start: datetime) -> list[dict]:
    records: list[dict] = []
    ledger_dir = DATA_DIR / "shadow_ledger"
    for path in sorted(ledger_dir.glob("ledger_*.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                dt = parse_dt(str(obj.get("timestamp", "")))
                if dt and dt >= start:
                    obj["_dt"] = dt
                    obj["_path"] = str(path.relative_to(ROOT))
                    records.append(obj)
        except Exception:
            continue
    records.sort(key=lambda row: row["_dt"])
    return records


def load_live_experiences(start: datetime) -> list[dict]:
    records: list[dict] = []
    exp_dir = DATA_DIR / "live_experiences"
    for path in sorted(exp_dir.glob("experiences_*.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                dt = parse_dt(str(obj.get("timestamp", "")))
                if dt and dt >= start:
                    obj["_dt"] = dt
                    obj["_path"] = str(path.relative_to(ROOT))
                    records.append(obj)
        except Exception:
            continue
    records.sort(key=lambda row: row["_dt"])
    return records


def load_diagnostics(start: datetime) -> list[dict]:
    diag_dir = DATA_DIR / "diagnostics"
    rows: list[dict] = []
    for path in sorted(diag_dir.glob("diag_*.json")):
        dt = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        if dt < start:
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            tick_dt = parse_dt(str(obj.get("tick_time", ""))) or dt
            obj["_dt"] = tick_dt
            obj["_path"] = str(path.relative_to(ROOT))
            rows.append(obj)
        except Exception:
            continue
    rows.sort(key=lambda row: row["_dt"])
    return rows


def load_feature_manifest() -> dict:
    path = ROOT / "configs" / "feature_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_manual_step16_checks() -> dict:
    if not STEP16_MANUAL_CHECKS.exists():
        return {}
    try:
        obj = json.loads(STEP16_MANUAL_CHECKS.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def load_step15_status() -> dict:
    if not STEP15_STATUS.exists():
        return {}
    try:
        obj = json.loads(STEP15_STATUS.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def load_dashboard_state() -> dict:
    if not DASHBOARD_STATE.exists():
        return {}
    try:
        obj = json.loads(DASHBOARD_STATE.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def find_latest_profile(events: list[tuple[datetime, str, str]]) -> str:
    for _, _, line in reversed(events):
        match = PROFILE_RE.search(line)
        if match:
            return match.group("profile")
    return "UNKNOWN"


def build_report(hours: int = 48) -> dict:
    events = iter_log_events()
    anchor_end = gather_anchor(events)
    start_48h = anchor_end - timedelta(hours=hours)
    start_24h = anchor_end - timedelta(hours=24)
    step15_status = load_step15_status()
    dashboard_state = load_dashboard_state()

    events_48h = window_filter(events, start_48h, lambda item: item[0])
    events_24h = window_filter(events, start_24h, lambda item: item[0])
    ledger_48h = load_shadow_ledger(start_48h)
    ledger_24h = [row for row in ledger_48h if row["_dt"] >= start_24h]
    exp_48h = load_live_experiences(start_48h)
    exp_24h = [row for row in exp_48h if row["_dt"] >= start_24h]
    diag_48h = load_diagnostics(start_48h)
    diag_24h = [row for row in diag_48h if row["_dt"] >= start_24h]

    fills_48h = [row for row in ledger_48h if row.get("entry_type") == "FILL"]
    gate_rejects_48h = [row for row in ledger_48h if row.get("entry_type") == "GATE_REJECT"]
    fills_24h = [row for row in ledger_24h if row.get("entry_type") == "FILL"]
    gate_rejects_24h = [row for row in ledger_24h if row.get("entry_type") == "GATE_REJECT"]
    position_opens_48h = [
        row for row in ledger_48h
        if row.get("entry_type") == "POSITION"
        and abs(float(row.get("data", {}).get("old_size", 0.0) or 0.0)) <= 1e-9
        and abs(float(row.get("data", {}).get("new_size", 0.0) or 0.0)) > 1e-9
    ]
    fill_keys_48h = {
        (row.get("session_id"), row.get("tick_id"), row.get("asset"))
        for row in fills_48h
    }
    position_without_fill_48h = [
        row for row in position_opens_48h
        if (row.get("session_id"), row.get("tick_id"), row.get("asset")) not in fill_keys_48h
    ]

    crash_events = [
        {"timestamp": fmt_dt(dt), "source": source, "line": line}
        for dt, source, line in events_48h
        if any(marker in line for marker in CRASH_MARKERS)
    ]
    gmm_assets = sorted({match.group("asset") for _, _, line in events_24h for match in [GMM_RE.search(line)] if match})
    ood_loaded_assets = sorted({match.group("asset") for _, _, line in events_24h for match in [OOD_LOAD_RE.search(line)] if match})
    film_assets = sorted({match.group("asset") for _, _, line in events_24h for match in [LSTM_RE.search(line)] if match})
    soft_degrade_events = [(dt, line) for dt, _, line in events_48h if "[OOD]" in line and "soft degrade" in line]
    hard_switch_events = [(dt, line) for dt, _, line in events_48h if "[OOD]" in line and "HARD SWITCH" in line]

    obs_rows = [row for row in exp_24h if row.get("type") == "observation"]
    obs_rows_48h = [row for row in exp_48h if row.get("type") == "observation"]
    decision_rows_24h = [row for row in exp_24h if row.get("type") == "decision"]
    obs_lengths = [len(row.get("obs", [])) for row in obs_rows if isinstance(row.get("obs"), list)]
    actions = [float(row.get("action", 0.0)) for row in obs_rows if isinstance(row.get("action"), (int, float))]
    finite_actions = [value for value in actions if math.isfinite(value)]

    manifest = load_feature_manifest()
    feature_names = manifest.get("all_features", [])
    wavelet_summary = {}
    if feature_names and obs_rows:
        for feature in WAVELET_FEATURES:
            if feature not in feature_names:
                wavelet_summary[feature] = {"present": False}
                continue
            idx = feature_names.index(feature)
            values = []
            for row in obs_rows:
                obs = row.get("obs", [])
                if isinstance(obs, list) and len(obs) > idx:
                    try:
                        values.append(float(obs[idx]))
                    except Exception:
                        pass
            wavelet_summary[feature] = {
                "present": True,
                "samples": len(values),
                "abs_max": max((abs(v) for v in values), default=0.0),
                "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
            }

    type_counts_48h = Counter(row.get("type", "<missing>") for row in exp_48h)
    fills_by_asset_48h = Counter(row.get("asset", "?") for row in fills_48h)
    fills_by_asset_24h = Counter(row.get("asset", "?") for row in fills_24h)
    trade_attr_files = [
        path for path in sorted(DATA_DIR.glob("trade_attribution*.jsonl"))
        if path.exists() and path.stat().st_size > 0
    ]
    eq_files = [
        path for path in sorted((LOG_DIR / "execution_quality").glob("execution_quality_*.jsonl"))
        if path.exists() and path.stat().st_size > 0
    ]
    manual_checks = load_manual_step16_checks()
    runtime_updated_at = parse_dt(str(step15_status.get("updated_at", "")))
    runtime_started_at = parse_dt(str((step15_status.get("run", {}) or {}).get("started_at", "")))
    runtime_status_fresh = bool(runtime_updated_at and abs((anchor_end - runtime_updated_at).total_seconds()) <= 6 * 3600)
    runtime_within_24h = bool(runtime_started_at and runtime_started_at >= start_24h)
    runtime_within_48h = bool(runtime_started_at and runtime_started_at >= start_48h)
    dashboard_updated_at = parse_dt(str(dashboard_state.get("updated_at", "")))
    runtime_dashboard_fresh = bool(dashboard_updated_at and abs((anchor_end - dashboard_updated_at).total_seconds()) <= 6 * 3600)
    runtime_obs = step15_status.get("observation", {}) if isinstance(step15_status.get("observation"), dict) else {}
    runtime_wavelet = step15_status.get("wavelet", {}) if isinstance(step15_status.get("wavelet"), dict) else {}
    runtime_alpha = step15_status.get("alpha_gate", {}) if isinstance(step15_status.get("alpha_gate"), dict) else {}
    runtime_fills = step15_status.get("fills", {}) if isinstance(step15_status.get("fills"), dict) else {}
    runtime_positions = step15_status.get("positions", {}) if isinstance(step15_status.get("positions"), dict) else {}
    runtime_ood = step15_status.get("ood", {}) if isinstance(step15_status.get("ood"), dict) else {}
    runtime_kill = step15_status.get("kill_switch", {}) if isinstance(step15_status.get("kill_switch"), dict) else {}
    runtime_realized_pnl = step15_status.get("realized_pnl", {}) if isinstance(step15_status.get("realized_pnl"), dict) else {}
    runtime_execution_advisory = step15_status.get("execution_advisory", {}) if isinstance(step15_status.get("execution_advisory"), dict) else {}
    runtime_profit_calibration = step15_status.get("profit_calibration", {}) if isinstance(step15_status.get("profit_calibration"), dict) else {}
    runtime_drl = dashboard_state.get("drl", {}) if isinstance(dashboard_state.get("drl"), dict) else {}
    runtime_dashboard_assets = dashboard_state.get("assets", {}) if isinstance(dashboard_state.get("assets"), dict) else {}
    fee_bps_values = []
    for row in fills_48h:
        data = row.get("data", {})
        try:
            fee = float(data.get("fee", 0.0) or 0.0)
            notional = abs(float(data.get("notional", 0.0) or 0.0))
            if notional > 0 and fee >= 0:
                fee_bps_values.append(fee / notional * 10000.0)
        except Exception:
            continue

    decision_count_48h = max(len(diag_48h), len(obs_rows_48h), 0)
    ood_soft_degrade_ratio = safe_ratio(len(soft_degrade_events), decision_count_48h)

    alpha_gate_rows_24h = [
        row for row in decision_rows_24h
        if isinstance(row.get("alpha_gate_passed"), bool)
    ]
    alpha_gate_passes_24h = sum(1 for row in alpha_gate_rows_24h if row.get("alpha_gate_passed"))
    latest_banner_dt, latest_banner_runtime_lines = latest_start_banner()

    pass_rate_24h = (
        safe_ratio(alpha_gate_passes_24h, len(alpha_gate_rows_24h))
        if alpha_gate_rows_24h
        else safe_ratio(len(fills_24h), len(fills_24h) + len(gate_rejects_24h))
    )
    pass_rate_48h = safe_ratio(len(fills_48h), len(fills_48h) + len(gate_rejects_48h))

    latest_session_id = ""
    if ledger_48h:
        latest_session_id = str(ledger_48h[-1].get("session_id", ""))
    latest_session_records = [row for row in ledger_48h if str(row.get("session_id", "")) == latest_session_id] if latest_session_id else []
    latest_session_fills = [row for row in latest_session_records if row.get("entry_type") == "FILL"]
    latest_session_rejects = [row for row in latest_session_records if row.get("entry_type") == "GATE_REJECT"]
    latest_session_decisions = []
    if latest_banner_dt:
        latest_session_decisions = [
            row for row in decision_rows_24h
            if row.get("_dt") and row["_dt"] >= latest_banner_dt
            and isinstance(row.get("alpha_gate_passed"), bool)
        ]
    latest_session_pass_rate = (
        safe_ratio(
            sum(1 for row in latest_session_decisions if row.get("alpha_gate_passed")),
            len(latest_session_decisions),
        )
        if latest_session_decisions
        else safe_ratio(len(latest_session_fills), len(latest_session_fills) + len(latest_session_rejects))
    )

    first_candidates: list[datetime] = []
    if events_48h:
        first_candidates.append(events_48h[0][0])
    if ledger_48h:
        first_candidates.append(ledger_48h[0]["_dt"])
    if diag_48h:
        first_candidates.append(diag_48h[0]["_dt"])
    if exp_48h:
        first_candidates.append(exp_48h[0]["_dt"])
    first_artifact_48h = min(first_candidates) if first_candidates else None

    runtime_wavelet_ok = bool(runtime_wavelet.get("meets_nonzero_and_changing_evidence", False))
    wavelet_ok = runtime_wavelet_ok or (bool(wavelet_summary) and all(
        item.get("present") and item.get("samples", 0) > 0 and item.get("abs_max", 0.0) > 1e-8
        for item in wavelet_summary.values()
    ))
    runtime_ready_assets = sorted(
        str(asset).upper()
        for asset in list(runtime_drl.get("ready_assets", []) or [])
        if str(asset).upper() in {"BTC", "ETH", "SOL"}
    )
    runtime_ready_assets = sorted(
        set(runtime_ready_assets)
        | {
            str(asset).upper()
            for asset, payload in runtime_dashboard_assets.items()
            if str(asset).upper() in {"BTC", "ETH", "SOL"}
            and isinstance(payload, dict)
            and int(payload.get("drl_models_ready", 0) or 0) > 0
        }
    )
    runtime_film_assets = (
        list(runtime_ready_assets)
        if runtime_dashboard_fresh and int(runtime_drl.get("models_ready", 0) or 0) >= len(runtime_ready_assets)
        else []
    )
    runtime_ood_loaded_assets = sorted(
        str(asset).upper()
        for asset, payload in runtime_dashboard_assets.items()
        if str(asset).upper() in {"BTC", "ETH", "SOL"}
        and isinstance(payload, dict)
        and (
            payload.get("ood_distance") is not None
            or payload.get("ood_threshold") is not None
            or bool(payload.get("ood_hard_switch", False))
            or bool(payload.get("ood_shadow_alert", False))
        )
    )
    if not runtime_ood_loaded_assets and runtime_dashboard_fresh:
        runtime_ood_loaded_assets = list(runtime_ready_assets)
    effective_film_assets = sorted(set(film_assets) | set(runtime_film_assets))
    effective_ood_loaded_assets = sorted(set(ood_loaded_assets) | set(runtime_ood_loaded_assets))

    checks: list[CheckResult] = []

    coverage_hours = (anchor_end - first_artifact_48h).total_seconds() / 3600.0 if first_artifact_48h else 0.0
    checks.append(CheckResult(
        "Window Coverage",
        "PASS" if coverage_hours >= max(hours - 1, 1) else "WARN",
        f"{coverage_hours:.1f}h of artifacts ending at {fmt_dt(anchor_end)}",
    ))
    checks.append(CheckResult(
        "Latest Start Banner",
        "WARN" if latest_banner_dt and latest_banner_runtime_lines == 0 else "PASS",
        (
            f"{fmt_dt(latest_banner_dt)} with {latest_banner_runtime_lines} runtime lines after banner"
            if latest_banner_dt else "No start banner found in paper_run_stderr.log"
        ),
    ))
    checks.append(CheckResult(
        "48h Zero Crash",
        "PASS" if not crash_events else "FAIL",
        "no runtime error markers in last 48h" if not crash_events else crash_events[-1]["line"],
    ))
    checks.append(CheckResult(
        "GMM Source 100%",
        "PASS" if len(gmm_assets) == 3 else "FAIL",
        f"assets with [GMM] evidence in 24h: {', '.join(gmm_assets) or 'none'}",
    ))
    checks.append(CheckResult(
        "LSTM FiLM Extractor Loads",
        "PASS" if len(effective_film_assets) == 3 else "FAIL",
        (
            f"assets with LSTM_FILM_A load evidence in 24h/runtime: {', '.join(effective_film_assets) or 'none'} "
            f"(log={','.join(film_assets) or 'none'} runtime={','.join(runtime_film_assets) or 'none'})"
        ),
    ))
    checks.append(CheckResult(
        "OOD Detector Running",
        "PASS" if len(effective_ood_loaded_assets) == 3 and soft_degrade_events else ("FAIL" if not effective_ood_loaded_assets else "WARN"),
        (
            f"loaded={','.join(effective_ood_loaded_assets) or 'none'} "
            f"soft_degrade_events={len(soft_degrade_events)} hard_switches={len(hard_switch_events)} "
            f"(log={','.join(ood_loaded_assets) or 'none'} runtime={','.join(runtime_ood_loaded_assets) or 'none'})"
        ),
    ))
    runtime_obs_ok = (
        int(runtime_obs.get("single_obs_dim", 0) or 0) == OBS_DIM
        and int(runtime_obs.get("effective_stacked_dim", 0) or 0) == OBS_DIM * STACK_LEN
    )
    obs_shape_ok = runtime_obs_ok or (bool(obs_lengths) and all(length == OBS_DIM for length in obs_lengths))
    checks.append(CheckResult(
        "DRL Shadow Obs Shape",
        "PASS" if obs_shape_ok else "FAIL",
        (
            (
                f"runtime obs={runtime_obs.get('single_obs_dim')} stacked={runtime_obs.get('effective_stacked_dim')} "
                f"status_updated_at={step15_status.get('updated_at', 'n/a')}"
            )
            if runtime_obs_ok else (
                f"observation len={OBS_DIM}; implied stacked shape={OBS_DIM * STACK_LEN}"
                if obs_shape_ok else f"observation lengths found: {sorted(set(obs_lengths)) or 'none'}"
            )
        ),
    ))
    nontrivial = bool(finite_actions) and max(abs(v) for v in finite_actions) > 0.05
    no_nan = len(finite_actions) == len(actions)
    checks.append(CheckResult(
        "DRL Shadow Actions",
        "PASS" if nontrivial and no_nan else "FAIL",
        (
            f"samples={len(actions)} min={min(finite_actions):+.4f} max={max(finite_actions):+.4f}"
            if finite_actions else "no observation actions found"
        ),
    ))
    checks.append(CheckResult(
        "Wavelet Features Nonzero",
        "PASS" if wavelet_ok else "WARN",
        (
            (
                f"runtime meets_nonzero_and_changing={runtime_wavelet.get('meets_nonzero_and_changing_evidence')} "
                f"observed_assets={runtime_wavelet.get('observed_assets', [])}"
            )
            if runtime_wavelet_ok else (
                ", ".join(
                    f"{name}:abs_max={item.get('abs_max', 0.0):.4f}"
                    for name, item in wavelet_summary.items()
                )
                if wavelet_summary else "No observation evidence or feature manifest unavailable"
            )
        ),
    ))
    report_alpha_pass_rate = runtime_alpha.get("pass_rate") if (runtime_status_fresh and runtime_within_24h and runtime_alpha) else pass_rate_24h
    report_alpha_passed = int(runtime_alpha.get("passed", 0) or 0) if (runtime_status_fresh and runtime_within_24h and runtime_alpha) else alpha_gate_passes_24h
    report_alpha_total = int(runtime_alpha.get("total", 0) or 0) if (runtime_status_fresh and runtime_within_24h and runtime_alpha) else len(alpha_gate_rows_24h)
    checks.append(CheckResult(
        "Alpha Gate Pass Rate 24h",
        "PASS" if (report_alpha_pass_rate or 0.0) > 0.40 else "FAIL",
        (
            (
                f"runtime passes={report_alpha_passed} decisions={report_alpha_total} "
                f"pass_rate={fmt_pct(float(report_alpha_pass_rate or 0.0))} "
                f"status_updated_at={step15_status.get('updated_at', 'n/a')}"
            )
            if (runtime_status_fresh and runtime_within_24h and runtime_alpha) else (
                f"alpha_passes={alpha_gate_passes_24h} decisions={len(alpha_gate_rows_24h)} "
                f"pass_rate={fmt_pct(pass_rate_24h)} latest_session={latest_session_id or 'n/a'} "
                f"latest_session_decisions={len(latest_session_decisions)} latest_session_pass={fmt_pct(latest_session_pass_rate)}"
                if alpha_gate_rows_24h else
                f"fills={len(fills_24h)} rejects={len(gate_rejects_24h)} pass_rate={fmt_pct(pass_rate_24h)} latest_session={latest_session_id or 'n/a'} latest_session_pass={fmt_pct(latest_session_pass_rate)}"
            )
        ),
    ))
    runtime_fills_24h = (
        dict(runtime_fills.get("rolling_24h", {}) or {})
        if isinstance(runtime_fills.get("rolling_24h"), dict)
        else dict(runtime_fills)
    )
    runtime_fills_48h = (
        dict(runtime_fills.get("rolling_48h", {}) or {})
        if isinstance(runtime_fills.get("rolling_48h"), dict)
        else dict(runtime_fills)
    )

    def _merge_fill_views(runtime_view: dict, ledger_counter: Counter, ledger_total: int) -> tuple[int, dict]:
        runtime_counts = {
            str(asset): int(count or 0)
            for asset, count in dict(runtime_view.get("fills_by_asset", {}) or {}).items()
            if int(count or 0) > 0
        }
        merged_counts = {}
        for asset in runtime_counts:
            merged_value = max(int(runtime_counts.get(asset, 0) or 0), int(ledger_counter.get(asset, 0) or 0))
            if merged_value > 0:
                merged_counts[asset] = merged_value
        for asset in sorted(set(ledger_counter) - set(runtime_counts)):
            merged_value = int(ledger_counter.get(asset, 0) or 0)
            if merged_value > 0:
                merged_counts[asset] = merged_value
        merged_total = max(int(runtime_view.get("count", 0) or 0), int(ledger_total or 0), sum(merged_counts.values()))
        return merged_total, merged_counts

    runtime_rolling_24h_available = runtime_status_fresh and bool(runtime_fills_24h)
    runtime_rolling_48h_available = runtime_status_fresh and bool(runtime_fills_48h)

    if runtime_rolling_24h_available:
        report_fills_24h, report_fills_by_asset_24h = _merge_fill_views(runtime_fills_24h, fills_by_asset_24h, len(fills_24h))
    else:
        report_fills_24h = len(fills_24h)
        report_fills_by_asset_24h = dict(fills_by_asset_24h)
    checks.append(CheckResult(
        "Simulated Trades 24h",
        "PASS" if report_fills_24h >= 5 else "FAIL",
        f"fills_24h={report_fills_24h} by_asset={report_fills_by_asset_24h}",
    ))
    if runtime_rolling_48h_available:
        report_fills_48h, report_fills_by_asset_48h = _merge_fill_views(runtime_fills_48h, fills_by_asset_48h, len(fills_48h))
    else:
        report_fills_48h = len(fills_48h)
        report_fills_by_asset_48h = dict(fills_by_asset_48h)
    checks.append(CheckResult(
        "48h Trades / Assets",
        "PASS" if report_fills_48h >= 10 and len(report_fills_by_asset_48h) == 3 else "FAIL",
        f"fills_48h={report_fills_48h} by_asset={report_fills_by_asset_48h}",
    ))
    if position_without_fill_48h:
        _last_pos_wo_fill = position_without_fill_48h[-1]
        _last_pos_data = _last_pos_wo_fill.get("data", {})
        checks.append(CheckResult(
            "Paper Position Events Missing Fill",
            "WARN",
            (
                f"position_opens_without_fill={len(position_without_fill_48h)} "
                f"last={_last_pos_wo_fill.get('asset', '?')} "
                f"session={_last_pos_wo_fill.get('session_id', 'n/a')} "
                f"tick={_last_pos_wo_fill.get('tick_id', 'n/a')} "
                f"new_size={_last_pos_data.get('new_size', 'n/a')}"
            ),
        ))
    report_ood_ratio = runtime_ood.get("soft_degrade_ratio") if (runtime_status_fresh and runtime_within_48h and runtime_ood) else ood_soft_degrade_ratio
    report_ood_decisions = int(runtime_ood.get("scored_decisions", 0) or 0) if (runtime_status_fresh and runtime_within_48h and runtime_ood) else decision_count_48h
    report_ood_soft = int(runtime_ood.get("soft_degrade_count", 0) or 0) if (runtime_status_fresh and runtime_within_48h and runtime_ood) else len(soft_degrade_events)
    checks.append(CheckResult(
        "OOD Soft Degrade Ratio",
        "PASS" if report_ood_decisions > 0 and float(report_ood_ratio or 0.0) <= 0.10 else ("FAIL" if report_ood_decisions > 0 else "WARN"),
        f"soft_degrade={report_ood_soft} decisions={report_ood_decisions} ratio={fmt_pct(float(report_ood_ratio or 0.0))}",
    ))
    if fee_bps_values:
        median_fee_bps = statistics.median(fee_bps_values)
        fee_status = "PASS" if 5.0 <= median_fee_bps <= 50.0 else "WARN"
        fee_detail = (
            f"fee_bps median={median_fee_bps:.2f} min={min(fee_bps_values):.2f} "
            f"max={max(fee_bps_values):.2f} samples={len(fee_bps_values)}"
        )
    else:
        fee_status = "WARN"
        fee_detail = "No fills in window to estimate fee range"
    checks.append(CheckResult("Cost Logs In Range", fee_status, fee_detail))
    exp_growth_ok = len(exp_48h) > 0 and type_counts_48h.get("observation", 0) > 0
    checks.append(CheckResult(
        "LiveExperienceBuffer Growth",
        "PASS" if exp_growth_ok else "FAIL",
        f"records_48h={len(exp_48h)} type_counts={dict(type_counts_48h)}",
    ))
    checks.append(CheckResult(
        "LiveExperienceBuffer Structure",
        "PASS" if type_counts_48h.get("decision", 0) > 0 and type_counts_48h.get("outcome", 0) > 0 else "WARN",
        f"type_counts={dict(type_counts_48h)}",
    ))
    checks.append(CheckResult(
        "Trade Attribution Persisted",
        "PASS" if trade_attr_files else "WARN",
        trade_attr_files[-1].name if trade_attr_files else "No non-empty data/trade_attribution*.jsonl yet",
    ))
    checks.append(CheckResult(
        "Execution Quality Persisted",
        "PASS" if eq_files else "WARN",
        eq_files[-1].name if eq_files else "No non-empty logs/execution_quality/execution_quality_*.jsonl yet",
    ))
    _manual_ts = str(manual_checks.get("verified_at", "") or "")
    _manual_note = str(manual_checks.get("note", "") or "")
    _manual_suffix = (
        f" verified_at={_manual_ts}" + (f" note={_manual_note}" if _manual_note else "")
        if (_manual_ts or _manual_note) else ""
    )
    _risk_controls = manual_checks.get("risk_controls_verified")
    if _risk_controls is True:
        _risk_controls_status = "PASS"
        _risk_controls_detail = f"Manual artifact{_manual_suffix}".strip()
    elif _risk_controls is False:
        _risk_controls_status = "FAIL"
        _risk_controls_detail = f"Manual artifact marks risk_controls_verified=false{_manual_suffix}".strip()
    else:
        _risk_controls_status = "UNKNOWN"
        _risk_controls_detail = (
            f"No manual artifact at {STEP16_MANUAL_CHECKS.relative_to(ROOT)}"
        )
    checks.append(CheckResult(
        "Risk Controls Verification",
        _risk_controls_status,
        _risk_controls_detail,
    ))
    _kill_switch_runtime = runtime_kill.get("manual_test_passed")
    _kill_switch_runtime_at = runtime_kill.get("last_manual_test_at", "")
    _kill_switch_runtime_source = runtime_kill.get("last_manual_test_source", "")
    if _kill_switch_runtime is True:
        _kill_switch_status = "PASS"
        _kill_switch_detail = (
            f"step15_status manual_test_passed=true "
            f"at={_kill_switch_runtime_at or 'n/a'} source={_kill_switch_runtime_source or 'n/a'}"
        )
    else:
        _kill_switch = manual_checks.get("kill_switch_tested")
        if _kill_switch is True:
            _kill_switch_status = "PASS"
            _kill_switch_detail = f"Manual artifact{_manual_suffix}".strip()
        elif _kill_switch is False:
            _kill_switch_status = "FAIL"
            _kill_switch_detail = f"Manual artifact marks kill_switch_tested=false{_manual_suffix}".strip()
        else:
            _kill_switch_status = "UNKNOWN"
            _kill_switch_detail = (
                f"No manual artifact at {STEP16_MANUAL_CHECKS.relative_to(ROOT)}"
            )
    checks.append(CheckResult(
        "Kill Switch Manual Test",
        _kill_switch_status,
        _kill_switch_detail,
    ))
    _execution_guard_enabled = runtime_execution_advisory.get("guard_enabled")
    if runtime_execution_advisory:
        _execution_advisory_status = "PASS"
        _execution_advisory_detail = (
            f"guard_enabled={_execution_guard_enabled} "
            f"min_session_fills={runtime_execution_advisory.get('min_session_fills', 'n/a')} "
            f"min_asset_fills={runtime_execution_advisory.get('min_asset_fills', 'n/a')} "
            f"max_slippage_bps={runtime_execution_advisory.get('max_latest_slippage_bps', 'n/a')} "
            f"max_toxicity={runtime_execution_advisory.get('max_toxicity_rate', 'n/a')}"
        )
    else:
        _execution_advisory_status = "WARN"
        _execution_advisory_detail = "No runtime execution_advisory snapshot in step15_status.json"
    checks.append(CheckResult(
        "Execution Advisory Runtime",
        _execution_advisory_status,
        _execution_advisory_detail,
    ))
    _profit_by_side = runtime_profit_calibration.get("by_side", {}) if isinstance(runtime_profit_calibration.get("by_side"), dict) else {}
    _profit_long = _profit_by_side.get("long", {}) if isinstance(_profit_by_side.get("long"), dict) else {}
    _profit_short = _profit_by_side.get("short", {}) if isinstance(_profit_by_side.get("short"), dict) else {}
    if runtime_profit_calibration:
        _profit_calibration_status = "PASS"
        _profit_calibration_detail = (
            f"mode={runtime_profit_calibration.get('mode', 'n/a')} "
            f"long={_profit_long.get('recommendation', 'n/a')} "
            f"short={_profit_short.get('recommendation', 'n/a')} "
            f"min_closed_trades_per_side={runtime_profit_calibration.get('min_closed_trades_per_side', 'n/a')}"
        )
    else:
        _profit_calibration_status = "WARN"
        _profit_calibration_detail = "No runtime profit_calibration snapshot in step15_status.json"
    checks.append(CheckResult(
        "Profit Calibration Shadow",
        _profit_calibration_status,
        _profit_calibration_detail,
    ))

    step16_ready = all(
        result.status == "PASS"
        for result in checks
        if result.name in {
            "48h Zero Crash",
            "48h Trades / Assets",
            "OOD Soft Degrade Ratio",
            "Cost Logs In Range",
            "LiveExperienceBuffer Growth",
        }
    ) and all(
        result.status == "PASS"
        for result in checks
        if result.name in {"Risk Controls Verification", "Kill Switch Manual Test"}
    )

    return {
        "anchor_end_utc": fmt_dt(anchor_end),
        "window_hours": hours,
        "latest_profile_seen": find_latest_profile(events_48h),
        "latest_session_id": latest_session_id or None,
        "step16_ready": step16_ready,
        "checks": [asdict(item) for item in checks],
        "metrics": {
            "fills_24h": len(fills_24h),
            "gate_rejects_24h": len(gate_rejects_24h),
            "fills_48h": len(fills_48h),
            "gate_rejects_48h": len(gate_rejects_48h),
            "reported_fills_24h": report_fills_24h,
            "reported_fills_by_asset_24h": report_fills_by_asset_24h,
            "reported_fills_48h": report_fills_48h,
            "reported_fills_by_asset_48h": report_fills_by_asset_48h,
            "alpha_pass_rate_24h": pass_rate_24h,
            "alpha_pass_rate_48h": pass_rate_48h,
            "latest_session_pass_rate": latest_session_pass_rate,
            "diagnostics_48h": len(diag_48h),
            "experience_type_counts_48h": dict(type_counts_48h),
            "trade_attribution_files": [path.name for path in trade_attr_files],
            "execution_quality_files": [path.name for path in eq_files],
            "soft_degrade_events_48h": len(soft_degrade_events),
            "hard_switch_events_48h": len(hard_switch_events),
            "crash_events_48h": len(crash_events),
            "runtime_step15_status_loaded": bool(step15_status),
            "runtime_step15_status_updated_at": step15_status.get("updated_at"),
            "runtime_run_started_at": (step15_status.get("run", {}) or {}).get("started_at"),
            "runtime_fills_count": runtime_fills.get("count"),
            "runtime_assets_with_positions": (step15_status.get("positions", {}) or {}).get("assets_with_positions"),
            "runtime_realized_pnl_total_usd": (runtime_realized_pnl.get("total", {}) or {}).get("total_pnl_usd"),
            "runtime_realized_pnl_by_side": runtime_realized_pnl.get("by_side", {}),
            "runtime_execution_advisory": runtime_execution_advisory,
            "runtime_profit_calibration": runtime_profit_calibration,
            "manual_checks_file": str(STEP16_MANUAL_CHECKS.relative_to(ROOT)),
            "manual_checks_loaded": bool(manual_checks),
        },
        "crash_events": crash_events[-5:],
    }


def render_text(report: dict) -> str:
    lines = []
    lines.append("Step 15 / Step 16 Gate Report")
    lines.append(f"Anchor end (UTC): {report['anchor_end_utc']}")
    lines.append(f"Window hours: {report['window_hours']}")
    lines.append(f"Latest profile seen: {report['latest_profile_seen']}")
    lines.append(f"Latest session id: {report['latest_session_id'] or 'n/a'}")
    lines.append(f"Step 16 ready: {'YES' if report['step16_ready'] else 'NO'}")
    lines.append("")
    for item in report["checks"]:
        lines.append(f"[{item['status']}] {item['name']}: {item['detail']}")
    if report.get("crash_events"):
        lines.append("")
        lines.append("Recent crash markers:")
        for event in report["crash_events"]:
            lines.append(f"- {event['timestamp']} {event['source']}: {event['line']}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Report Step 15 / Step 16 readiness from local paper-run artifacts.")
    parser.add_argument("--hours", type=int, default=48, help="Lookback window in hours")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args(argv)

    report = build_report(hours=max(args.hours, 1))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
