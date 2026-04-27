"""
ci_check_invariants.py — gate the static scanners against a frozen baseline.

Runs `authority_consistency_audit.py` and `silent_failure_audit.py`,
compares their output to baselines in `tools/scanner_baselines/`, exits
0 if there's no NEW finding vs the baseline and 1 otherwise.

The comparison is structural for the authority scanner (per-agent issues
+ per-constant drift) and count-based for silent-failure (counts can't
go up without explicit baseline update).

Usage
-----
    # CI mode (default) — fail if new findings:
    python -X utf8 tools/ci_check_invariants.py

    # Re-baseline after an intentional change (e.g. you accept a new
    # constant drift, or a new false positive surfaced):
    python -X utf8 tools/ci_check_invariants.py --update

    # Pretty diff to stdout without exit-code semantics:
    python -X utf8 tools/ci_check_invariants.py --diff-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "tools" / "scanner_baselines"
AUTHORITY_BASELINE = BASELINES_DIR / "authority_consistency_baseline.json"
SILENT_BASELINE = BASELINES_DIR / "silent_failure_baseline.json"
SWALLOW_BASELINE = BASELINES_DIR / "silent_swallow_baseline.json"
NAIVE_DT_BASELINE = BASELINES_DIR / "naive_datetime_baseline.json"  # [P111]
SELF_CONFIG_BASELINE = BASELINES_DIR / "self_config_undefined_baseline.json"  # [P111]
MYPY_BASELINE = BASELINES_DIR / "mypy_baseline.json"  # [P113 (4/6)]


def _run_scanner(args: List[str]) -> Dict[str, Any]:
    """Run a scanner and parse its JSON stdout. Stderr is allowed (non-fatal
    diagnostics from scanners go there). Non-zero exit is fatal — scanner
    crashed and the gate can't make a decision."""
    r = subprocess.run(
        [sys.executable, "-X", "utf8", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if r.returncode != 0:
        print(
            f"[ci_check] scanner crashed (exit {r.returncode}):\n"
            f"  cmd: {' '.join(args)}\n"
            f"  stderr:\n{r.stderr}",
            file=sys.stderr,
        )
        sys.exit(2)
    if r.stderr.strip():
        # Scanners surface git-grep errors etc. to stderr. Echo them so
        # CI logs make these visible (they were silent pre-13812f5).
        print(r.stderr, file=sys.stderr)
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as e:
        print(
            f"[ci_check] scanner produced non-JSON stdout: {e}\n"
            f"  cmd: {' '.join(args)}\n"
            f"  stdout (first 500 chars):\n{r.stdout[:500]}",
            file=sys.stderr,
        )
        sys.exit(2)


def _normalize_authority(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce the authority scanner output to a stable comparison shape.

    Keep only fields that meaningfully signal "new bug". Strip volatile
    metadata like file/line numbers (those move on any edit). Bucket drifts
    by VALUE not file:line so cosmetic moves don't trip CI.
    """
    out: Dict[str, Any] = {}

    # --- Section A (authority matrix) ---
    auth = raw.get("authority", {})
    out["authority"] = {
        "matrix_size": auth.get("matrix_size", 0),
        "by_agent_issues": {
            a: sorted(rec.get("issues", []))
            for a, rec in (auth.get("by_agent") or {}).items()
            if rec.get("issues")
        },
    }

    # --- Section B (declared ENABLE_* flags w/o reader) ---
    flags = raw.get("flags", {})
    flags_summary = flags.get("summary") or {}
    dead_flags = flags_summary.get("dead", []) or flags_summary.get("dead_flags", [])
    out["flags"] = {
        "dead_count": len(dead_flags),
        "dead_flags": sorted(dead_flags),
    }

    # --- Section C (numerical constant drift) ---
    constants = raw.get("constants", {})
    out["constants"] = {}
    for cname, cdef in (constants.get("by_constant") or {}).items():
        observed = cdef.get("observed_values") or {}
        expected = str(cdef.get("expected", ""))
        # Drift = any observed value != expected
        drift_values = sorted(v for v in observed.keys() if str(v) != expected)
        if drift_values:
            out["constants"][cname] = {
                "expected": expected,
                "drift_values": drift_values,
            }

    # --- Section D (DRL invariants) ---
    drl = raw.get("drl_invariants", {})
    out["drl"] = {
        "issues": sorted(drl.get("issues", [])),
        "total_feature_count": drl.get("total_feature_count"),
        "expected_count": drl.get("expected_count"),
        "expected_obs_dim": drl.get("expected_obs_dim"),
    }

    # --- Section E (flag real-gate audit) ---
    gates = raw.get("flag_gates", {})
    no_real_gate = (gates.get("summary") or {}).get("no_real_gate", [])
    out["gates"] = {
        "without_real_gates_count": len(no_real_gate),
        "without_real_gates": sorted(no_real_gate),
    }

    # --- Section F (multi-call-site kwarg consistency) ---
    multisite = raw.get("multi_site", {})
    out["multisite"] = {}
    for fname, fdef in (multisite.get("by_function") or {}).items():
        # An issue is when a site is MISSING a kwarg that's not in
        # intentional_omits. Sites listed as "(full)" have no missing.
        intentional = set((fdef.get("intentional_omits") or {}).keys())
        bad_sites: List[str] = []
        for site in (fdef.get("sites") or []):
            missing = set(site.get("missing", []) or [])
            real_missing = missing - intentional
            if real_missing:
                bad_sites.append(
                    f"{site.get('location', '?')}:missing={sorted(real_missing)}"
                )
        if bad_sites:
            out["multisite"][fname] = sorted(bad_sites)

    # --- Section G (veto-reason classification, P74) ---
    # Lock in the COUNT of UNCLASSIFIED + BOTH veto_reason assignments
    # so a new unclassified veto fails CI. The actual list of locations
    # is also tracked for diff visibility but the count is the gate.
    vetos = raw.get("vetos", {})
    veto_counts = vetos.get("by_category_counts", {})
    veto_summary = vetos.get("summary", {})
    out["vetos"] = {
        "unclassified_count": veto_counts.get("UNCLASSIFIED", 0),
        "both_count": veto_counts.get("BOTH", 0),
        # Track sorted location list — a new unclassified at a NEW
        # location adds a string to the list, which the diff catches.
        "unclassified_locations": sorted(
            veto_summary.get("unclassified_locations") or []
        ),
        "both_locations": sorted(veto_summary.get("both_locations") or []),
    }

    return out


def _normalize_silent(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Silent-failure scanner output → comparable counts.

    The scanner produces 1000+ hits across 3 patterns; per-line tracking
    would be too noisy. Instead we lock in COUNTS per pattern and refuse
    to let them go up without explicit baseline update.
    """
    return {
        "tryexcept_count": len(raw.get("tryexcept_hits") or []),
        "dictget_count": len(raw.get("dictget_hits") or []),
        "flags_count": len(raw.get("flags_hits") or []),
    }


def _normalize_swallow(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Silent-swallow lint output → comparable counts.

    Unannotated silent-except blocks. Bucketed by kind so a regression
    in one category surfaces independently. Files-with-findings count
    tracked separately so refactors that consolidate sites without
    fixing them don't accidentally go green.
    """
    findings = raw.get("findings") or []
    return {
        "total_count": len(findings),
        "by_kind": {
            "pass": sum(1 for f in findings if f.get("kind") == "pass"),
            "debug": sum(1 for f in findings if f.get("kind") == "debug"),
            "no-log-early-exit": sum(
                1 for f in findings if f.get("kind") == "no-log-early-exit"
            ),
        },
        "files_with_findings": len({f.get("file") for f in findings}),
    }


def _diff(label: str, baseline: Any, current: Any) -> List[str]:
    """Return list of human-readable diffs (empty = clean)."""
    diffs: List[str] = []
    if isinstance(baseline, dict) and isinstance(current, dict):
        all_keys = sorted(set(baseline.keys()) | set(current.keys()))
        for k in all_keys:
            sub_label = f"{label}.{k}"
            if k not in baseline:
                diffs.append(f"+ {sub_label}: NEW = {current[k]!r}")
            elif k not in current:
                # Removed entries are GOOD — fewer findings. Don't fail.
                pass
            else:
                diffs.extend(_diff(sub_label, baseline[k], current[k]))
    elif isinstance(baseline, list) and isinstance(current, list):
        b_set = set(map(str, baseline))
        c_set = set(map(str, current))
        new = sorted(c_set - b_set)
        if new:
            diffs.append(f"+ {label}: NEW entries = {new}")
    elif isinstance(baseline, (int, float)) and isinstance(current, (int, float)):
        if current > baseline:
            diffs.append(
                f"+ {label}: count INCREASED {baseline} → {current} "
                f"(+{current - baseline})"
            )
    elif baseline != current:
        diffs.append(f"~ {label}: {baseline!r} → {current!r}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--update",
        action="store_true",
        help="Re-write baselines with the current scanner output. "
             "Use after an intentional change.",
    )
    ap.add_argument(
        "--diff-only",
        action="store_true",
        help="Print diff but always exit 0 (dev/inspection mode).",
    )
    args = ap.parse_args()

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    print("[ci_check] running authority_consistency_audit...", file=sys.stderr)
    auth_raw = _run_scanner([
        "scripts/authority_consistency_audit.py",
        "--section", "all",
        "--json",
    ])
    auth_norm = _normalize_authority(auth_raw)

    print("[ci_check] running silent_failure_audit...", file=sys.stderr)
    silent_raw = _run_scanner([
        "scripts/silent_failure_audit.py",
        "--pattern", "all",
        "--json",
    ])
    silent_norm = _normalize_silent(silent_raw)

    print("[ci_check] running lint_silent_swallow...", file=sys.stderr)
    swallow_raw = _run_scanner([
        "tools/lint_silent_swallow.py",
        "--json",
    ])
    swallow_norm = _normalize_swallow(swallow_raw)

    # [P111 Tier2#4 2026-04-27] lint_naive_datetime added to gate.
    # Same baseline-diff semantics as silent_swallow.
    print("[ci_check] running lint_naive_datetime...", file=sys.stderr)
    naive_dt_raw = _run_scanner([
        "tools/lint_naive_datetime.py",
        "--baseline-format",
    ])
    naive_dt_norm = naive_dt_raw  # already in normalized shape

    # [P111 Tier2#5 2026-04-27] lint_self_config_undefined added to gate.
    print("[ci_check] running lint_self_config_undefined...", file=sys.stderr)
    self_config_raw = _run_scanner([
        "tools/lint_self_config_undefined.py",
        "--baseline-format",
    ])
    self_config_norm = self_config_raw

    # [P113 (4/6) 2026-04-27] mypy baseline — locks current 982-error
    # state by error-code; new errors block CI, fixes lower the count.
    print("[ci_check] running lint_mypy_baseline...", file=sys.stderr)
    mypy_raw = _run_scanner([
        "tools/lint_mypy_baseline.py",
        "--baseline-format",
    ])
    mypy_norm = mypy_raw

    if args.update:
        AUTHORITY_BASELINE.write_text(
            json.dumps(auth_norm, indent=2, sort_keys=True), encoding="utf-8"
        )
        SILENT_BASELINE.write_text(
            json.dumps(silent_norm, indent=2, sort_keys=True), encoding="utf-8"
        )
        SWALLOW_BASELINE.write_text(
            json.dumps(swallow_norm, indent=2, sort_keys=True), encoding="utf-8"
        )
        NAIVE_DT_BASELINE.write_text(
            json.dumps(naive_dt_norm, indent=2, sort_keys=True), encoding="utf-8"
        )
        SELF_CONFIG_BASELINE.write_text(
            json.dumps(self_config_norm, indent=2, sort_keys=True), encoding="utf-8"
        )
        MYPY_BASELINE.write_text(
            json.dumps(mypy_norm, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"[ci_check] baselines updated:\n"
            f"  - {AUTHORITY_BASELINE.relative_to(REPO_ROOT)}\n"
            f"  - {SILENT_BASELINE.relative_to(REPO_ROOT)}\n"
            f"  - {SWALLOW_BASELINE.relative_to(REPO_ROOT)}\n"
            f"  - {NAIVE_DT_BASELINE.relative_to(REPO_ROOT)}\n"
            f"  - {SELF_CONFIG_BASELINE.relative_to(REPO_ROOT)}\n"
            f"  - {MYPY_BASELINE.relative_to(REPO_ROOT)}",
            file=sys.stderr,
        )
        return 0

    missing = [
        p.name for p in (AUTHORITY_BASELINE, SILENT_BASELINE,
                         SWALLOW_BASELINE, NAIVE_DT_BASELINE,
                         SELF_CONFIG_BASELINE, MYPY_BASELINE)
        if not p.exists()
    ]
    if missing:
        print(
            f"[ci_check] no baseline files yet for: {missing}. "
            f"Run with --update to seed.",
            file=sys.stderr,
        )
        return 2

    auth_baseline = json.loads(AUTHORITY_BASELINE.read_text(encoding="utf-8"))
    silent_baseline = json.loads(SILENT_BASELINE.read_text(encoding="utf-8"))
    swallow_baseline = json.loads(SWALLOW_BASELINE.read_text(encoding="utf-8"))
    naive_dt_baseline = json.loads(NAIVE_DT_BASELINE.read_text(encoding="utf-8"))
    self_config_baseline = json.loads(SELF_CONFIG_BASELINE.read_text(encoding="utf-8"))
    mypy_baseline = json.loads(MYPY_BASELINE.read_text(encoding="utf-8"))

    auth_diffs = _diff("authority", auth_baseline, auth_norm)
    silent_diffs = _diff("silent", silent_baseline, silent_norm)
    swallow_diffs = _diff("swallow", swallow_baseline, swallow_norm)
    naive_dt_diffs = _diff("naive_datetime", naive_dt_baseline, naive_dt_norm)
    self_config_diffs = _diff("self_config", self_config_baseline, self_config_norm)
    mypy_diffs = _diff("mypy", mypy_baseline, mypy_norm)

    if not auth_diffs and not silent_diffs and not swallow_diffs and not naive_dt_diffs and not self_config_diffs and not mypy_diffs:
        print("[ci_check] OK — no new findings vs baseline.", file=sys.stderr)
        return 0

    print("=" * 70)
    print("[ci_check] NEW SCANNER FINDINGS vs baseline:")
    print("=" * 70)
    for d in auth_diffs:
        print(f"  {d}")
    for d in silent_diffs:
        print(f"  {d}")
    for d in swallow_diffs:
        print(f"  {d}")
    for d in naive_dt_diffs:
        print(f"  {d}")
    for d in self_config_diffs:
        print(f"  {d}")
    for d in mypy_diffs:
        print(f"  {d}")
    print("=" * 70)
    print(
        "If these are intentional, re-baseline:\n"
        "  python -X utf8 tools/ci_check_invariants.py --update\n"
        "and commit the updated tools/scanner_baselines/*.json files.\n"
        "\n"
        "If new silent-swallow findings: either annotate with\n"
        "`# noqa: silent-swallow` (intentional) or convert the swallow into\n"
        "a logger.warning/error call with type(e).__name__ + context."
    )

    return 0 if args.diff_only else 1


if __name__ == "__main__":
    sys.exit(main())
