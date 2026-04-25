"""
Authority / Flag / Constant consistency audit (P57-A).

Targets the three highest-frequency latent-bug classes per CLAUDE.md:

  A) AUTHORITY LEVEL drift across files
     - For each agent in AUTHORITY_MATRIX_NORMAL, verify the same authority
       level appears in:
         * signals/authority_fusion.py:AUTHORITY_MATRIX_NORMAL
         * agents/signal_envelope.py:_EXTRACTORS (or main.py:_ATTR_AUTHORITY)
       and that the matching `<agent>_direction` / `<agent>_confidence`
       keys are written somewhere (writer site exists) AND consumed by
       integration_v36.py:_build_fusion_signals (reader site exists).

  B) ENABLE_* flags without a real runtime gate
     - For each ENABLE_* in configs/sota_flags.py, check that at least one
       non-config call site does `getattr(flags, "ENABLE_X", ...)` or
       `if flags.ENABLE_X` against it. Decoration-only flags are P16-shape.

  C) Numerical constant drift
     - For a curated list of "known multi-location constants" (MAX_LEVERAGE,
       BEST_FOLDS, MAX_DATA_AGE_SECONDS, WEEKEND_MIN_CONFIDENCE, ...), grep
       every occurrence and flag mismatched values.

Output: JSON (machine-readable) + human-readable summary on stdout.
Run:   python -X utf8 scripts/authority_consistency_audit.py
       python -X utf8 scripts/authority_consistency_audit.py --json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

EXCLUDE_DIRS = (
    "archive/",
    "legacy/",
    "pytorch_build/",
    ".git/",
    "tests/",
    "docs/",
    "models/",
    "data/",
    "reports/",
    "scripts/",
)


def _git_grep(pattern: str, files_only: bool = False) -> list[str]:
    """Run git grep against the live tree (excluding archive/legacy)."""
    args = ["git", "grep", "-n", "-E", pattern]
    if files_only:
        args = ["git", "grep", "-l", "-E", pattern]
    try:
        r = subprocess.run(args, capture_output=True, text=True, cwd=REPO_ROOT)
    except FileNotFoundError:
        return []
    out = []
    for line in r.stdout.splitlines():
        if not line:
            continue
        if any(line.startswith(d) or "/" + d in line[:60] for d in EXCLUDE_DIRS):
            continue
        out.append(line)
    return out


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="ignore")


# ----------------------------------------------------------------------
# Section A: Authority level drift
# ----------------------------------------------------------------------

def audit_authority_consistency() -> dict[str, Any]:
    """For each agent in AUTHORITY_MATRIX_NORMAL, verify wiring quartet:
    writer key, fusion reader, attribution extractor, authority label.
    """
    af_path = REPO_ROOT / "signals" / "authority_fusion.py"
    af_src = _read(af_path)

    # Extract AUTHORITY_MATRIX_NORMAL block.
    m = re.search(
        r"AUTHORITY_MATRIX_NORMAL\s*=\s*\{(.*?)\n\}",
        af_src,
        re.DOTALL,
    )
    if not m:
        return {"error": "AUTHORITY_MATRIX_NORMAL block not found"}
    block = m.group(1)
    # Pull "name": Authority.LEVEL pairs.
    matrix = dict(re.findall(r'"([a-z_]+)"\s*:\s*Authority\.([A-Z_]+)', block))

    findings: dict[str, Any] = {}

    for agent, expected in matrix.items():
        rec: dict[str, Any] = {
            "matrix_authority": expected,
            "writer_key_found": False,
            "fusion_consumes": False,
            "attribution_extractor": False,
            "issues": [],
        }

        dir_key = f"{agent}_direction"
        # Some agents use different key names — handle aliases.
        aliases = {
            "kraken_quant": "kq_direction",
            "options": "options_short_confirmation",
            "structure": "structure_confirmed",
            "macro": "macro_leverage_cap",
            "lead_lag": "lead_lag_edge",
            "risk": "risk_veto",
            "squeeze": "squeeze_risk",
            "cvd": "cvd_divergence",
            "risk_appetite": "macro_risk_appetite",
            "microstructure": "micro_imbalance",
            "model_alpha": "model_alpha_direction",
            "soldex": "soldex_arb_direction",
            "flow": "flow_direction",
            "vol_alpha": "vol_alpha_direction",
            "whale": "whale_flow_direction",
            "two_stage": "two_stage_direction",
            "short_bias": "short_bias_direction",
            "funding_rate": "funding_direction",
            "onchain": "onchain_direction",
            "llm_sentiment": "llm_sentiment_direction",
            "onchain_graph": "onchain_graph_direction",
        }
        actual_key = aliases.get(agent, dir_key)

        # 1. Writer site (any line: agent_signals[KEY] = ...)
        writer_pat = (
            rf'agent_signals\[\s*["\']{re.escape(actual_key)}["\']\s*\]\s*='
        )
        writer_hits = _git_grep(writer_pat)
        rec["writer_key_found"] = bool(writer_hits)
        if not writer_hits:
            # Try dynamic dict-merge writers (e.g., agent_signals.update(...)
            # is hard to detect; treat absence here as a soft warning).
            rec["issues"].append(
                f"no direct writer for '{actual_key}' (may be set via dict-update)"
            )

        # 2. Fusion reader: integration_v36.py:_build_fusion_signals OR
        #    signals/authority_fusion.py reads.
        reader_pat = (
            rf'agent_signals\.get\(\s*["\']{re.escape(actual_key)}["\']'
        )
        reader_hits = _git_grep(reader_pat)
        rec["fusion_consumes"] = any(
            "integration_v36" in h or "authority_fusion" in h or "main.py" in h
            for h in reader_hits
        )

        # 3. Attribution extractor: agents/signal_envelope.py:_EXTRACTORS dict
        #    has a key for this agent.
        env_path = REPO_ROOT / "agents" / "signal_envelope.py"
        if env_path.exists():
            env_src = _read(env_path)
            # _EXTRACTORS: {"agent_name": ...}
            rec["attribution_extractor"] = bool(
                re.search(rf'["\']\s*{re.escape(agent)}\s*["\']\s*:', env_src)
            )

        if not rec["fusion_consumes"] and expected in (
            "DECIDE",
            "ADVISE",
            "CONFIRM",
            "EXECUTE",
        ):
            rec["issues"].append(
                f"DECIDE/ADVISE/CONFIRM agent — fusion does NOT consume '{actual_key}'"
            )

        if expected in ("DECIDE", "ADVISE") and not rec["attribution_extractor"]:
            rec["issues"].append(
                "no _EXTRACTORS entry — attribution will silently zero this agent (P3)"
            )

        findings[agent] = rec

    return {
        "matrix_size": len(matrix),
        "by_agent": findings,
        "summary": {
            "with_issues": sorted(
                a for a, r in findings.items() if r["issues"]
            ),
            "fully_clean": sorted(
                a for a, r in findings.items() if not r["issues"]
            ),
        },
    }


# ----------------------------------------------------------------------
# Section B: Dead ENABLE_* flags
# ----------------------------------------------------------------------

def audit_enable_flags() -> dict[str, Any]:
    """For each ENABLE_* declaration in configs/sota_flags.py, verify
    at least one runtime consumer exists (gate / instantiation guard)."""
    flags_path = REPO_ROOT / "configs" / "sota_flags.py"
    if not flags_path.exists():
        return {"error": "configs/sota_flags.py not found"}

    src = _read(flags_path)
    # Extract `ENABLE_NAME: bool = ...` declarations.
    flag_decls = re.findall(r"^\s*(ENABLE_[A-Z0-9_]+)\s*:\s*bool\s*=", src, re.M)

    findings: dict[str, Any] = {}
    for flag in flag_decls:
        # Look for runtime readers anywhere outside the declaration file.
        readers = []
        for hit in _git_grep(rf"\b{flag}\b"):
            if hit.startswith("configs/sota_flags.py"):
                continue
            if "test_" in hit:
                continue
            readers.append(hit)
        rec: dict[str, Any] = {
            "n_readers": len(readers),
            "readers_sample": readers[:3],
            "issues": [],
        }
        if not readers:
            rec["issues"].append("DEAD: declared but zero runtime readers (P16-shape)")
        findings[flag] = rec

    return {
        "n_flags": len(flag_decls),
        "by_flag": findings,
        "summary": {
            "dead_flags": sorted(
                f for f, r in findings.items() if r["issues"]
            ),
        },
    }


# ----------------------------------------------------------------------
# Section C: Numerical constant drift
# ----------------------------------------------------------------------

# Curated list of multi-location constants that have caused drift in past P-fixes.
TRACKED_CONSTANTS = [
    {
        "name": "MAX_LEVERAGE",
        "patterns": [
            r"\bMAX_LEVERAGE\s*[:=]\s*([0-9.]+)",
            r'"max_leverage"\s*:\s*([0-9.]+)',
        ],
        "expected": "3.0",
        "p_history": "P50 KRAKEN_DERIVS_MAX_LEVERAGE 5.0->3.0; live_phase1.json=2.0 intentional Phase 1",
    },
    {
        "name": "MAX_DATA_AGE_SECONDS",
        "patterns": [
            r"\bMAX_DATA_AGE_SECONDS\s*=\s*([0-9.]+)",
            r'"data_age_seconds"\.\s*max\s*[=:]\s*([0-9.]+)',
        ],
        "expected": "60.0",
        "p_history": "P22 schema vs runtime aligned to 60.0",
    },
    {
        "name": "WEEKEND_MIN_CONFIDENCE",
        "patterns": [
            r"\bWEEKEND_MIN_CONFIDENCE\s*=\s*([0-9.]+)",
            r'"min_confidence_weekend"\s*:\s*([0-9.]+)',
        ],
        "expected": "0.30",
        "p_history": "P52 lowered class default 0.50->0.30",
    },
    {
        "name": "WEEKEND_MIN_ALPHA_BPS",
        "patterns": [
            r"\bWEEKEND_MIN_ALPHA_BPS\s*=\s*([0-9.]+)",
            r'"weekend_min_alpha_bps"\s*:\s*([0-9.]+)',
        ],
        "expected": "20.0",
        "p_history": "P42 hardcoded 33->configurable 20",
    },
    {
        "name": "WEEKEND_MIN_ALPHA_MULTIPLIER",
        "patterns": [
            r"\bWEEKEND_MIN_ALPHA_MULTIPLIER\s*=\s*([0-9.]+)",
            r'"min_alpha_multiplier_weekend"\s*:\s*([0-9.]+)',
        ],
        "expected": "1.0",
        "p_history": "P42 lowered 2.0->1.0",
    },
    {
        "name": "BEST_FOLDS_ETH",
        "patterns": [
            r'BEST_FOLDS\s*=\s*\{[^}]*"ETH"\s*:\s*"(fold_\d)"',
        ],
        "expected": "fold_3",
        "p_history": "P4 + P53 ETH fold_1 stale -> fold_3",
    },
    {
        "name": "DRL_PUNCH_THROUGH_CONF",
        "patterns": [
            r"_drl_conf\s*>=\s*0\.(\d+)",  # most live thresholds are 0.30
        ],
        "expected": "30",
        "p_history": "P19/P20/P46 all use 0.30",
    },
]


def audit_constant_drift() -> dict[str, Any]:
    findings = {}
    for cdef in TRACKED_CONSTANTS:
        name = cdef["name"]
        all_values: dict[str, list[str]] = defaultdict(list)
        for pat in cdef["patterns"]:
            for hit in _git_grep(pat):
                # hit format: "path:line:matched_line"
                parts = hit.split(":", 2)
                if len(parts) < 3:
                    continue
                file_path = parts[0]
                line_no = parts[1]
                rest = parts[2]
                # Re-run the pattern against the rest to extract the value.
                m = re.search(pat, rest)
                if m:
                    val = m.group(1) if m.lastindex else m.group(0)
                    all_values[val].append(f"{file_path}:{line_no}")
        unique_values = sorted(all_values.keys())
        is_drift = len(unique_values) > 1
        findings[name] = {
            "expected": cdef["expected"],
            "observed_values": dict(all_values),
            "drift_detected": is_drift,
            "p_history": cdef["p_history"],
        }
    return {
        "by_constant": findings,
        "summary": {
            "drifted": sorted(
                k for k, v in findings.items() if v["drift_detected"]
            ),
        },
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--section",
        choices=["authority", "flags", "constants", "all"],
        default="all",
    )
    args = parser.parse_args()

    out: dict[str, Any] = {}
    if args.section in ("authority", "all"):
        out["authority"] = audit_authority_consistency()
    if args.section in ("flags", "all"):
        out["flags"] = audit_enable_flags()
    if args.section in ("constants", "all"):
        out["constants"] = audit_constant_drift()

    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0

    # Human-readable summary.
    print("=" * 76)
    print("HMATS AUTHORITY / FLAG / CONSTANT CONSISTENCY AUDIT (P57-A)")
    print("=" * 76)

    if "authority" in out:
        a = out["authority"]
        print(f"\n--- A) AUTHORITY MATRIX (n={a.get('matrix_size', 0)}) ---")
        s = a.get("summary", {})
        print(f"  with issues: {len(s.get('with_issues', []))}")
        print(f"  fully clean: {len(s.get('fully_clean', []))}")
        for agent in s.get("with_issues", []):
            rec = a["by_agent"][agent]
            print(f"\n  ✗ {agent} ({rec['matrix_authority']}):")
            for issue in rec["issues"]:
                print(f"      - {issue}")

    if "flags" in out:
        f = out["flags"]
        print(f"\n--- B) ENABLE_* FLAGS (n={f.get('n_flags', 0)}) ---")
        dead = f.get("summary", {}).get("dead_flags", [])
        print(f"  dead flags: {len(dead)}")
        for flag in dead:
            print(f"  ✗ {flag} — declared but no runtime readers (P16-shape)")

    if "constants" in out:
        c = out["constants"]
        print("\n--- C) NUMERICAL CONSTANT DRIFT ---")
        drifted = c.get("summary", {}).get("drifted", [])
        for name in drifted:
            rec = c["by_constant"][name]
            print(f"\n  ✗ {name}: expected {rec['expected']}")
            for val, locs in rec["observed_values"].items():
                marker = " ← drift" if val != rec["expected"] else ""
                print(f"      {val}{marker}")
                for loc in locs[:5]:
                    print(f"        {loc}")
        if not drifted:
            print("  ✓ no drift detected across tracked constants")

    print("\n" + "=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
