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


# [P158] Regex-engine canary. Every pattern in this file is authored in
# PYTHON `re` syntax (they are built with re.escape and use \s / \b), but they
# are executed by git grep, whose engine depends on how git was BUILT. Apple
# git 2.39 with `-E` uses the system POSIX ERE, where \s and \b are NOT
# supported: `\b` matches nothing and `\s*` collapses to zero-width, so
# `agent_signals\["quant_direction"\]\s*=` never matches the real
# `agent_signals["quant_direction"] = sig`.
#
# The failure is silent in the worst possible way: the pattern is syntactically
# VALID, so git grep exits 1 = "no matches" — indistinguishable from a genuine
# absence. The 2026-04-25 hardening above only catches exit > 1 (syntax
# errors). On macOS this made Section A report "no direct writer" for all 20
# agents and Section B report 22 dead ENABLE_* flags, none of which are real.
#
# So: probe the engine once against a line in THIS file whose content is known,
# using both \s and \b. If the probe fails the escapes are not honoured, and
# every finding this scanner produces would be a false positive — that is a
# hard error, never a quiet degradation.
_REGEX_ENGINE_CANARY = "canary_probe = 1"  # do not edit: _detect_grep_mode greps this
_GREP_MODE: str | None = None


def _detect_grep_mode() -> str:
    """Return the git-grep regex flag whose engine honours \\s and \\b.

    Prefers `-P` (PCRE, exactly the Python syntax the patterns are written in)
    and falls back to `-E` only if the local git honours the escapes anyway
    (GNU builds do). Raises if neither works, because the alternative is
    emitting a wall of false findings that reads like a real regression.
    """
    global _GREP_MODE
    if _GREP_MODE is not None:
        return _GREP_MODE
    # Probe ALL THREE escape classes the patterns in this file actually use.
    # \s and \b alone are not enough: glibc regcomp (Linux CI) implements
    # those two but NOT \d, so an \s/\b-only canary would happily select `-E`
    # there and leave every \d pattern — DRL_PUNCH_THROUGH_CONF, BEST_FOLDS_ETH
    # — silently matching nothing, which is the exact bug P158 fixes.
    probe = r"\bcanary_probe\b\s*=\s*\d"
    for mode in ("-P", "-E"):
        try:
            r = subprocess.run(
                ["git", "grep", "-l", mode, probe],
                capture_output=True, text=True, cwd=REPO_ROOT,
            )
        except FileNotFoundError:
            break
        if r.returncode == 0 and "authority_consistency_audit.py" in r.stdout:
            _GREP_MODE = mode
            return mode
    raise RuntimeError(
        "[_git_grep] no available git grep engine honours \\s, \\b AND \\d "
        "(tried -P then -E). Every pattern in this audit is written in Python "
        "re syntax, so under such an engine the audit silently reports zero "
        "hits for wiring that exists — 20 phantom 'no direct writer' issues "
        "and 22 phantom dead flags on BSD regcomp, or a silently unevaluated "
        "DRL_PUNCH_THROUGH_CONF on glibc regcomp, which implements \\s and \\b "
        "but not \\d. Refusing to produce false findings. "
        "Install a git built with PCRE support (git grep -P)."
    )


def _git_grep(pattern: str, files_only: bool = False) -> list[str]:
    """Run git grep against the live tree (excluding archive/legacy).

    git grep exit codes:
      0 = matches found, 1 = no matches, 128 = error (e.g. invalid regex).
    Treating 128 as "no matches" silently masks regex-syntax bugs that
    permanently disable a scanner branch (caught by ultrareview on
    audit/safety-defense slice 1, 2026-04-25 — POSIX ERE rejected a
    Perl negative-lookahead and the audit never noticed). Surface any
    return code above 1 to stderr so the next syntax error is loud.

    [P158] Exit 1 is the *other* half of that hazard — a valid pattern the
    engine cannot honour also reports "no matches". See _detect_grep_mode.
    """
    mode = _detect_grep_mode()
    args = ["git", "grep", "-n", mode, pattern]
    if files_only:
        args = ["git", "grep", "-l", mode, pattern]
    try:
        r = subprocess.run(args, capture_output=True, text=True, cwd=REPO_ROOT)
    except FileNotFoundError:
        return []
    if r.returncode > 1:
        print(
            f"[_git_grep] git grep failed (exit {r.returncode}) for pattern "
            f"{pattern!r}: {r.stderr.strip()}",
            file=sys.stderr,
        )
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
        #    has a key for this agent. The runtime registers extractors under
        #    different keys than the matrix uses (e.g. matrix "microstructure"
        #    → runtime "micro" → _extract_micro). Use the alias map so we
        #    check the actual runtime key, not the matrix label.
        # Updated: 2026-04-25 (refined after first false-positive batch).
        extractor_alias = {
            "microstructure": "micro",
            "funding_rate": "funding",
        }
        extractor_key = extractor_alias.get(agent, agent)

        env_path = REPO_ROOT / "agents" / "signal_envelope.py"
        if env_path.exists():
            env_src = _read(env_path)
            # _EXTRACTORS: {"agent_name": ...}
            rec["attribution_extractor"] = bool(
                re.search(rf'["\']\s*{re.escape(extractor_key)}\s*["\']\s*:', env_src)
            )

        # Agents that are matrix-listed as DECIDE/ADVISE/CONFIRM but
        # architecturally non-direction-producing — they contribute as caps,
        # vetoes, timing edges, or boolean confirms, NOT as direction signals.
        # Per the comment at main.py:8731-8732, these are deliberately not
        # wrapped into attribution envelopes. Keep this list in sync with
        # that runtime comment.
        # Updated: 2026-04-25.
        non_direction_skip = {
            "regime",         # CONFIRM via direction-fallback only (D1 deferred)
            "macro",          # CAP authority — emits leverage cap, not direction
            "lead_lag",       # EXECUTE — timing edge, not direction
            "risk",           # VETO authority
            "structure",      # CONFIRM via boolean
            "squeeze",        # one-sided veto on squeeze_risk > threshold
            "cvd",            # one-sided divergence — not symmetric direction
            "risk_appetite",  # derived direction, fed via macro_risk_appetite cap path
            # NOTE: P57 promoted whale + options to direction-producing — they
            # ARE in _EXTRACTORS now and SHOULD be checked. Do NOT add them
            # here even though main.py:8731-8732's stale comment lists them.
        }

        if not rec["fusion_consumes"] and expected in (
            "DECIDE",
            "ADVISE",
            "CONFIRM",
            "EXECUTE",
        ):
            rec["issues"].append(
                f"DECIDE/ADVISE/CONFIRM agent — fusion does NOT consume '{actual_key}'"
            )

        if (
            expected in ("DECIDE", "ADVISE")
            and not rec["attribution_extractor"]
            and agent not in non_direction_skip
        ):
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
        # [P158] This check had NEVER RUN before 2026-08-04. It is the only
        # tracked constant whose pattern used `\d`, which git's POSIX engine
        # does not support on EITHER platform in use here — glibc regcomp
        # implements the GNU escapes \s \b \w but not \d, and BSD regcomp
        # implements none of them. So it silently matched nothing everywhere,
        # which is why the baseline has no entry for it despite the drift
        # predating the baseline commit. Now runs under `-P`.
        #
        # The old capture was `0\.(\d+)`, which reported "35"/"3" — and made
        # `expected: "30"` unsatisfiable by construction, since the canonical
        # sites are written `0.3`, not `0.30`. Capture the whole literal.
        "name": "DRL_PUNCH_THROUGH_CONF",
        "patterns": [
            r"_drl_conf\s*>=\s*(0\.\d+)",
        ],
        "expected": "0.3",
        "p_history": "P19/P20/P46 all use 0.30; P158 check first executed",
    },
    {
        # [P59 2026-04-25] hard_drawdown_halt — CLAUDE.md says canonical=20%
        # but live_high_risk.json uses 25%. config_resolver warns; both are
        # readers, so drift is observable but not always intentional.
        # Match only assignments / JSON literals — NOT format strings.
        "name": "hard_drawdown_halt",
        "patterns": [
            r'"hard_drawdown_halt"\s*:\s*([0-9.]+)',          # JSON
            # Python assignment — exclude docstring percent-form ("= 20%")
            # which canonical_config.py:14 uses for prose documentation.
            # Original `(?!\s*%)` was Perl negative lookahead and POSIX ERE
            # rejects it; git grep -E exited 128 silently (caught by
            # ultrareview on slice 1 audit/safety-defense, 2026-04-25). The
            # POSIX-compatible form below requires the value be followed by
            # a non-percent terminator (whitespace, comma, newline, EOL),
            # so docstring "20%" no longer matches but real assignments do.
            # The trailing terminator captures into group(2); audit code
            # uses group(1) so the value extraction is unchanged.
            r'\bhard_drawdown_halt\s*=\s*([0-9.]+)([^0-9.%]|$)',
        ],
        "expected": "0.25",  # match live_high_risk.json (the loaded config)
        "p_history": "canonical=0.20 vs live=0.25 — config_resolver warns",
    },
    {
        # initial_capital — multiple references, production .env=10000 but
        # scripts/tools default to 100000. Match only call-keyword form.
        "name": "initial_capital",
        "patterns": [
            # POSIX ERE has no non-capturing groups; use a plain group.
            # P68 _git_grep error-surfacing caught this silent regex bug
            # right after the same-shape bug_006 fix landed. Without the
            # surfacing, this pattern was returning [] (silent dead branch).
            r"\binitial_capital\s*=\s*([0-9]+(_[0-9]+)*)\b",
        ],
        "expected": "10000",  # production .env value
        "p_history": "scripts/ + tools/ default to 100000; production .env=10000",
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
# Section D: DRL feature/state invariants (P59)
# ----------------------------------------------------------------------

def audit_drl_invariants() -> dict[str, Any]:
    """Verify DRL feature manifest + state-space invariants per CLAUDE.md.

    Critical invariants (any drift here breaks training-serving parity):
      - feature_manifest.total_feature_count must equal len(all_features)
      - total_feature_count must equal 122 (CLAUDE.md documented value)
      - no_scale_features must contain regime_proba_0..7 + has_external_data
      - DRL state space = total_feature_count + 4 env state = 126
    """
    findings: dict[str, Any] = {"issues": []}

    manifest_path = REPO_ROOT / "configs" / "feature_manifest.json"
    if not manifest_path.exists():
        findings["issues"].append("configs/feature_manifest.json missing")
        return findings

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        findings["issues"].append(f"manifest parse error: {e}")
        return findings

    n_total = manifest.get("total_feature_count")
    all_feats = manifest.get("all_features", [])
    n_actual = len(all_feats)

    findings["total_feature_count"] = n_total
    findings["len_all_features"] = n_actual
    findings["expected_count"] = 122

    if n_total != n_actual:
        findings["issues"].append(
            f"total_feature_count={n_total} but len(all_features)={n_actual} "
            f"— manifest is internally inconsistent"
        )
    if n_total != 122:
        findings["issues"].append(
            f"total_feature_count={n_total} differs from CLAUDE.md "
            f"documented invariant 122 — DRL state space drifted"
        )

    # No-scale features must include 8 regime_proba + has_external_data.
    no_scale = set(manifest.get("no_scale_features", []))
    expected_no_scale = {f"regime_proba_{i}" for i in range(8)} | {
        "has_external_data"
    }
    missing_no_scale = expected_no_scale - no_scale
    extra_no_scale = no_scale - expected_no_scale
    if missing_no_scale:
        findings["issues"].append(
            f"no_scale_features missing: {sorted(missing_no_scale)}"
        )
    if extra_no_scale:
        findings["issues"].append(
            f"no_scale_features unexpected extras: {sorted(extra_no_scale)} "
            "— RobustScaler will skip them (may break inference)"
        )

    # DRL obs space = features + 4 env state.
    findings["expected_obs_dim"] = (n_total or 0) + 4
    findings["expected_obs_dim_documented"] = 126

    return findings


# ----------------------------------------------------------------------
# Section E: ENABLE_* readers depth (P59 — extends Section B)
# ----------------------------------------------------------------------

def audit_enable_flag_gates() -> dict[str, Any]:
    """For each ENABLE_* in configs/sota_flags.py, verify the reader is
    a REAL gate (`if FLAG:` / `getattr(flags, FLAG, ...)`) rather than
    just an import or string mention. P16 was about flags declared with
    no reader; this section is about flags read but not actually GATING.
    """
    flags_path = REPO_ROOT / "configs" / "sota_flags.py"
    if not flags_path.exists():
        return {"error": "sota_flags.py not found"}

    src = _read(flags_path)
    flag_decls = re.findall(r"^\s*(ENABLE_[A-Z0-9_]+)\s*:\s*bool\s*=", src, re.M)

    findings: dict[str, Any] = {}
    # Compile once per flag — Python regex (richer than ERE).
    for flag in flag_decls:
        # All mentions of the flag name outside the declaration / tests.
        all_hits = _git_grep(rf"\b{flag}\b")
        all_hits = [
            h for h in all_hits
            if not h.startswith("configs/sota_flags.py")
            and "test_" not in h
        ]
        # Now classify each hit line as a "real gate" or just a mention.
        # Patterns that count as a real control-flow gate:
        py_gate_re = re.compile(
            r'(?:'
            # 1. getattr(<expr>, "FLAG", ...) — canonical lazy-import gate
            rf'getattr\(.*["\'](?:{flag})["\']'
            # 2. <expr>.FLAG — direct attr access
            rf'|\.(?:{flag})\b'
            # 3. `if FLAG`/`elif FLAG`/`while FLAG` after `from x import FLAG`
            rf'|\b(?:if|elif|while)\s+(?:not\s+)?(?:{flag})\b'
            # 4. assignment from getattr (e.g. `_sa_enabled = getattr(..., "FLAG", ...)`)
            rf'|=\s*getattr\(.*["\'](?:{flag})["\']'
            r')'
        )
        gate_hits = []
        for h in all_hits:
            # Format: "path:line:matched_line"
            parts = h.split(":", 2)
            if len(parts) < 3:
                continue
            line_text = parts[2]
            if py_gate_re.search(line_text):
                gate_hits.append(h)
        rec: dict[str, Any] = {
            "n_real_gates": len(gate_hits),
            "n_total_mentions": len(all_hits),
            "gates_sample": gate_hits[:3],
            "issues": [],
        }
        if not gate_hits and all_hits:
            rec["issues"].append(
                f"{len(all_hits)} mentions but ZERO real gates — "
                "flag is read but never actually controls behavior (P16-shape)"
            )
        elif not all_hits:
            rec["issues"].append("DEAD: zero readers anywhere")
        findings[flag] = rec

    return {
        "n_flags": len(flag_decls),
        "by_flag": findings,
        "summary": {
            "no_real_gate": sorted(
                f for f, r in findings.items() if r["issues"]
            ),
        },
    }


# ----------------------------------------------------------------------
# Section F: Multi-call-site kwarg consistency (P60)
# ----------------------------------------------------------------------
# Per CLAUDE.md non-negotiable rule #6:
#   "Three trade_gate call sites — main veto_chain, authority_chain, AND
#   p0_safety_integrator ALL call `trade_gate.check()`. Fix ALL three when
#   changing the gate API."
#
# Per P57-B: same shape for `execute_intent_v2` — 3 call sites must all use
# the same kwargs (one missing `agent_signals` was flagged as intentional).
#
# This section finds every call to a tracked multi-site function and diffs
# the kwarg lists. If any kwarg appears in some sites but not others, flag.

TRACKED_MULTI_SITE_FUNCS = [
    {
        "name": "trade_gate.check",
        # Pattern matches `<expr>.trade_gate.check(...)` AND
        # `self.p0_integrator.trade_gate.check(...)` — the latter is the
        # second canonical site per P48.
        "call_pattern": r"trade_gate\.check\s*\(",
        "rule": "CLAUDE.md non-negotiable rule #6 — all 3 sites identical kwargs",
    },
    {
        "name": "execute_intent_v2",
        "call_pattern": r"\bexecute_intent_v2\s*\(",
        "rule": "P57-B verified 3 sites; site 1 (max-hold) intentionally omits agent_signals",
        "intentional_omits": {
            # Sites that legitimately omit a kwarg present in others.
            # Maps {kwarg: reason}.
            "agent_signals": "site 1 (MAX_HOLD_TIMEOUT) is rule-based exit, no signal context needed",
        },
    },
]


def _extract_call_kwargs(file_path: str, line_no: int, max_lines: int = 30) -> set[str]:
    """Read a function call starting at file:line_no and return the set of
    kwarg names passed. Stops at matching close-paren or 30 lines.
    """
    try:
        text = (REPO_ROOT / file_path).read_text(encoding="utf-8-sig", errors="ignore")
    except Exception:
        return set()
    lines = text.splitlines()
    start = max(0, line_no - 1)
    chunk = "\n".join(lines[start:start + max_lines])
    # Find first '(' and matching ')'
    depth = 0
    start_idx = chunk.find("(")
    if start_idx == -1:
        return set()
    end_idx = -1
    for i in range(start_idx, len(chunk)):
        if chunk[i] == "(":
            depth += 1
        elif chunk[i] == ")":
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx == -1:
        return set()
    body = chunk[start_idx + 1:end_idx]
    # [P60 fix] Strip Python comments (# to end-of-line) BEFORE parsing —
    # otherwise `# [BUGFIX] Use caller-provided values, not hardcoded` will
    # contaminate the next kwarg's word buffer (the `[` increases depth, the
    # `]` decreases, but text-after-`]` keeps appending to `word`).
    # NOTE: this is approximate — it doesn't handle # inside strings — but
    # for this scanner's purpose (finding kwarg names) the failure mode is
    # under-counting, never over-counting.
    body = re.sub(r"#[^\n]*", "", body)

    # Extract `kwname=` patterns. Strip out nested call args by tracking
    # paren-depth — a kwarg is at depth 0 only.
    kwargs = set()
    depth = 0
    word = ""
    seen_eq = False
    i = 0
    while i < len(body):
        c = body[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0 and c == "," and not seen_eq:
            word = ""
        elif depth == 0 and c == "=" and not seen_eq:
            # Single = (not ==) at depth 0 → previous word is a kwarg name.
            if i + 1 < len(body) and body[i + 1] == "=":
                pass  # ==, not assignment
            else:
                kw = word.strip()
                if kw and re.match(r"^[a-zA-Z_]\w*$", kw):
                    kwargs.add(kw)
                word = ""
                seen_eq = True
        elif depth == 0 and c == ",":
            word = ""
            seen_eq = False
        else:
            word += c
        i += 1
    return kwargs


# ----------------------------------------------------------------------
# Section G: Veto-reason classification (P74)
# ----------------------------------------------------------------------
# The post-tick invariant `_check_invariants` at main.py:13708-13742
# enforces "veto_active=True ⇒ target_exposure==0", with an allow-list
# of vetoes that legitimately leave exposure populated (the
# "block-new-entry" shape, where target_exposure carries the proposed
# size and downstream execution skips because veto_active=True).
#
# Two classes of veto exist in the codebase:
#   - HOLD vetoes (block-new-entry): MUST be in `_HOLD_VETOES`.
#   - HARD vetoes (must-flatten): MUST NOT be in `_HOLD_VETOES`.
#
# When a developer adds a new veto site (`intent.veto_reason = "..."`)
# they MUST classify it. P74 found that PATCH-4 SOFT and FRICTION
# were added without being added to the allow-list, producing CRITICAL
# log noise in production.
#
# This section enumerates every `*.veto_reason = "..."` assignment in
# the live tree, extracts the static prefix of each veto string, and
# flags any prefix that's neither in the allow-list nor in an
# operator-curated must-flatten deny-list.

# Operator-curated: vetoes that must flatten (NOT in allow-list).
# These should NOT appear in _HOLD_VETOES — adding them would silence
# a real "veto fired but didn't flatten" bug. Each entry is a
# normalized substring (uppercase, _→space, -→space).
MUST_FLATTEN_VETO_TAGS = {
    "PATCH 4] HARD",       # main.py:12107 — HARD correlation/dvol veto
    "P0 FORCE FLAT",       # main.py:9891/9911 — P0 force-flatten
    "BLACK SWAN SENTINEL", # integration_v36.py:845 — BSS=0 crisis
    "PROD] HARD VETO",     # integration_v36.py:1171/1724
    "TRANCHE DEADLOCK",    # integration_v36.py:1580
    "DEADLOCK ABORT",      # integration_v36.py:1614
    # P76 2026-04-26: risk_agent.py drawdown/correlation crisis vetoes.
    # All set risk_multiplier = 0.0 → explicit force-flatten intent.
    "CRITICAL DRAWDOWN",   # agents/risk_agent.py:594
    "HALT DRAWDOWN",       # agents/risk_agent.py:601
    "CORRELATION CRISIS",  # agents/risk_agent.py:622
}


def _parse_main_py_hold_vetoes() -> set[str]:
    """Read main.py's `_HOLD_VETOES` set. Returns the set of normalized
    substring tags. Returns empty set on any parse failure (caller
    treats that as 'all vetoes unclassified').

    Strips Python `# ...` comments per-line BEFORE extracting string
    literals, so quoted strings inside comments (used as cross-refs to
    other vetoes — e.g. `# don't substring-match "[PROD] HARD VETO"`)
    don't pollute the tag set."""
    main_py = REPO_ROOT / "main.py"
    if not main_py.exists():
        return set()
    src = _read(main_py)
    # Match the literal `_HOLD_VETOES = { ... }` block.
    m = re.search(
        r"_HOLD_VETOES\s*=\s*\{(.*?)\n\s*\}",
        src,
        re.DOTALL,
    )
    if not m:
        return set()
    block = m.group(1)
    # Strip Python comments line-by-line before extracting string literals.
    # Naive but correct for this block (no `#` inside the tag literals).
    stripped_lines = []
    for line in block.splitlines():
        comment_idx = line.find("#")
        if comment_idx >= 0:
            line = line[:comment_idx]
        stripped_lines.append(line)
    code_only = "\n".join(stripped_lines)
    tags = re.findall(r'"([^"]+)"', code_only)
    return {t.upper() for t in tags}


def _extract_static_prefix(line: str) -> str | None:
    """Given a code line like:
        intent.veto_reason = f"[PATCH-4] SOFT block(NORMAL): {x}"
    extract "[PATCH-4] SOFT block(NORMAL): " (everything before the
    first `{`).

    Plain strings:
        intent.veto_reason = "FRICTION_EXCEEDS_EDGE"
    extract "FRICTION_EXCEEDS_EDGE".

    Returns None for empty assignments (`= ""`) and unparseable lines.
    """
    # Find the RHS of the assignment.
    m = re.search(r'veto_reason\s*=\s*(.+)$', line)
    if not m:
        return None
    rhs = m.group(1).strip()
    # Strip trailing comma (kwarg form: `veto_reason="...",`).
    rhs = rhs.rstrip(",").strip()
    # Match f"..." or "..." or '...'.
    sm = re.match(r'^f?(["\'])(.*?)(?<!\\)\1', rhs)
    if not sm:
        return None
    raw = sm.group(2)
    if not raw:
        return None  # empty string assignment — clearing, not setting
    # Truncate at first interpolation `{...}`.
    if "{" in raw:
        raw = raw.split("{", 1)[0]
    raw = raw.strip()
    return raw or None


def audit_veto_reason_classification() -> dict[str, Any]:
    """Enumerate every veto_reason assignment, classify each as
    HOLD-listed / MUST-FLATTEN / UNCLASSIFIED."""
    hold_tags = _parse_main_py_hold_vetoes()

    # Find every veto_reason assignment in the live tree.
    # Pattern matches `<expr>.veto_reason = "..."` or `f"..."` or `[`.
    hits = _git_grep(r'veto_reason\s*=\s*[f"\[\x27]')
    assignments: list[dict[str, Any]] = []
    for h in hits:
        parts = h.split(":", 2)
        if len(parts) < 3:
            continue
        file_path, line_no, line_text = parts
        if not file_path.endswith(".py"):
            continue
        # Skip dataclass field declarations and function-arg defaults.
        if re.search(r"veto_reason\s*:\s*[A-Za-z]", line_text):
            continue
        if re.search(r"def\s+\w+\(.*veto_reason\s*=", line_text):
            continue
        prefix = _extract_static_prefix(line_text)
        if not prefix:
            continue
        # Skip purely-dynamic prefixes (`[`, `[ `, etc.) where the
        # actual veto string is an f-string interpolation like
        # `f"[{budget_result.veto_reason.value}] ..."`. Runtime
        # classification handles these via the inner string's tag.
        # The audit can't statically know what the value will be.
        if len(prefix.strip()) <= 2:
            continue
        normalized = prefix.upper().replace("_", " ").replace("-", " ")

        # Classify.
        in_hold = any(tag in normalized for tag in hold_tags)
        in_must_flatten = any(
            tag in normalized for tag in MUST_FLATTEN_VETO_TAGS
        )

        category = (
            "HOLD" if in_hold and not in_must_flatten
            else "MUST_FLATTEN" if in_must_flatten and not in_hold
            else "BOTH" if in_hold and in_must_flatten
            else "UNCLASSIFIED"
        )

        assignments.append({
            "location": f"{file_path}:{line_no}",
            "prefix": prefix,
            "normalized": normalized,
            "category": category,
        })

    # Group by category.
    by_category: dict[str, list[dict]] = {
        "HOLD": [],
        "MUST_FLATTEN": [],
        "BOTH": [],
        "UNCLASSIFIED": [],
    }
    for a in assignments:
        by_category[a["category"]].append(a)

    # Issues = UNCLASSIFIED + BOTH (latter is suspicious overlap).
    issues = []
    if by_category["UNCLASSIFIED"]:
        issues.append(
            f"{len(by_category['UNCLASSIFIED'])} veto_reason assignments "
            f"are NEITHER in _HOLD_VETOES NOR in MUST_FLATTEN_VETO_TAGS — "
            f"each must be classified or main.py:_check_invariants will "
            f"either CRITICAL-log false positives (if it should hold) or "
            f"silently miss a real flatten-bug (if it should hard-veto)."
        )
    if by_category["BOTH"]:
        issues.append(
            f"{len(by_category['BOTH'])} veto_reason assignments match BOTH "
            f"_HOLD_VETOES and MUST_FLATTEN_VETO_TAGS — substring tags "
            f"overlap. Tighten one tag or the other."
        )

    return {
        "n_total_assignments": len(assignments),
        "n_hold_tags_in_main": len(hold_tags),
        "n_must_flatten_tags": len(MUST_FLATTEN_VETO_TAGS),
        "by_category_counts": {k: len(v) for k, v in by_category.items()},
        "by_category": by_category,
        "issues": issues,
        "summary": {
            "unclassified_locations": [
                a["location"] for a in by_category["UNCLASSIFIED"]
            ],
            "both_locations": [
                a["location"] for a in by_category["BOTH"]
            ],
        },
    }


def audit_multi_site_consistency() -> dict[str, Any]:
    findings: dict[str, Any] = {}
    for fdef in TRACKED_MULTI_SITE_FUNCS:
        name = fdef["name"]
        sites = []
        for hit in _git_grep(fdef["call_pattern"]):
            parts = hit.split(":", 2)
            if len(parts) < 3:
                continue
            file_path, line_no, line_text = parts
            # Skip non-.py files (doc references in .md/.json/etc.)
            if not file_path.endswith(".py"):
                continue
            # Skip definitions, imports, comments — they aren't call sites.
            stripped = line_text.lstrip()
            if (
                stripped.startswith("def ")
                or stripped.startswith("async def ")
                or stripped.startswith("from ")
                or stripped.startswith("import ")
                or stripped.startswith("#")
                or stripped.startswith('"')
                or stripped.startswith("'")
            ):
                continue
            # Skip docstring prose: line like "3. Pre-execute: trade_gate.check()"
            # — has a digit-period-space prefix typical of doc list items.
            if re.match(r"^\s*\d+\.\s+\w", line_text):
                continue
            # Skip lines that look like a comma-separated method list inside
            # a docstring (e.g. "    trade_gate.check(), execution_guards.clip()")
            # — heuristic: more than one `()` pattern on the same line and no `=` /
            # `await` / `return` / `if`.
            if (
                line_text.count("()") + line_text.count("(self)") >= 2
                and not re.search(r"=|await|return|\bif\b|\bwhile\b", line_text)
            ):
                continue
            kwargs = _extract_call_kwargs(file_path, int(line_no))
            sites.append({
                "location": f"{file_path}:{line_no}",
                "kwargs": sorted(kwargs),
            })
        if not sites:
            findings[name] = {
                "n_sites": 0,
                "sites": [],
                "issues": [f"no call sites found for pattern {fdef['call_pattern']!r}"],
            }
            continue

        # Compute kwarg union and per-site diff.
        all_kwargs: set[str] = set()
        for s in sites:
            all_kwargs |= set(s["kwargs"])
        # For each site, list the kwargs it's MISSING vs the union.
        for s in sites:
            s["missing"] = sorted(all_kwargs - set(s["kwargs"]))

        # Issues: any site missing a kwarg that's not in intentional_omits.
        intentional = set((fdef.get("intentional_omits") or {}).keys())
        issues = []
        for s in sites:
            unexpected_missing = [m for m in s["missing"] if m not in intentional]
            if unexpected_missing:
                issues.append(
                    f"{s['location']} missing kwargs not in intentional_omits: "
                    f"{unexpected_missing}"
                )
        findings[name] = {
            "rule": fdef["rule"],
            "n_sites": len(sites),
            "sites": sites,
            "kwarg_union": sorted(all_kwargs),
            "intentional_omits": fdef.get("intentional_omits", {}),
            "issues": issues,
        }
    return {
        "by_function": findings,
        "summary": {
            "with_issues": sorted(
                k for k, v in findings.items() if v.get("issues")
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
        choices=[
            "authority", "flags", "constants",
            "drl", "gates", "multisite", "vetos", "all",
        ],
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
    if args.section in ("drl", "all"):
        out["drl_invariants"] = audit_drl_invariants()
    if args.section in ("gates", "all"):
        out["flag_gates"] = audit_enable_flag_gates()
    if args.section in ("multisite", "all"):
        out["multi_site"] = audit_multi_site_consistency()
    if args.section in ("vetos", "all"):
        out["vetos"] = audit_veto_reason_classification()

    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0

    # Human-readable summary.
    print("=" * 76)
    print("HMATS AUTHORITY / FLAG / CONSTANT CONSISTENCY AUDIT (P57-A + P59)")
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
        print(f"\n--- B) ENABLE_* FLAGS (declared, n={f.get('n_flags', 0)}) ---")
        dead = f.get("summary", {}).get("dead_flags", [])
        print(f"  dead flags: {len(dead)}")
        for flag in dead:
            print(f"  ✗ {flag} — declared but no runtime readers (P16-shape)")
        if not dead:
            print("  ✓ all declared flags have at least one reader")

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

    if "drl_invariants" in out:
        d = out["drl_invariants"]
        print("\n--- D) DRL FEATURE/STATE INVARIANTS (P59) ---")
        print(
            f"  total_feature_count={d.get('total_feature_count')} "
            f"(expected {d.get('expected_count')})"
        )
        print(
            f"  len(all_features)={d.get('len_all_features')}; "
            f"obs_dim={d.get('expected_obs_dim')} "
            f"(documented {d.get('expected_obs_dim_documented')})"
        )
        if d.get("issues"):
            for issue in d["issues"]:
                print(f"  ✗ {issue}")
        else:
            print("  ✓ DRL feature/state invariants consistent")

    if "flag_gates" in out:
        g = out["flag_gates"]
        print(f"\n--- E) ENABLE_* REAL GATE READERS (P59, n={g.get('n_flags', 0)}) ---")
        no_gate = g.get("summary", {}).get("no_real_gate", [])
        print(f"  flags without real gates: {len(no_gate)}")
        for flag in no_gate:
            rec = g["by_flag"][flag]
            print(
                f"  ✗ {flag} — {rec['n_total_mentions']} mentions, "
                f"{rec['n_real_gates']} real gates"
            )
            for sample in rec.get("gates_sample", []):
                print(f"      e.g. {sample[:140]}")
        if not no_gate:
            print("  ✓ all readable flags have at least one control-flow gate")

    if "vetos" in out:
        v = out["vetos"]
        print("\n--- G) VETO-REASON CLASSIFICATION (P74) ---")
        cc = v.get("by_category_counts", {})
        print(
            f"  total assignments: {v.get('n_total_assignments', 0)} "
            f"(HOLD={cc.get('HOLD', 0)}, MUST_FLATTEN={cc.get('MUST_FLATTEN', 0)}, "
            f"BOTH={cc.get('BOTH', 0)}, UNCLASSIFIED={cc.get('UNCLASSIFIED', 0)})"
        )
        for issue in v.get("issues", []):
            print(f"  ✗ {issue}")
        if v.get("by_category", {}).get("UNCLASSIFIED"):
            print("\n  Unclassified vetos (need allow-list OR deny-list entry):")
            for a in v["by_category"]["UNCLASSIFIED"][:20]:
                print(f"    {a['location']}  prefix={a['prefix']!r}")
        if v.get("by_category", {}).get("BOTH"):
            print("\n  Vetos matching BOTH lists (substring overlap — fix tags):")
            for a in v["by_category"]["BOTH"][:20]:
                print(f"    {a['location']}  prefix={a['prefix']!r}")
        if not v.get("issues"):
            print("  ✓ all veto_reason assignments classified")

    if "multi_site" in out:
        m = out["multi_site"]
        print("\n--- F) MULTI-CALL-SITE KWARG CONSISTENCY (P60) ---")
        with_issues = m.get("summary", {}).get("with_issues", [])
        for fname, rec in m.get("by_function", {}).items():
            n = rec.get("n_sites", 0)
            issues = rec.get("issues", [])
            marker = "✗" if issues else "✓"
            print(f"\n  {marker} {fname} ({n} call sites)")
            print(f"      rule: {rec.get('rule', '')}")
            for s in rec.get("sites", []):
                miss_str = (
                    f"  MISSING={s['missing']}" if s["missing"] else "  (full)"
                )
                print(f"      {s['location']:50}{miss_str}")
            if rec.get("intentional_omits"):
                print("      intentional omits:")
                for k, why in rec["intentional_omits"].items():
                    print(f"        - {k}: {why}")
            for issue in issues:
                print(f"      ✗ {issue}")
        if not with_issues:
            print("\n  ✓ all multi-site call kwargs consistent (or in intentional_omits)")

    print("\n" + "=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
