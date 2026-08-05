"""[P158] The authority audit must not silently match nothing.

Every pattern in scripts/authority_consistency_audit.py is authored in PYTHON
`re` syntax (built with re.escape, using \\s / \\b / \\d) but executed by
`git grep`, whose regex engine depends on how git was compiled:

  * glibc regcomp (Linux CI) — implements the GNU escapes \\s \\b \\w, NOT \\d
  * BSD regcomp (macOS, Apple git 2.39) — implements NONE of them
  * PCRE (`git grep -P`)          — implements all of them

An unsupported escape is not a syntax error. The pattern compiles, matches
nothing, and git grep exits 1 = "no matches" — indistinguishable from the
wiring genuinely being absent. That is how this scanner reported 20 phantom
"no direct writer" issues and 22 phantom dead ENABLE_* flags on macOS, and how
DRL_PUNCH_THROUGH_CONF (the one tracked constant written with \\d) went
unevaluated on every platform for the project's entire history.

These tests pin the engine contract itself, not any particular finding.
"""

import subprocess

import pytest

from scripts.authority_consistency_audit import (
    REPO_ROOT,
    TRACKED_CONSTANTS,
    _detect_grep_mode,
    _git_grep,
)


def _engine_supports(escape_pattern: str) -> bool:
    """True if the selected engine honours the pattern against a known line."""
    r = subprocess.run(
        ["git", "grep", "-l", _detect_grep_mode(), escape_pattern],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return r.returncode == 0


# ---------------------------------------------------------------------------
# engine contract
# ---------------------------------------------------------------------------

def test_selected_engine_honours_whitespace_escape():
    """`\\s` must match a real space, not collapse to zero-width.

    The canary line is exactly `canary_probe = 1` — one space either side.
    """
    assert _engine_supports(r"canary_probe\s=\s1"), (
        "a single \\s must match the single real space around '='"
    )
    assert _engine_supports(r"canary_probe\s*=\s*1"), (
        "\\s* must span the real spaces in 'canary_probe = 1'"
    )
    assert _engine_supports(r"canary_probe\s\s+=") is False, (
        "negative control: there is only ONE space before '=', so demanding "
        "two must fail. If this passes, \\s is being treated as a literal 's' "
        "and \\s+ is matching zero-width — the exact silent failure P158 fixes"
    )


def test_selected_engine_honours_word_boundary():
    assert _engine_supports(r"\bcanary_probe\b")


def test_selected_engine_honours_digit_class():
    """The gap that kept DRL_PUNCH_THROUGH_CONF dark on glibc too."""
    assert _engine_supports(r"canary_probe = \d")


def test_detect_grep_mode_is_stable_and_valid():
    mode = _detect_grep_mode()
    assert mode in ("-P", "-E")
    assert _detect_grep_mode() == mode  # cached, no re-probe drift


def test_a_digit_blind_engine_is_rejected_rather_than_selected(monkeypatch):
    """Simulates glibc regcomp (Linux CI): \\s and \\b work, \\d does not.

    An \\s/\\b-only canary would happily select `-E` there and leave every
    \\d pattern matching nothing — reintroducing P158 on the machine that
    gates every push. The probe must therefore exercise \\d too.
    """
    import scripts.authority_consistency_audit as audit

    real_run = subprocess.run

    def fake_run(args, **kwargs):
        if "-P" in args:                      # git built without PCRE
            class R:
                returncode, stdout, stderr = 128, "", "PCRE not supported"
            return R()
        if r"\d" in args[-1]:                 # glibc: \d unimplemented
            class R:
                returncode, stdout, stderr = 1, "", ""
            return R()
        return real_run(args, **kwargs)

    monkeypatch.setattr(audit, "_GREP_MODE", None)
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match=r"\\d"):
        audit._detect_grep_mode()


# ---------------------------------------------------------------------------
# the findings the broken engine fabricated
# ---------------------------------------------------------------------------

def test_writer_site_lookup_finds_a_known_writer():
    """`agent_signals["quant_direction"] = ...` exists in the live tree.

    Under a broken engine `\\s*\\]\\s*=` cannot span the real ' = ', so this
    returned [] and Section A reported "no direct writer" for all 20 agents.
    """
    hits = _git_grep(r'agent_signals\[\s*["\']quant_direction["\']\s*\]\s*=')
    assert hits, "writer site exists in core/trend_decision_layer.py"


def test_flag_lookup_finds_a_live_flag():
    """Under a broken engine `\\b` matched nothing, so every ENABLE_* flag
    looked dead — 22 phantom dead flags."""
    assert _git_grep(r"\bENABLE_TRADE_GATE\b")


def test_every_git_grep_pattern_in_tracked_constants_matches_something():
    """A tracked constant that matches nothing is an unevaluated check, which
    reads in the baseline exactly like a clean one."""
    unmatched = []
    for cdef in TRACKED_CONSTANTS:
        if not any(_git_grep(p) for p in cdef["patterns"]):
            unmatched.append(cdef["name"])
    assert not unmatched, (
        f"tracked constants whose patterns match nothing: {unmatched} — "
        f"either the constant was removed from the codebase (drop the entry) "
        f"or the pattern uses an escape the engine does not honour"
    )


def test_drl_punch_through_pattern_captures_the_full_literal():
    """Old capture `0\\.(\\d+)` reported '35' and made expected '30'
    unsatisfiable, since the canonical sites are written 0.3 not 0.30."""
    cdef = next(c for c in TRACKED_CONSTANTS
                if c["name"] == "DRL_PUNCH_THROUGH_CONF")
    assert cdef["expected"] == "0.3"
    import re
    m = re.search(cdef["patterns"][0], "if _drl_conf >= 0.35 and x:")
    assert m and m.group(1) == "0.35"
