"""[P307f] ShortControl has never run in production — correcting P275/P276.

P275 recorded, under its findings, that "SHORT_CONTROL's veto-override IS
live on the sleeve", and P276 then hardened `_SC_PROTECTED_VETOES` on that
premise. Verified 2026-08-18 and the premise is false:

  * main.py builds the object only `if self.config.short_control_config`;
  * `short_control` has NEVER appeared in configs/live_high_risk.json
    (`git log -S` over that file returns no commit), and the only profile
    that sets it is configs/ultra_aggressive_5y.json, which is not what
    docker-compose runs;
  * the live engine logs ZERO `[P5-01] ShortControl: ACTIVE` lines.

The P276 hardening stays — it is correct if the module is ever enabled, and
removing a tightening because its subject turned out to be dormant would be
the wrong direction. What changes is the CLAIM: no veto-override has been
reaching the sleeve from here.

These tests exist so that enabling it becomes a DECISION rather than a side
effect: turning the key on flips this file red and sends the author to the
P141 activation rule.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _live() -> dict:
    return json.loads(
        (REPO / "configs" / "live_high_risk.json").read_text(encoding="utf-8"))


def test_short_control_is_absent_from_the_live_profile():
    """Adding the key CONSTRUCTS a module that can override short vetoes on
    the sleeve. That is a P141 activation and needs its own P-entry — not a
    config line added while fixing something else."""
    assert "short_control" not in _live(), (
        "short_control appeared in the live profile: this arms a veto-override "
        "path that has never run in production. Record the decision, and "
        "re-read P275 finding #3 and P276 item 2 — both were written when the "
        "module was believed live and are only correct once it actually is.")


def test_the_construction_is_still_gated_on_the_config_key():
    """If this gate is ever removed, the module becomes live everywhere and
    the test above stops meaning anything."""
    src = io.open(REPO / "main.py", encoding="utf-8").read()
    i = src.index("self._short_control = None")
    blk = src[i:i + 900]
    assert "if self.config.short_control_config:" in blk
    assert "self._short_control = ShortControl(config=_sc_cfg)" in blk


def test_the_bull_regime_set_is_reporting_only():
    """_BULL_REGIMES is stale against every live GMM vocabulary (only
    MOMENTUM_RALLY of its seven entries exists; the rest belong to the
    EnhancedRegimeNavigator's naming). That is harmless ONLY because the set
    feeds a reporting field and gates nothing: the override branch keys on
    `self.config.allow_short_in_bull`, not on `is_bull`. Pinned, because if
    `is_bull` ever becomes load-bearing the staleness turns into a live
    loosening — shorts permitted through STEADY_UPTREND, which the P307 GMM
    refit made 10-20% of bars."""
    src = io.open(REPO / "defense" / "short_control.py", encoding="utf-8").read()
    i = src.index("is_bull = regime_upper in _BULL_REGIMES")
    body = src[i:src.index("# --- Exposure cap ---", i)]
    uses = [ln for ln in body.split("\n")
            if "is_bull" in ln and "is_bull_regime=is_bull" not in ln
            and "is_bull = regime_upper" not in ln]
    assert not uses, (
        "is_bull is now consulted by logic, not just reported: "
        f"{uses}. _BULL_REGIMES is stale against the live GMM vocabulary, so "
        "this would permit shorts through genuinely bullish regimes.")
