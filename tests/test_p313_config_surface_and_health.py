"""
[P313] The config surface that could neither take effect nor complain — and
the hold-cost trap that fixing it naively would have ARMED.

Found by a read-through, verified through the real parser rather than by
reading the dataclass:

  * 28 ProductionConfig fields were declared, defaulted (17 to True), read on
    the live path, and assigned NOWHERE in from_file. Writing
    `profit_max_enabled: false` into the live profile returned True.
  * validate_loaded_config only inspected the `risk` and `tranche` SECTIONS,
    so the same key produced no "unknown key" warning either. Silent in both
    directions: it could not work, and it could not tell you.
  * `regime_leverage_enabled` — one of the unsettable flags — transitively
    gated P291b's venue-true hold cost, because `update_funding_rate` has
    exactly ONE production caller and it sat inside that flag's block. With
    the flag off the whole hold-cost term collapses to 0.0, the UNDERCHARGE
    direction (P167). So making the flags settable WITHOUT decoupling that
    call would have shipped a live hazard rather than closed one — which is
    why both land here together.
  * HEALTH_T1 keyed on a strategy LABEL, and the P298 seat overwrites
    direction/edge without touching it, so a certified-flat book WARNed every
    tick on 2 of 3 assets — blunting the one check that exists to catch a
    genuine "nothing can trade" fault.
"""
from __future__ import annotations

import ast
import dataclasses
import glob
import io
import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import main  # noqa: E402
from configs.config_resolver import (  # noqa: E402
    validate_loaded_config,
    validate_toplevel_keys,
)

LIVE = REPO / "configs" / "live_high_risk.json"


def _live_data() -> dict:
    return json.loads(LIVE.read_text(encoding="utf-8"))


def _cfg_from(data: dict):
    import tempfile
    p = Path(tempfile.mkdtemp()) / "cfg.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return main.ProductionConfig.from_file(p)


def _engine_profiles():
    skip = {"feature_manifest.json", "split_manifest.json",
            "strategy_v5_1_decisions.json", "sota_config.json"}
    out = []
    for f in sorted(glob.glob(str(REPO / "configs" / "*.json"))):
        if os.path.basename(f) in skip:
            continue
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if isinstance(d, dict):
            out.append((os.path.basename(f), d))
    return out


# =============================================================================
# A. The hold-cost trap: funding must not be gated on the leverage flag
# =============================================================================

class TestFundingIsNotGatedOnLeverage:

    def _leverage_if_nodes(self):
        """EVERY `if ... regime_leverage_enabled` block, not the first.

        The falsification probe caught this: with the funding call re-gated
        there are TWO such blocks, and a version of this helper that returned
        only the FIRST inspected the innocent one and stayed GREEN while the
        defect was present. Anchor uniqueness is never something to assume —
        assert it or iterate (P238).
        """
        tree = ast.parse(io.open(REPO / "main.py", encoding="utf-8").read())
        found = [n for n in ast.walk(tree)
                 if isinstance(n, ast.If)
                 and "regime_leverage_enabled" in ast.dump(n.test)]
        assert found, "the regime_leverage_enabled block vanished"
        return found

    def test_update_funding_rate_is_outside_the_leverage_block(self):
        """THE BUG. `update_funding_rate` has exactly one production caller;
        while it lived inside this block, turning the leverage flag off also
        silently stopped charging the venue's real carry."""
        blk = "".join(ast.dump(n) for n in self._leverage_if_nodes())
        assert "update_funding_rate" not in blk, (
            "update_funding_rate is INSIDE the regime_leverage_enabled block "
            "again — P291b's venue-true hold cost is transitively gated on a "
            "flag named for leverage, and disabling it drops the hold term "
            "to 0.0 (the undercharge direction, P167)")

    def test_the_call_still_exists_and_is_asset_tagged(self):
        """Anti-vacuity: 'not inside the block' must not be satisfied by the
        call having been deleted. P291 requires the asset= tag."""
        src = io.open(REPO / "main.py", encoding="utf-8").read()
        assert src.count("FRICTION.update_funding_rate(") == 1
        i = src.index("FRICTION.update_funding_rate(")
        assert "asset=asset" in src[i:i + 200]

    def test_funding_runs_after_the_leverage_block(self):
        """ORDER IS LOAD-BEARING: update_funding_rate takes max(rollover,
        funding) while update_for_leverage ASSIGNS rollover, so funding first
        would be clobbered."""
        src = io.open(REPO / "main.py", encoding="utf-8").read()
        assert src.index("FRICTION.update_for_leverage(") < \
            src.index("FRICTION.update_funding_rate(")

    def test_why_it_mattered_the_hold_term_collapses_without_funding(self):
        """The CONSEQUENCE, pinned on the real object: with neither call made,
        _margin_cost_bps is 0.0 — no borrow fee AND no carry."""
        from defense.constitution import FrictionComponents
        f = FrictionComponents()
        f.venue_true_hold_enabled = True
        assert f._margin_cost_bps == 0.0, (
            "the defaults must be the zero state this test reasons about")
        f.update_funding_rate(0.000400, asset="SOL")
        f._venue_true_hold_asset = "SOL"
        assert f._margin_cost_bps > 0.0, (
            "with funding wired the venue-true branch must charge real carry")


# =============================================================================
# B. The config surface: declared + read + now PARSED
# =============================================================================

class TestLateConfigOverrides:

    def test_the_roster_only_names_real_fields(self):
        names = {f.name for f in dataclasses.fields(main.ProductionConfig)}
        bad = sorted(k for k in main._LATE_CONFIG_KEYS if k not in names)
        assert not bad, f"roster names non-existent fields: {bad}"

    def test_every_declared_and_read_field_is_now_reachable(self):
        """ANTI-ROT, both directions. Recomputes the original finding: a field
        that is declared, read somewhere in the tree, and assigned nowhere in
        from_file must either be in the roster or in the documented exclusion
        list. A new one added later fails here instead of becoming the next
        silently-unsettable flag."""
        src = io.open(REPO / "main.py", encoding="utf-8").read()
        lines = src.split("\n")
        start = next(i for i, l in enumerate(lines) if "def from_file" in l)
        body = []
        for l in lines[start + 1:]:
            if re.match(r'^\s{0,4}(def |class )', l) and l.strip():
                break
            body.append(l)
        body = "\n".join(body)
        assigned = set(re.findall(r'^\s*(\w+)\s*=', body, re.M))
        declared = set(re.findall(r'^\s{4}(\w+)\s*:\s*[^=\n]+=', src, re.M))
        reads = set()
        for p in REPO.rglob("*.py"):
            sp = str(p).replace("\\", "/")
            if any(x in sp for x in ("/archive/", "/legacy/", "/venv/",
                                     "/tests/", "/training/")):
                continue
            t = p.read_text(encoding="utf-8", errors="replace")
            reads |= set(re.findall(r'(?:self\.)?config\.(\w+)', t))
            reads |= set(re.findall(
                r'getattr\(\s*(?:self\.)?config\s*,\s*[\'"](\w+)[\'"]', t))
        # Documented exclusions, each with its reason in _LATE_CONFIG_KEYS.
        excluded = {"mode", "simulate_disconnect_tick"}
        gap = sorted(f for f in reads
                     if f in declared and f not in assigned
                     and f not in main._LATE_CONFIG_KEYS and f not in excluded)
        assert not gap, (
            f"these config fields are declared and read but can be set from "
            f"NEITHER from_file NOR the P313 roster, so an operator writing "
            f"them gets silence: {gap}")

    def test_the_deliberate_exclusions_stay_excluded(self):
        """`mode` is documentation-only (the CLI overwrites it, P227) and
        every profile already sets it — parsing it would be a live behaviour
        change. `simulate_disconnect_tick` is a test hook; making it
        config-settable hands a live profile an exchange-disconnect
        simulator (P141)."""
        for k in ("mode", "simulate_disconnect_tick"):
            assert k not in main._LATE_CONFIG_KEYS

    @pytest.mark.parametrize("field,payload,expected", [
        ("profit_max_enabled", {"profit_max": {"enabled": False}}, False),
        ("profit_max_loss_streak_enabled",
         {"profit_max": {"loss_streak_enabled": False}}, False),
        ("profit_max_extreme_funding_threshold",
         {"profit_max": {"extreme_funding_threshold": 0.25}}, 0.25),
        ("thesis_budget_enabled", {"thesis_budget": {"enabled": False}}, False),
        ("regime_leverage_enabled", {"regime_leverage_enabled": False}, False),
        ("regime_power_enabled", {"regime_power_enabled": False}, False),
        ("regime_aggression_enabled", {"regime_aggression_enabled": False}, False),
        ("lead_lag_amplifier_enabled", {"lead_lag_amplifier_enabled": False}, False),
        ("lead_lag_min_confidence_floor",
         {"lead_lag_min_confidence_floor": 0.42}, 0.42),
    ])
    def test_the_operator_can_actually_set_it(self, field, payload, expected):
        """THE BUG: every one of these returned its code default before."""
        d = _live_data()
        for k, v in payload.items():
            d[k] = dict(d.get(k, {}), **v) if isinstance(v, dict) else v
        assert getattr(_cfg_from(d), field) == expected

    def test_absent_keys_are_byte_identical_to_before(self):
        """The live profile sets NONE of these, so this change must be a
        no-op for it. Compares every field, not a sample."""
        base = main.ProductionConfig.from_file(LIVE)
        again = _cfg_from(_live_data())
        diffs = [f.name for f in dataclasses.fields(base)
                 if repr(getattr(base, f.name)) != repr(getattr(again, f.name))]
        assert not diffs, f"parsing changed untouched fields: {diffs}"

    def test_a_malformed_value_keeps_the_default_and_does_not_raise(self):
        """A bad config line must not take the boot down (P85) and must not
        silently install a wrong-typed value either."""
        d = _live_data()
        d["regime_power_multipliers"] = "not-a-dict"
        d["lead_lag_min_confidence_floor"] = "not-a-float"
        cfg = _cfg_from(d)
        assert isinstance(cfg.regime_power_multipliers, dict)
        assert cfg.regime_power_multipliers  # the rich default survived
        assert cfg.lead_lag_min_confidence_floor == 0.10

    def test_bools_are_not_coerced_through_int(self):
        """bool is an int subclass; the coercion order must check bool first
        or a bool field would take an int."""
        d = _live_data()
        d["profit_max"] = {"enabled": 0}
        cfg = _cfg_from(d)
        assert cfg.profit_max_enabled is False


# =============================================================================
# C. The schema: a top-level key can no longer be silently ignored
# =============================================================================

class TestTopLevelSchema:

    def _known(self):
        return {f.name for f in dataclasses.fields(main.ProductionConfig)}

    def test_a_typo_is_flagged(self):
        d = _live_data()
        d["profit_max_enabledd"] = False
        errs = validate_toplevel_keys(d, self._known())
        assert any("profit_max_enabledd" in e for e in errs)

    def test_underscore_keys_stay_exempt(self):
        assert validate_toplevel_keys({"_note": "doc"}, self._known()) == []

    @pytest.mark.parametrize("name,data", _engine_profiles())
    def test_no_real_profile_cries_wolf(self, name, data):
        """P303: a schema that cannot tell a real key from a typo makes the
        operator ignore the one line that would flag a REAL typo."""
        errs = validate_toplevel_keys(data, self._known())
        assert not errs, f"{name} produced false warnings: {errs}"

    def test_omitting_the_argument_preserves_the_old_behaviour(self):
        """Additive by construction — existing callers are unaffected."""
        d = _live_data()
        d["totally_unknown_key"] = 1
        assert not any("totally_unknown_key" in e
                       for e in validate_loaded_config(d))

    def test_from_file_actually_passes_the_known_set(self, caplog):
        """WIRING. validate_toplevel_keys working in isolation proves nothing
        if from_file never hands it the field names — that argument is what
        turns the check on, and omitting it is byte-identical to the silent
        behaviour this entry exists to end."""
        import logging
        d = _live_data()
        d["profit_max_enabledd"] = False
        with caplog.at_level(logging.WARNING):
            _cfg_from(d)
        assert any("profit_max_enabledd" in r.getMessage()
                   and "CONFIG_SCHEMA" in r.getMessage()
                   for r in caplog.records), (
            "loading a profile with an unknown top-level key produced no "
            "[CONFIG_SCHEMA] warning — from_file is not passing known_toplevel")

    def test_the_known_set_comes_from_the_dataclass_not_a_restated_list(self):
        """P310's rule. If the schema module restated the field names they
        would drift the first time a field was added."""
        src = io.open(REPO / "configs" / "config_resolver.py",
                      encoding="utf-8").read()
        # Scoped to the frozenset LITERAL. A wider window swallows the
        # docstring below it, which legitimately NAMES profit_max_enabled
        # while explaining the bug — a guard that matches its own
        # explanation is the P177 trap (caught here by this test failing).
        i = src.index("_TOPLEVEL_ALIAS_KEYS = frozenset({")
        blk = src[i:src.index("})", i)]
        for f in ("profit_max_enabled", "regime_leverage_enabled",
                  "thesis_budget_enabled"):
            assert f not in blk, (
                f"{f} is restated in the schema module; it must come from "
                f"dataclasses.fields(ProductionConfig)")


# =============================================================================
# D. HEALTH_T1: key on the intended position, not on a label
# =============================================================================

class _Intent:
    def __init__(self, direction):
        self.direction = direction


class TestT1KeysOnIntendedPosition:

    def _check(self, direction, edge, strategy):
        from core.health_validator import PerTickInvariantChecker
        return PerTickInvariantChecker()._t1_alpha_estimate_nonzero(
            "ETH", _Intent(direction), {"signal_edge_bps": edge},
            {"primary_strategy": strategy})

    def test_the_live_case_a_flat_book_no_longer_warns(self):
        """ETH/SOL live: the certified trend-only book is FLAT outside bull,
        so edge=0 is correct — it warned every tick."""
        assert self._check(0.0, 0.0, "regimebook").status == "PASS"

    def test_a_real_fault_still_warns(self):
        """THE DISCRIMINATING POWER. A position IS intended and there is no
        edge to pay for it — the P155 'nothing can trade' condition. If this
        ever passes, the fix silenced the check instead of sharpening it."""
        assert self._check(1.0, 0.0, "trend_following").status == "WARN"

    def test_a_healthy_directional_tick_passes(self):
        assert self._check(1.0, 30.0, "regimebook").status == "PASS"

    def test_hold_is_still_short_circuited(self):
        assert self._check(0.0, 0.0, "hold").status == "PASS"

    def test_an_unknown_direction_is_not_read_as_flat(self):
        """ABSENCE IS NOT FLAT (P2). A missing intent says nothing about
        whether a position is intended; reading it as 0.0 would PASS the
        check whenever the direction is merely unknown — silencing T1
        exactly when there is least information. The pre-existing
        test_warn_when_momentum_zero_edge (intent=None) caught this."""
        from core.health_validator import PerTickInvariantChecker
        c = PerTickInvariantChecker()
        for intent in (None, object()):   # no intent / no .direction
            got = c._t1_alpha_estimate_nonzero(
                "BTC", intent, {"signal_edge_bps": 0},
                {"primary_strategy": "momentum"})
            assert got.status == "WARN", (
                "an unknown direction was treated as flat, which silences T1")

    def test_the_seat_reports_itself_in_both_dicts(self):
        """The misattribution: the seat overwrote direction/edge and left
        primary_strategy reading 'trend_following'. Both dicts, per P170."""
        src = io.open(REPO / "main.py", encoding="utf-8").read()
        # Anchor on the SEAT-TAKEN log line specifically: the first
        # "[REGIMEBOOK-SEAT]" in the file belongs to the stale/no-target
        # warning branch, which is a different block entirely.
        anchor = "book holds the seat"
        assert src.count(anchor) == 1, "seat log anchor is no longer unique"
        i = src.index(anchor)
        blk = src[max(0, i - 4000):i]
        assert 'market_data["primary_strategy"] = "regimebook"' in blk
        assert 'agent_signals["primary_strategy"] = "regimebook"' in blk
