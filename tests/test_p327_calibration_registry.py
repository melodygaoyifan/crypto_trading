"""[P327] Guards for the calibration contract.

The load-bearing ones are the TWO DIRECTIONS (P310): every registered entry
must resolve to a live symbol, AND every module that carries the measurement
stamps must be registered. One direction alone leaves the hole the registry
exists to close — a registry that only checks itself grows stale silently, and
a convention that can be used without registering is not a contract.
"""
from __future__ import annotations

import ast
import datetime as dt
import io
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.calibration_registry import (  # noqa: E402
    REGISTRY,
    STAMP_MEASURED_BY,
    STAMP_MEASURED_ON,
    Calibration,
    by_symbol,
    resolve,
    stale_entries,
)

CHECK = REPO / "scripts" / "calibration_check.py"

# Directories a measured constant could plausibly live in. Scanned for the
# stamps, so the convention cannot be used outside the registry's reach.
SCAN_DIRS = ("core", "defense", "risk", "signals", "execution", "exchange")


class TestDirectionOne_EveryEntryResolves:
    """A registry entry describing a symbol that no longer exists is worse
    than no entry: it reads as coverage (P174)."""

    @pytest.mark.parametrize("cal", REGISTRY, ids=lambda c: c.symbol)
    def test_the_symbol_is_live(self, cal):
        assert resolve(cal) is not None

    @pytest.mark.parametrize("cal", REGISTRY, ids=lambda c: c.symbol)
    def test_the_value_is_not_empty(self, cal):
        """An empty dict resolves fine and measures nothing."""
        v = resolve(cal)
        if isinstance(v, (dict, list, tuple, set, frozenset)):
            assert len(v) > 0

    def test_the_registry_is_not_empty(self):
        """Anti-vacuity: an empty registry passes every other test here."""
        assert len(REGISTRY) >= 3


class TestDirectionTwo_EveryStampedModuleIsRegistered:
    """The half that makes it a contract rather than a list. Without it, the
    next author stamps a module, believes they have declared provenance, and
    the check never looks at it."""

    def _stamped_modules(self):
        out = []
        for d in SCAN_DIRS:
            for p in sorted((REPO / d).rglob("*.py")):
                try:
                    src = io.open(p, encoding="utf-8").read()
                except OSError:
                    continue
                # the stamp must be a real module-level assignment, not a
                # mention in a comment or docstring
                try:
                    tree = ast.parse(src)
                except SyntaxError:
                    continue
                names = set()
                for node in tree.body:
                    if isinstance(node, ast.Assign):
                        for t in node.targets:
                            if isinstance(t, ast.Name):
                                names.add(t.id)
                if STAMP_MEASURED_ON in names:
                    rel = p.relative_to(REPO).as_posix()
                    out.append(rel[:-3].replace("/", "."))
        return out

    def test_the_scan_finds_the_known_stamped_modules(self):
        """Anti-vacuity (P174): if this ever returns nothing, the direction-two
        guard below passes forever."""
        mods = self._stamped_modules()
        assert "core.seat_alpha" in mods
        assert "core.cde_fees" in mods
        assert "defense.constitution" in mods

    def test_every_stamped_module_has_a_registry_entry(self):
        registered = {c.module for c in REGISTRY}
        for mod in self._stamped_modules():
            assert mod in registered, (
                f"{mod} carries {STAMP_MEASURED_ON} but is not in the "
                f"calibration registry. Stamping a module is a claim that it "
                f"holds a measurement; register it, or the staleness check "
                f"never looks at it.")

    def test_every_registered_module_carries_the_stamps(self):
        for cal in REGISTRY:
            mod = sys.modules.get(cal.module) or __import__(
                cal.module, fromlist=["x"])
            for stamp in (STAMP_MEASURED_ON, STAMP_MEASURED_BY):
                assert hasattr(mod, stamp), f"{cal.module} lacks {stamp}"

    def test_the_stamp_date_agrees_with_the_registry(self):
        """Two places record the date; they must not drift (P172)."""
        for cal in REGISTRY:
            mod = __import__(cal.module, fromlist=["x"])
            assert getattr(mod, STAMP_MEASURED_ON) == cal.measured_on, (
                f"{cal.module}: module stamp "
                f"{getattr(mod, STAMP_MEASURED_ON)} != registry "
                f"{cal.measured_on}")


class TestTheFieldsAreUsable:
    """Each field exists because a specific incident happened without it."""

    @pytest.mark.parametrize("cal", REGISTRY, ids=lambda c: c.symbol)
    def test_the_producer_is_a_runnable_command_naming_a_real_file(self, cal):
        """P326: a module NAME is not a derivation. The producer must be a
        command, and the file it invokes must exist."""
        assert cal.producer.startswith("python"), cal.producer
        m = re.search(r"(\S+\.py)", cal.producer)
        assert m, f"no script path in producer: {cal.producer}"
        assert (REPO / m.group(1)).exists(), m.group(1)

    @pytest.mark.parametrize("cal", REGISTRY, ids=lambda c: c.symbol)
    def test_the_source_names_the_data(self, cal):
        """P316: a constant calibrated from the wrong source is invisible
        unless the source is written down."""
        assert len(cal.source) > 25

    @pytest.mark.parametrize("cal", REGISTRY, ids=lambda c: c.symbol)
    def test_the_staleness_horizon_carries_its_reason(self, cal):
        """A horizon without a reason is a number someone will 'fix' by
        raising it the first time the check goes red."""
        assert 1 <= cal.staleness_days <= 730
        assert len(cal.staleness_reason) > 40

    @pytest.mark.parametrize("cal", REGISTRY, ids=lambda c: c.symbol)
    def test_the_revision_rule_states_a_direction_or_a_procedure(self, cal):
        """P167: several of these may only move one way on evidence, and a
        refresh must not be able to smuggle in a loosening."""
        assert len(cal.revision_rule) > 40

    @pytest.mark.parametrize("cal", REGISTRY, ids=lambda c: c.symbol)
    def test_the_date_parses_and_is_not_in_the_future(self, cal):
        d = dt.date.fromisoformat(cal.measured_on)
        assert d <= dt.date.today() + dt.timedelta(days=1)


class TestStaleness:

    def test_nothing_is_stale_today(self):
        """If this fails, a calibration aged past its horizon — run
        scripts/calibration_check.py and read the revision rule. Do NOT raise
        the horizon to make it green."""
        assert stale_entries() == []

    def test_staleness_is_computed_from_the_date_not_hardcoded(self):
        cal = REGISTRY[0]
        far = dt.date.fromisoformat(cal.measured_on) + dt.timedelta(
            days=cal.staleness_days + 1)
        assert cal.is_stale(far)
        near = dt.date.fromisoformat(cal.measured_on) + dt.timedelta(
            days=cal.staleness_days)
        assert not cal.is_stale(near)

    def test_by_symbol_finds_and_misses_correctly(self):
        assert by_symbol(REGISTRY[0].symbol) is REGISTRY[0]
        assert by_symbol("core.nope.NOPE") is None


class TestTheCheckScript:
    """Exit codes must be distinguishable, or 'I could not read it' reads as
    'it is fine' (P159/P199/P213)."""

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-X", "utf8", str(CHECK), *args],
            capture_output=True, text=True, encoding="utf-8",
            cwd=str(REPO), timeout=300)

    def test_exit_zero_when_fresh(self):
        r = self._run()
        assert r.returncode == 0, r.stdout[-600:] + r.stderr[-600:]

    def test_exit_three_when_stale(self):
        far = (max(dt.date.fromisoformat(c.measured_on) for c in REGISTRY)
               + dt.timedelta(days=max(c.staleness_days for c in REGISTRY) + 1))
        r = self._run("--today", far.isoformat())
        assert r.returncode == 3, r.stdout[-600:]
        assert "producer" in r.stdout

    def test_it_never_edits_a_constant(self):
        src = io.open(CHECK, encoding="utf-8").read()
        for bad in ("open(", "write", "setattr"):
            assert bad not in src.split('"""', 2)[-1], bad


class TestTheSpreadHoistIsBehaviourPreserving:
    """[P327] CDE_SPREAD_BPS was an inline literal inside __post_init__ with no
    provenance at all. Hoisting it must not change a single number, and the
    instance must not alias the module constant — the instance is mutated
    per-venue at runtime (set_spread_venue) and would otherwise write back into
    the measurement itself."""

    def test_the_instance_default_equals_the_measured_table(self):
        from defense.constitution import (CDE_SPREAD_BPS_MEASURED,
                                          FrictionComponents)
        assert FrictionComponents().CDE_SPREAD_BPS == CDE_SPREAD_BPS_MEASURED

    def test_the_values_are_unchanged_from_P289(self):
        from defense.constitution import CDE_SPREAD_BPS_MEASURED
        assert CDE_SPREAD_BPS_MEASURED == {"BTC": 2.0, "ETH": 5.5, "SOL": 4.0}

    def test_mutating_an_instance_cannot_corrupt_the_measurement(self):
        from defense.constitution import (CDE_SPREAD_BPS_MEASURED,
                                          FrictionComponents)
        f = FrictionComponents()
        f.CDE_SPREAD_BPS["BTC"] = 99.0
        assert CDE_SPREAD_BPS_MEASURED["BTC"] == 2.0
        assert FrictionComponents().CDE_SPREAD_BPS["BTC"] == 2.0


class TestTheRegistryIsNotACopy:
    """[P172/P310] It must describe where values live, never restate them —
    a second copy is the defect the whole contract exists to prevent."""

    def test_no_measured_value_is_duplicated_in_the_registry(self):
        src = io.open(REPO / "core" / "calibration_registry.py",
                      encoding="utf-8").read()
        code = src.split('"""', 2)[-1]
        for literal in ("0.603", "0.635", "5.5", "68.5", "24.1", "251.7"):
            assert literal not in code, (
                f"{literal} is restated in the registry; resolve() imports the "
                f"live value for exactly this reason")

    def test_resolve_reads_the_live_module(self):
        import core.cde_fees as m
        cal = by_symbol("core.cde_fees.CDE_FEE_PER_CONTRACT_USD")
        assert resolve(cal) is m.CDE_FEE_PER_CONTRACT_USD
