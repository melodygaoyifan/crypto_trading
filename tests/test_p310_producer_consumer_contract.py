"""
[P310] The mechanism for the class, not a fourth fix for an instance.

Three times now a consumer's belief about a producer diverged silently:

  P264   the ledger parser read `ts` as ISO-only; regimebook writes an epoch
         float, so the whole family was invisible to the promotion gate.
  P295d  seat_check read `agents.<name>.<h>`; agent_ic_review emits
         `agents.<name>.horizons.<h>`, so every candidate parsed to n=0 and
         the tool recommended flattening a live book from nothing.
  P309   the shadow allowlists were keyed on the LEDGER-FILE PREFIX while the
         scorer groups by the record's `strategy` FIELD, so two families were
         never pooled and an archive section never rendered.

All three passed their own tests, because in each case the fixture and the
consumer were written by the same hand and agreed with each other. A test that
constructs BOTH sides cannot catch a naming or shape mismatch — only the
producer can.

So this file never writes a name or a record shape of its own. It asks the
PRODUCERS what they emit, and holds the CONSUMERS to it:

  1. NAME CONTRACT (kills P309's class). Every producer declares
     `SHADOW_STRATEGY_NAMES`. Every consumer classification must name a real
     producer, and every producer name must be classified — in both
     directions, so neither a typo nor a new unclassified family can hide.
  2. SHAPE CONTRACT (kills P264's class). A record built by a producer's own
     writer must survive the consumer's loader, including its timestamp form.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Every module that writes a shadow-ledger record. Adding a producer without
# adding it here is caught by test_every_ledger_writer_is_registered below.
PRODUCER_MODULES = (
    "defense.regime_book_shadow",
    "defense.etf_flow_shadow",
    "defense.mlp_shadow",
    "defense.enhancement_shadows",
    "defense.trend_rule_shadow",
    "defense.sentiment_variant_shadow",
    "defense.strategy_shadow_v5_1",
    "strategies.derivatives_flow_v1",
)


def producer_names() -> dict:
    """{strategy_name: module} straight from the producers."""
    out = {}
    for m in PRODUCER_MODULES:
        mod = importlib.import_module(m)
        declared = getattr(mod, "SHADOW_STRATEGY_NAMES", None)
        assert declared, (
            f"{m} writes shadow records but declares no "
            f"SHADOW_STRATEGY_NAMES — consumers would have to restate its "
            f"names, which is exactly how P309 happened")
        for n in declared:
            out[n] = m
    return out


def consumer_classes() -> dict:
    from analytics.shadow_ic.compute_shadow_ic import (
        ARCHIVED_FAMILIES, PER_ASSET_FAMILIES, POOLABLE_FAMILIES)
    return {"poolable": set(POOLABLE_FAMILIES),
            "archived": set(ARCHIVED_FAMILIES),
            "per_asset": set(PER_ASSET_FAMILIES)}


# =============================================================================
# 1. The NAME contract — both directions
# =============================================================================

class TestNameContract:

    def test_the_registry_is_not_empty(self):
        """Anti-vacuity (P174): a contract over an empty set passes always."""
        names = producer_names()
        assert len(names) >= 25, f"only {len(names)} names collected"
        for known in ("regimebook", "ma_filtered", "liquidation_squeeze",
                      "donchian", "etfflow"):
            assert known in names, known

    def test_every_consumer_name_is_emitted_by_a_real_producer(self):
        """THE P309 BUG. `ma_filter` vs `ma_filtered` was invisible because
        nothing compared the consumer's spelling to the producer's."""
        names = set(producer_names())
        for kind, members in consumer_classes().items():
            orphans = sorted(members - names)
            assert not orphans, (
                f"{kind}: {orphans} are not emitted by any producer. Either "
                f"the name is wrong (a ledger-file PREFIX rather than the "
                f"record's `strategy` field?) or the producer is gone.")

    def test_every_producer_name_is_classified_exactly_once(self):
        """A new family that nobody classified is the other half of the gap:
        it silently defaults to per-asset scoring and to being in the
        promotion table, with no one having decided either."""
        names = producer_names()
        cls = consumer_classes()
        for n, mod in sorted(names.items()):
            hits = [k for k, v in cls.items() if n in v]
            assert hits, (
                f"{n!r} (from {mod}) is emitted but classified nowhere — "
                f"add it to POOLABLE_FAMILIES, ARCHIVED_FAMILIES or "
                f"PER_ASSET_FAMILIES so the decision is recorded")
            assert len(hits) == 1, f"{n!r} classified {hits} — pick one"

    # A shadow-ledger RECORD is a dict literal carrying all of these. Matching
    # on `"strategy"` alone false-positived on a logging `extra={"strategy":
    # ...}` in p0_safety_integrator — and a scanner that cries wolf is a
    # scanner someone disables, so the detector is keyed on the actual record
    # shape rather than on one field name.
    _RECORD_KEYS = {"strategy", "asset", "direction"}

    def _ledger_writers(self):
        import ast
        found = set()
        for sub in ("defense", "strategies", "analytics"):
            for f in (REPO / sub).rglob("*.py"):
                if "archive" in f.parts or f.name.startswith("test_"):
                    continue
                try:
                    src = f.read_text(encoding="utf-8-sig")
                    tree = ast.parse(src)
                except Exception:
                    continue
                # Scoped to the SHADOW-LEDGER directory. exit_alpha_tracker
                # and signal_quality_tracker build dicts with the same three
                # key names but write to data/*.jsonl, a different record
                # family entirely — flagging them would be crying wolf.
                if "strategy_shadow" not in src:
                    continue
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Dict):
                        continue
                    keys = {k.value for k in node.keys
                            if isinstance(k, ast.Constant)
                            and isinstance(k.value, str)}
                    if self._RECORD_KEYS <= keys:
                        found.add(".".join(
                            f.relative_to(REPO).with_suffix("").parts))
                        break
        return found

    # [P313] Every module that ACTUALLY builds a ledger record inline. This is
    # deliberately not `PRODUCER_MODULES`: that tuple serves two roles, and
    # `strategies.derivatives_flow_v1` is in it for its NAME DECLARATION only
    # — it returns FlowSignal objects and the writing is done by
    # strategy_shadow_v5_1's harness. So the writer roster is its own list.
    _KNOWN_WRITERS = (
        "defense.regime_book_shadow",
        "defense.etf_flow_shadow",
        "defense.mlp_shadow",
        "defense.enhancement_shadows",
        "defense.trend_rule_shadow",
        "defense.sentiment_variant_shadow",
        "defense.strategy_shadow_v5_1",
    )

    def test_the_detector_actually_finds_the_known_writers(self):
        """Anti-vacuity: if the shape detector matches nothing, the rot check
        below passes forever (P174).

        [P313] Pinned against EVERY known writer, not two of them. The old
        version asserted only regime_book_shadow and mlp_shadow, so the
        detector could quietly stop matching the other five — its three
        narrowing preconditions (directory list, the literal `strategy_shadow`
        string, an inline 3-key dict) each shrink coverage, and a guard that
        samples two modules cannot notice that shrinkage. Coverage loss in a
        rot detector is exactly the failure it exists to prevent.
        """
        found = self._ledger_writers()
        missed = [m for m in self._KNOWN_WRITERS if m not in found]
        assert not missed, (
            f"the record-shape detector missed {missed}; it would then "
            f"miss a NEW producer too")

    def test_every_ledger_writer_is_registered_here(self):
        """The registry must not rot: a module that builds a shadow record has
        to appear in PRODUCER_MODULES, or its names go unchecked."""
        missed = sorted(m for m in self._ledger_writers()
                        if m not in PRODUCER_MODULES)
        assert not missed, (
            f"these build a shadow-ledger record but are not in "
            f"PRODUCER_MODULES, so their strategy names are unchecked: "
            f"{missed}")


# =============================================================================
# 2. The SHAPE contract — a producer record must survive the consumer's loader
# =============================================================================

class TestShapeContract:

    def test_a_producer_written_record_parses(self, tmp_path):
        """THE P264 BUG, generalised. regimebook writes `ts` as an epoch
        FLOAT; the loader parsed ISO strings only, swallowed the
        AttributeError per file, and the family was invisible to the gate.
        Build the record the way the producer does and feed it to the real
        loader."""
        import json
        import time
        from analytics.shadow_ic.compute_shadow_ic import load_shadow_ledgers

        # The producer's own timestamp convention (regime_book_shadow._record).
        rec = {
            "ts": time.time() - 3600,          # EPOCH FLOAT, not ISO
            "iso": "2026-08-18T00:00:00+00:00",
            "strategy": "regimebook", "asset": "BTC",
            "direction": 1.0, "confidence": 1.0,
        }
        d = tmp_path / "led"
        d.mkdir()
        (d / "regimebook_BTC.jsonl").write_text(
            json.dumps(rec) + "\n", encoding="utf-8")

        got = load_shadow_ledgers(d, prefixes=("regimebook",))
        assert got, "an epoch-float `ts` must parse — P264"
        assert got[0].get("_parsed_ts") is not None

    def test_both_timestamp_forms_are_accepted(self, tmp_path):
        """Some producers write ISO. The loader must take either, or fixing
        one shape breaks the other."""
        import json
        import time
        from datetime import datetime, timezone
        from analytics.shadow_ic.compute_shadow_ic import load_shadow_ledgers

        d = tmp_path / "led"
        d.mkdir()
        iso = datetime.now(timezone.utc).isoformat()
        with open(d / "regimebook_BTC.jsonl", "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time() - 60, "strategy": "regimebook",
                                 "asset": "BTC", "direction": 1.0,
                                 "confidence": 1.0}) + "\n")
            fh.write(json.dumps({"ts": iso, "strategy": "regimebook",
                                 "asset": "BTC", "direction": -1.0,
                                 "confidence": 1.0}) + "\n")
        got = load_shadow_ledgers(d, prefixes=("regimebook",))
        assert len(got) == 2, f"both forms must parse, got {len(got)}"

    def test_the_scorer_groups_on_the_field_the_producer_writes(self, tmp_path):
        """The premise P309 violated, pinned as a premise rather than left
        implicit in a comment: grouping is by the record's `strategy` value,
        NOT the filename."""
        import time
        from analytics.shadow_ic.compute_shadow_ic import compute_per_strategy_ic
        recs = [{"strategy": "ma_filtered", "asset": "BTC", "direction": 1.0,
                 "confidence": 1.0, "_parsed_ts": None}]
        out = compute_per_strategy_ic(recs, horizons_bars=(4,))
        assert set(out) == {("ma_filtered", "BTC")}, (
            "if this ever groups by something else, every allowlist in the "
            "consumer is keyed on the wrong thing")


# =============================================================================
# 3. The SECOND boundary: agent_ic_review -> seat_check (the P295d class)
# =============================================================================

class TestSeatCheckReadsWhatAgentIcReviewWrites:
    """[P312] P310 covered the shadow-ledger boundary and explicitly left this
    one out, pinned only by a hand-copied literal fixture — which is the very
    pattern that let P295d through (fixture and reader written by the same
    hand, agreeing with each other).

    So the report here is built through the PRODUCER's own shape functions.
    If the nesting ever moves, this report moves with it and the reader fails,
    instead of the reader silently parsing every agent to n=0 and the seat
    controller recommending that a live book go flat.
    """

    def _seat_check(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "seat_check_p311", REPO / "scripts" / "seat_check.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _producer_report(self):
        from analytics.ic.agent_ic_review import build_agent_entry, build_report
        rep = build_report(30, {1: 64.05, 4: 117.69},
                           "2026-08-19T06:10:00+00:00")
        for agent, cells in (
            ("quant", {1: {"n": 713, "ic": -0.0326, "t": -0.87},
                       4: {"n": 704, "ic": -0.086, "t": -1.14}}),
            ("whale", {1: {"n": 226, "ic": -0.0698, "t": -1.05},
                       4: {"n": 225, "ic": -0.0651, "t": -0.48}}),
        ):
            rep["agents"][agent] = build_agent_entry(cells, "HOLD")
        return rep

    def test_the_reader_parses_a_producer_built_report(self, tmp_path):
        """THE P295d BUG. A hand-written fixture cannot catch the nesting."""
        import json
        p = tmp_path / "agent_ic.json"
        p.write_text(json.dumps(self._producer_report()), encoding="utf-8")

        parsed = self._seat_check()._from_ic_report(p)
        assert parsed, "the reader must see the producer's agents"
        assert parsed["quant"][4] == (-0.086, -1.14, 704), (
            "n=0 here is the defect that made the seat controller recommend "
            "flattening a live book from nothing")
        assert parsed["whale"][1] == (-0.0698, -1.05, 226)

    def test_the_reader_never_silently_returns_zero_samples(self, tmp_path):
        """A shape mismatch shows up as n=0, which the seat controller treats
        as 'below the evidence floor' — indistinguishable from a genuinely
        thin window. That collapse is what made the bug invisible."""
        import json
        p = tmp_path / "agent_ic.json"
        p.write_text(json.dumps(self._producer_report()), encoding="utf-8")
        parsed = self._seat_check()._from_ic_report(p)
        for agent, series in parsed.items():
            for h, (_ic, _t, n) in series.items():
                assert n > 0, f"{agent}@{h} parsed to n=0 from a real report"

    def test_the_producer_actually_uses_its_own_shape_functions(self):
        """A seam nothing calls is decorative (P170) — the report would drift
        away from the contract the consumer is tested against."""
        from tests._source_scan import code_only
        src = code_only(REPO / "analytics" / "ic" / "agent_ic_review.py")
        assert "build_report(" in src and "build_agent_entry(" in src
        assert '{"horizons": rows' not in src, (
            "the nesting must exist in exactly one place")

    def test_an_end_to_end_seat_verdict_comes_out_of_a_producer_report(
            self, tmp_path):
        """The whole chain, since P295d's failure was only visible end to end:
        producer shape -> reader -> decide_seat. These are the real 30d
        numbers; both candidates are negative on both horizons, so flat wins
        BY COMPARISON (exit 3) rather than by refusal (exit 2)."""
        import json
        import subprocess
        p = tmp_path / "agent_ic.json"
        p.write_text(json.dumps(self._producer_report()), encoding="utf-8")
        r = subprocess.run(
            [sys.executable, "-X", "utf8", str(REPO / "scripts" / "seat_check.py"),
             "--ic-report", str(p), "--incumbent", "whale"],
            capture_output=True, text=True, encoding="utf-8", cwd=str(REPO))
        assert r.returncode == 3, (
            f"expected a measured verdict, got rc={r.returncode} — rc=2 means "
            f"the reader saw nothing\n{r.stdout}\n{r.stderr}")
        assert "REFUSING" not in r.stderr
