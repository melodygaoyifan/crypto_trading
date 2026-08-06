#!/usr/bin/env python3
"""
lint_orphan_signal_reads.py — reads of signal-dict keys that nobody writes (P171)
================================================================================

P170 was this bug: `integration_v36.decide()` read

    agent_signals.get("quant_data_quality", 1.0)

to decide whether the quant signal was fresh enough to fuse. The pipeline set
`quant_data_quality` faithfully on every path — into `market_data`. Nothing ever
copied it into `agent_signals`. So the read always missed, the default always
won, and the default was 1.0: healthy. P126's staleness guard, written
2026-04-27, never excluded a single stale signal.

`lint_signal_freshness.py` (P120) cannot catch this. It inventories agent_signals
**writes** and classifies their freshness guards — but the P170 key had no write
at all, so it was invisible to a writer census. This scanner is the complement:
it looks for **reads with no writer anywhere in the tree**.

Two severities:

  ORPHAN-HOT   — read with a *non-falsy* default. This is the dangerous shape:
                 the key never arrives, so the default is the value, every time,
                 and it asserts something positive (healthy / confident / large)
                 that nobody measured. Both P170 defaults were this
                 (`quant_data_quality` -> 1.0, `signal_edge_bps` -> 50.0).

  ORPHAN-COLD  — read with a falsy default (0, 0.0, "", False, None, [], {}).
                 Still drift, but absence degrades to "nothing", which is
                 usually the fail-safe direction.

A key is only reported when the tree contains NO static write of it. If a file
writes signal keys through a computed name (`d[k] = v`), that file's dict is
marked dynamic and its keys are exempted from the orphan check, because the
scanner cannot prove absence there. Those exemptions are reported in
`dynamic_write_sites` so the blind spot stays visible rather than silently
shrinking the finding count.

[P174] WHAT THIS SCANNER CANNOT PROVE — read before trusting a zero
-------------------------------------------------------------------
The first version of this file gated CI on `orphan_count: 0`. That number was
not a clean bill of health; it was arithmetically forced. `main.py` copies
signal keys in loops (`for k, v in ...: agent_signals[k] = v`), which marks
`agent_signals`, `market_data` and `position_state` permanently dynamic. Every
unmatched read of those three dicts is therefore downgraded to UNPROVABLE
before it can ever be counted. Measured at the time of writing: the ORPHAN
check adjudicated **0 of 458** unmatched reads. It could not have failed.

That is the same defect this scanner was built to find (P155-L5, P156, P158,
P159, P160, P164, P166, P169, P170, P171): a check that cannot fail is
indistinguishable from a check that passed. It shipped *inside the tool*, which
is the strongest possible argument that the class is not fixed by vigilance.

The soundness limit is real, not an implementation gap — a dynamic copy means
no key can be *proven* absent. So ORPHAN is kept, but it is no longer the
gate's headline and `orphan_adjudicable` is emitted beside it so a zero can
never again be read as coverage. What IS gated is what the scanner can prove
and what actually caught bugs: MISROUTED (P170's and P173's exact shape),
the size of the blind spot itself (`dynamic_site_count` — a rise means the tree
got less analyzable), and parse failures.

Usage:
    python -X utf8 tools/lint_orphan_signal_reads.py
    python -X utf8 tools/lint_orphan_signal_reads.py --json
    python -X utf8 tools/lint_orphan_signal_reads.py --paths main.py integration/
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# The dicts that carry cross-module signal state. These are the ones where a
# reader and a writer live in different files and can drift apart unnoticed.
TARGET_DICTS = frozenset({
    "agent_signals",
    "market_data",
    "system_state",
    "position_state",
})

# Directories that are not live code. archive/ is retired by definition, and a
# test asserting on a key it also constructs is not evidence of a producer.
EXCLUDED_DIRS = ("archive", "tests", ".git", "__pycache__", "node_modules",
                 "training_data", "venv", ".venv")

# Keys that are deliberately read-with-default as an optional input — the
# absence IS the contract, not a drift. Keep this list short and justified;
# every entry is a place the scanner has been told to stop looking.
KNOWN_OPTIONAL: Dict[str, str] = {
    # Private scratch keys are namespaced by convention and written ad hoc.
}


def _is_falsy_default(node: ast.AST) -> bool:
    """True when the default asserts nothing (0, "", False, None, [], {})."""
    if isinstance(node, ast.Constant):
        return not node.value if not isinstance(node.value, str) else node.value == ""
    if isinstance(node, (ast.List, ast.Dict, ast.Set, ast.Tuple)):
        return not (getattr(node, "elts", None) or getattr(node, "keys", None))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return False
    return False


def _default_repr(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<expr>"


def _is_null_coalesce(tgt: ast.AST, value: ast.AST) -> bool:
    """[P174] True for `x = x or {}` — a rebind that writes no keys.

    Eight sites in the tree use this idiom to normalise an optional argument.
    The first version treated it as an opaque alias and marked the whole dict
    dynamic, which is how `market_data` and `agent_signals` became unprovable
    tree-wide. It provably adds nothing: either the original dict survives
    unchanged, or an empty one is bound. Neither introduces a key.
    """
    if not (isinstance(tgt, ast.Name) and isinstance(value, ast.BoolOp)
            and isinstance(value.op, ast.Or) and len(value.values) == 2):
        return False
    lhs, rhs = value.values
    same_name = isinstance(lhs, ast.Name) and lhs.id == tgt.id
    empty_dict = isinstance(rhs, ast.Dict) and not rhs.keys
    return same_name and empty_dict


def _is_copy_of(key: str, value: ast.AST) -> bool:
    """[P174] True for `<signal dict>.get(key, ...)` — a copy, not a measurement.

    `agent_signals["whale_net_pressure"] = market_data.get("whale_net_pressure", 0.0)`
    moves a key between signal dicts. The original classifier counted that write
    as proof the key exists, which is backwards: a copy is downstream of a
    producer, and if there is no producer the copy faithfully propagates the
    default. Crediting it hid exactly the shape the scanner exists to find.

    A coercion wrapper is transparent here: `int(market_data.get("k", 0))` is
    still a copy. Missing that cost the first version its only two findings —
    `htf_trend_direction` is written exactly that way at main.py:9374.
    """
    while (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
           and value.func.id in ("int", "float", "bool", "str")
           and len(value.args) == 1):
        value = value.args[0]
    return (isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "get"
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id in TARGET_DICTS
            and bool(value.args)
            and isinstance(value.args[0], ast.Constant)
            and value.args[0].value == key)


# [P176] Modules whose returned dicts ARE a named signal dict downstream.
#
# This is a DECLARED architectural fact, not something the scanner infers, and
# it is written as a map so it can be audited and so its staleness is testable
# (see tests/test_orphan_gate_is_falsifiable.py).
#
# Why declared rather than inferred: the flow is real but not statically
# name-trackable. `fetch_and_prepare` builds a local called `raw` (80 keys) and
# returns it; `main.py:19582` delegates to it from `_prepare_market_data`; the
# callers bind THAT to `frt_md`, `_p6_md`, or an `asyncio.gather` result — the
# name `market_data` never appears in the chain. Inferring it would take
# interprocedural dataflow through a gather(); asserting it without the analysis
# would be the P174 mistake again. So it is stated, sourced, and pinned instead.
#
# Correction to the P174 docstring that stood here: P171 said "the pipeline
# fills `raw`", and P174 called that a wrong guess. P171 was right — the
# returned local IS named `raw`. The correction was the error.
# Keyed (module suffix, function name) -> the signal dict that return value
# becomes. "*" means every producing function in the module. Each entry is a
# claim about the architecture that a human checked; add one only with the call
# chain written down beside it.
PRODUCER_MODULES = {
    # fetch_and_prepare -> main.py:19582 _prepare_market_data -> frt_md /
    # _p6_md / asyncio.gather results, all consumed as market_data.
    ("data_mgmt/market_data_pipeline.py", "*"): "market_data",
    # main.py:6741  position_state = self._get_effective_position_state(...)
    # Builds current_exposure, direction, exposure, has_position, tranche.
    ("main.py", "_get_effective_position_state"): "position_state",
}


def collect_produced_by_function(tree: ast.AST) -> Dict[str, Set[str]]:
    """[P176] Same analysis as collect_produced_keys, split per function.

    Module-level attribution was too coarse: `main.py` produces position_state
    in one function and a dozen unrelated dicts elsewhere, so crediting the
    whole file would have handed `market_data` every key main.py ever returns
    and re-hidden the drift this scanner exists to find.
    """
    out: Dict[str, Set[str]] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        out.setdefault(fn.name, set()).update(_produced_in_function(fn))
    return out


def _produced_in_function(fn: ast.AST) -> Set[str]:
    produced: Set[str] = set()
    ret_names = {n.value.id for n in ast.walk(fn)
                 if isinstance(n, ast.Return) and isinstance(n.value, ast.Name)}
    for n in ast.walk(fn):
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict):
            produced |= {k.value for k in n.value.keys
                         if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if not isinstance(n, ast.Assign):
            continue
        for t in n.targets:
            if isinstance(t, ast.Name) and t.id in ret_names and isinstance(n.value, ast.Dict):
                produced |= {k.value for k in n.value.keys
                             if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            elif (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                  and t.value.id in ret_names
                  and isinstance(t.slice, ast.Constant)
                  and isinstance(t.slice.value, str)):
                produced.add(t.slice.value)
    return produced


def collect_produced_keys(tree: ast.AST) -> Set[str]:
    """[P174] Keys of dicts a function builds under a LOCAL name and returns.

    `data_mgmt/market_data_pipeline.py` is the real producer of `market_data`:
    `fetch_and_prepare` builds 80 keys into a local named `raw` and returns it,
    with `_fetch_live_data` and `generate_verification_data` doing the same on
    the fallback paths. None of that is a write to a name in TARGET_DICTS, so
    the scanner once saw ~2500 produced keys tree-wide as produced by nobody.

    Tree-wide these are credited as PRODUCED_ELSEWHERE — the destination is
    unknown, and pretending otherwise would let a real misroute hide behind a
    same-named producer. For the modules in PRODUCER_MODULES the destination IS
    known, and those keys are additionally credited to that specific dict; see
    `produced_into` in run().
    """
    produced: Set[str] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        ret_names = {n.value.id for n in ast.walk(fn)
                     if isinstance(n, ast.Return) and isinstance(n.value, ast.Name)}
        for n in ast.walk(fn):
            # `return {...}` straight out
            if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict):
                produced |= {k.value for k in n.value.keys
                             if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if not isinstance(n, ast.Assign):
                continue
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in ret_names and isinstance(n.value, ast.Dict):
                    produced |= {k.value for k in n.value.keys
                                 if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                elif (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                      and t.value.id in ret_names
                      and isinstance(t.slice, ast.Constant)
                      and isinstance(t.slice.value, str)):
                    produced.add(t.slice.value)
    return produced


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.reads: List[Dict[str, Any]] = []
        self.writes: Set[Tuple[str, str]] = set()      # (dict_name, key)
        self.dynamic: Set[str] = set()                 # dict names written by computed key
        self.dynamic_lines: List[Tuple[str, int]] = []  # (dict_name, lineno)
        # [P174] (lineno, col) of reads that are the fallback ARM of a
        # `a.get(k, b.get(k, ...))` chain. Reported through the outer read only.
        self.chain_arms: Set[Tuple[int, int]] = set()
        # [P174] Writes whose RHS just copies the same key off another signal
        # dict. Tracked separately: a copy is not evidence of a producer.
        self.copy_writes: Set[Tuple[str, str]] = set()

    # ---- writes -----------------------------------------------------------

    def _mark_dynamic(self, name: str, lineno: int) -> None:
        self.dynamic.add(name)
        self.dynamic_lines.append((name, lineno))

    def visit_Assign(self, node: ast.Assign) -> None:
        for tgt in node.targets:
            # agent_signals["key"] = ...
            if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
                name = tgt.value.id
                if name in TARGET_DICTS:
                    if isinstance(tgt.slice, ast.Constant) and isinstance(tgt.slice.value, str):
                        self.writes.add((name, tgt.slice.value))
                        if _is_copy_of(tgt.slice.value, node.value):
                            self.copy_writes.add((name, tgt.slice.value))
                    else:
                        self._mark_dynamic(name, node.lineno)
            # agent_signals = { "key": ... }  — the literal that builds the dict
            elif isinstance(tgt, ast.Name) and tgt.id in TARGET_DICTS:
                if _is_null_coalesce(tgt, node.value):
                    continue  # [P174] `x = x or {}` adds no key
                if isinstance(node.value, ast.Dict):
                    for k, v in zip(node.value.keys, node.value.values):
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            self.writes.add((tgt.id, k.value))
                            if _is_copy_of(k.value, v):
                                self.copy_writes.add((tgt.id, k.value))
                        elif k is None:  # {**other}
                            self._mark_dynamic(tgt.id, node.lineno)
                else:
                    # Aliased from something else — cannot prove key set.
                    self._mark_dynamic(tgt.id, node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            name, attr = func.value.id, func.attr
            if name in TARGET_DICTS:
                if attr in ("setdefault", "pop") and node.args:
                    a0 = node.args[0]
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                        if attr == "setdefault":
                            self.writes.add((name, a0.value))
                    else:
                        self._mark_dynamic(name, node.lineno)
                elif attr == "update":
                    if node.args and isinstance(node.args[0], ast.Dict):
                        for k in node.args[0].keys:
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                self.writes.add((name, k.value))
                            else:
                                self._mark_dynamic(name, node.lineno)
                    else:
                        self._mark_dynamic(name, node.lineno)
                    for kw in node.keywords:
                        if kw.arg:
                            self.writes.add((name, kw.arg))
                        else:
                            self._mark_dynamic(name, node.lineno)
                elif attr == "get" and node.args:
                    a0 = node.args[0]
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                        default = node.args[1] if len(node.args) > 1 else None
                        # [P174] `a.get(k, b.get(k, dflt))` is the hand-rolled
                        # form of core.market_data_helpers.signal_value: it
                        # reads BOTH dicts for the SAME key, so it is not a
                        # misroute even though each arm alone looks like one.
                        # Flagging it as a defect would have sent a reviewer to
                        # "fix" main.py:12086, which is already correct — and
                        # noise in a gated metric is how gates get re-baselined
                        # into silence.
                        chained_with: Set[str] = set()
                        inner = default
                        while (isinstance(inner, ast.Call)
                               and isinstance(inner.func, ast.Attribute)
                               and inner.func.attr == "get"
                               and isinstance(inner.func.value, ast.Name)
                               and inner.func.value.id in TARGET_DICTS
                               and inner.args
                               and isinstance(inner.args[0], ast.Constant)
                               and inner.args[0].value == a0.value):
                            chained_with.add(inner.func.value.id)
                            self.chain_arms.add((inner.lineno, inner.col_offset))
                            inner = inner.args[1] if len(inner.args) > 1 else None
                        self.reads.append({
                            "dict": name,
                            "key": a0.value,
                            "file": self.path,
                            "line": node.lineno,
                            "col": node.col_offset,
                            "chained_with": sorted(chained_with),
                            "default": _default_repr(default) if default is not None else "None",
                            # The terminal default of a chain is what actually
                            # lands when every arm misses; that is the one whose
                            # falsiness decides HOT.
                            "hot": (inner is not None and not _is_falsy_default(inner))
                                   if chained_with else
                                   (default is not None and not _is_falsy_default(default)),
                        })
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # agent_signals["key"] in a load context is also a read, but a missing
        # key raises there — loud, not silent. Recorded for completeness only.
        self.generic_visit(node)


#: Files the scanner could not parse. A file that fails to parse contributes
#: NO writes, which silently promotes every key it produces to "orphan". The
#: first version of this scanner swallowed the failure and returned empty —
#: main.py starts with a UTF-8 BOM, so `encoding="utf-8"` raised SyntaxError on
#: U+FEFF, the biggest producer in the tree vanished, and the scan reported 36
#: confident false positives. A scanner that cannot read the code is not a
#: scanner that found nothing. These are surfaced and made fatal in run().
PARSE_FAILURES: List[str] = []


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)  # scanned from outside the repo (tests)


class FileScan(NamedTuple):
    reads: List[Dict[str, Any]]
    writes: Set[Tuple[str, str]]
    dynamic: Set[str]
    dynamic_sites: List[str]       # "file:line:dict", one per real dynamic write
    produced: Set[str]             # [P174] keys built under a local name
    produced_by_fn: Dict[str, Set[str]]  # [P176] same, split per function
    chain_arms: Set[Tuple[str, int, int]]  # [P174] (file, line, col) fallback arms
    copy_writes: Set[Tuple[str, str]]      # [P174] writes that merely relay a key


_EMPTY_SCAN = FileScan([], set(), set(), [], set(), {}, set(), set())


def scan_file(path: Path) -> FileScan:
    rel = _rel(path)
    try:
        # utf-8-sig, not utf-8: a BOM is not a syntax error, it is an encoding
        # detail, and treating it as one hid main.py entirely.
        src = path.read_text(encoding="utf-8-sig", errors="replace")
        tree = ast.parse(src)
    except (SyntaxError, ValueError) as exc:
        PARSE_FAILURES.append(f"{rel}: {type(exc).__name__}: {exc}")
        return _EMPTY_SCAN
    except OSError as exc:
        PARSE_FAILURES.append(f"{rel}: unreadable: {exc}")
        return _EMPTY_SCAN
    v = _Visitor(rel)
    v.visit(tree)
    return FileScan(
        reads=v.reads,
        writes=v.writes,
        dynamic=v.dynamic,
        # [P174] Sites carry a line number now. "main.py:agent_signals" could not
        # be diffed usefully — one site or twelve looked identical, so the blind
        # spot could grow without the count moving.
        dynamic_sites=[f"{rel}:{ln}:{d}" for d, ln in sorted(v.dynamic_lines, key=lambda x: x[1])],
        produced=collect_produced_keys(tree),
        produced_by_fn=collect_produced_by_function(tree),
        chain_arms={(rel, ln, col) for ln, col in v.chain_arms},
        copy_writes=v.copy_writes,
    )


def iter_py(paths: List[str]):
    for p in paths:
        base = REPO_ROOT / p
        if base.is_file() and base.suffix == ".py":
            yield base
        elif base.is_dir():
            for f in base.rglob("*.py"):
                if any(part in EXCLUDED_DIRS for part in f.parts):
                    continue
                yield f


def run(paths: List[str]) -> Dict[str, Any]:
    # PARSE_FAILURES is module state; a second run() in the same process must
    # not inherit the first run's failures (or, worse, report a clean scan
    # because someone cleared it by hand).
    PARSE_FAILURES.clear()
    all_reads: List[Dict[str, Any]] = []
    all_writes: Set[Tuple[str, str]] = set()
    produced_elsewhere: Set[str] = set()
    dynamic: Set[str] = set()
    dynamic_sites: List[str] = []
    chain_arms: Set[Tuple[str, int, int]] = set()
    copy_writes: Set[Tuple[str, str]] = set()

    # [P176] Keys a PRODUCER_MODULES file builds and returns, credited to the
    # specific dict that return value becomes. Reading one of these off that
    # dict is correct code, even if the key is also copied into another signal
    # dict elsewhere — which is exactly the shape that produced 10 false HOTs.
    produced_into: Dict[str, Set[str]] = {}

    for f in iter_py(paths):
        s = scan_file(f)
        all_reads.extend(s.reads)
        all_writes |= s.writes
        produced_elsewhere |= s.produced
        norm = str(f).replace("\\", "/").lstrip("./")
        for (mod, func), dest in PRODUCER_MODULES.items():
            if not norm.endswith(mod):
                continue
            by_fn = s.produced_by_fn
            keys = (s.produced if func == "*"
                    else by_fn.get(func, set()))
            produced_into.setdefault(dest, set()).update(keys)
        dynamic |= s.dynamic
        dynamic_sites.extend(s.dynamic_sites)
        chain_arms |= s.chain_arms
        copy_writes |= s.copy_writes

    # [P174] A key whose every static write is a relay, with no function
    # anywhere building it. The copies prove somebody intended the key to exist;
    # the missing producer proves it never does.
    copy_only_keys = ({k for (_d, k) in copy_writes}
                      - {k for (_d, k) in (all_writes - copy_writes)}
                      - produced_elsewhere)

    orphans: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str, int]] = set()
    for r in all_reads:
        key, dname = r["key"], r["dict"]
        if key in KNOWN_OPTIONAL:
            continue
        # [P174] The inner arm of `a.get(k, b.get(k, d))`. The chain is judged
        # once, at the outer read, which knows about every dict it covers.
        if (r["file"], r["line"], r.get("col", -1)) in chain_arms:
            continue
        # A key written to ANY of the signal dicts counts as produced — the
        # cross-dict case (written to market_data, read from agent_signals) is
        # reported separately as MISROUTED, since that is P170's exact shape.
        written_same = (dname, key) in all_writes
        written_other = any(k == key for (_d, k) in all_writes)
        if written_same:
            continue
        sig = (dname, key, r["file"], r["line"])
        if sig in seen:
            continue
        seen.add(sig)
        r = dict(r)
        r["severity"] = "HOT" if r["hot"] else "COLD"
        if key in copy_only_keys:
            r["kind"] = "COPY_ONLY"
            orphans.append(r)
            continue
        if any((d, key) in all_writes for d in r.get("chained_with", ())):
            # [P174] A fallback chain that DOES cover the producing dict. The
            # read as a whole finds the key, so it is correct code.
            r["kind"] = "FALLBACK_CHAIN"
        elif key in produced_into.get(dname, ()):
            # [P176] A declared producer fills THIS dict with THIS key, so the
            # read succeeds. Must outrank MISROUTED: `data_valid` is produced
            # into market_data by the pipeline AND copied into system_state at
            # main.py:6755, and testing written_other first labelled all four
            # correct market_data reads as misroutes. All 10 HOT findings in the
            # P174 baseline were this, i.e. the metric meant to draw the eye was
            # 100% noise.
            #
            # It must NOT outrank the cross-dict case, which is why this is
            # keyed on `dname`: P170's bug was `agent_signals.get(
            # "quant_data_quality")` where the pipeline produces the key into
            # *market_data*. That read is not covered by produced_into
            # ["agent_signals"], so it still lands in MISROUTED below. Dropping
            # the dname check — plain "produced beats misrouted" — silences
            # P170, P173's `drl_confidence` and `phase`, and every bug this
            # scanner was built for.
            r["kind"] = "PRODUCED_HERE"
        elif any(key in ks for d2, ks in produced_into.items() if d2 != dname):
            # [P176] P170's EXACT shape, and until now the scanner could not see
            # it. A declared producer builds this key into a different signal
            # dict, and nothing anywhere copies it into the one being read —
            # `written_same` is False or we would have skipped this read
            # already. So the read always misses and the default always wins.
            #
            # This is what `agent_signals.get("quant_data_quality", 1.0)` was:
            # produced into market_data, never copied across, default 1.0 =
            # "healthy". Reconstructed synthetically, the P174 classifier filed
            # it as PRODUCED_ELSEWHERE — untriaged and ungated. The scanner
            # built to catch P170 would not have caught P170.
            r["kind"] = "MISROUTED"
        elif written_other:
            # Statically written to a DIFFERENT signal dict. The strongest
            # signal this scanner produces — P170 and all three P173 bugs had
            # exactly this shape, with the correct read nearby in the same file.
            # A chain that misses the producer still lands here: covering two
            # wrong dicts is not better than covering one.
            r["kind"] = "MISROUTED"
        elif key in produced_elsewhere:
            # [P174] Someone builds this key, under a local name, and returns
            # it. Whether it reaches THIS dict is not statically knowable, so
            # this is reported and not gated — but it is no longer lumped in
            # with keys that genuinely have no producer anywhere.
            r["kind"] = "PRODUCED_ELSEWHERE"
        elif dname in dynamic:
            r["kind"] = "UNPROVABLE"
        else:
            r["kind"] = "ORPHAN"
        orphans.append(r)

    # [P174] How many unmatched reads the ORPHAN verdict could even be applied
    # to. When this is 0, `orphan_count: 0` means "the check did not run", not
    # "the check passed". Emitted so the distinction survives into the baseline.
    adjudicable = len([o for o in orphans if o["dict"] not in dynamic])
    misrouted = [o for o in orphans if o["kind"] == "MISROUTED"]
    hot = [o for o in orphans if o["severity"] == "HOT"
           and o["kind"] in ("ORPHAN", "MISROUTED")]
    return {
        # Reported first and checked by the CI gate: findings computed from a
        # partial parse are not findings.
        "parse_failures": sorted(PARSE_FAILURES),
        "total_reads": len(all_reads),
        "total_written_keys": len(all_writes),
        "produced_elsewhere_keys": len(produced_elsewhere),
        "orphan_count": len([o for o in orphans if o["kind"] == "ORPHAN"]),
        "orphan_adjudicable": adjudicable,
        "misrouted_count": len(misrouted),
        "misrouted_hot_count": len([o for o in misrouted if o["severity"] == "HOT"]),
        "dynamic_dicts": sorted(dynamic),
        "dynamic_site_count": len(set(dynamic_sites)),
        "hot_count": len(hot),
        "by_kind": {
            k: len([o for o in orphans if o["kind"] == k])
            for k in ("ORPHAN", "COPY_ONLY", "MISROUTED", "FALLBACK_CHAIN",
                      "PRODUCED_HERE", "PRODUCED_ELSEWHERE", "UNPROVABLE")
        },
        "findings": sorted(
            orphans,
            key=lambda o: (o["kind"], o["severity"] != "HOT", o["file"], o["line"]),
        ),
        "dynamic_write_sites": sorted(set(dynamic_sites)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--baseline-format", action="store_true",
                    help="Emit only the gate-safe metrics (see below).")
    ap.add_argument("--paths", nargs="*", default=None)
    args = ap.parse_args()

    paths = args.paths or [
        p.name for p in REPO_ROOT.iterdir()
        if (p.is_dir() and p.name not in EXCLUDED_DIRS and not p.name.startswith("."))
        or (p.is_file() and p.suffix == ".py")
    ]
    result = run(paths)

    if args.baseline_format:
        # [P174] These metrics replace the original `orphan_count`-only gate,
        # which could not fail. Each one here can actually move:
        #
        #   misrouted_count      — a read of a key that IS statically written,
        #                          but to a different signal dict. P170 and all
        #                          three P173 bugs had this exact shape. Noisy
        #                          (a dynamic copy may legitimately deliver the
        #                          key) but bounded and hand-triaged, so a RISE
        #                          is a new suspect worth a human look. `_diff`
        #                          only fails on increases, which is the right
        #                          direction for a metric like this.
        #   dynamic_site_count   — the size of the blind spot. Every new dynamic
        #                          write makes more of the tree unprovable. If
        #                          this may grow freely, the other numbers can
        #                          be driven to zero by making the code less
        #                          analyzable, which is the wrong incentive.
        #   orphan_count         — still sound (no writer anywhere, no dynamic
        #                          write to that dict), so a rise is real.
        #   orphan_coverage_lost — negated `orphan_adjudicable`, so a DROP in
        #                          coverage reads as a rise and trips `_diff`.
        #                          BE CLEAR ABOUT WHAT THIS DOES TODAY: coverage
        #                          is 0, so this sits at its floor and cannot
        #                          rise. It is inert, not protective — recording
        #                          it as a live guard would repeat P171's error
        #                          one level up. It arms itself only if someone
        #                          reduces the dynamic writes and real coverage
        #                          appears; until then `dynamic_site_count` is
        #                          the metric doing the work.
        #   parse_failure_count  — a file that fails to parse contributes no
        #                          writes and inflates every other number. A
        #                          scan that could not read the code is not a
        #                          clean scan. NOT re-baselineable.
        #   copy_only_count      — keys that only ever get relayed between
        #                          signal dicts, with no function anywhere
        #                          building one. Small, precise and hand-vetted;
        #                          it independently rediscovered
        #                          `is_4h_bar_close`, which P173 had triaged by
        #                          hand, which is the evidence that it is worth
        #                          gating. Current members are documented at the
        #                          baseline file.
        print(json.dumps({
            "copy_only_count": result["by_kind"]["COPY_ONLY"],
            "misrouted_count": result["misrouted_count"],
            "misrouted_hot_count": result["misrouted_hot_count"],
            "dynamic_site_count": result["dynamic_site_count"],
            "orphan_count": result["orphan_count"],
            "orphan_coverage_lost": -result["orphan_adjudicable"],
            "parse_failure_count": len(result["parse_failures"]),
        }, indent=2, sort_keys=True))
        return 0

    if args.json:
        print(json.dumps(result, indent=2))
        return 2 if result["parse_failures"] else 0

    if result["parse_failures"]:
        print("[orphan-reads] REFUSING TO REPORT — files failed to parse:")
        for f in result["parse_failures"]:
            print(f"    {f}")
        print("  Every key produced by an unparsed file looks orphaned. Fix the "
              "parse before trusting any finding below.")
        return 2

    print(f"[orphan-reads] scanned {result['total_reads']} keyed .get() reads "
          f"across {result['total_written_keys']} statically-written keys "
          f"(+{result['produced_elsewhere_keys']} built under local names)")
    print(f"[orphan-reads] ORPHAN={result['by_kind']['ORPHAN']} "
          f"COPY_ONLY={result['by_kind']['COPY_ONLY']} "
          f"MISROUTED={result['by_kind']['MISROUTED']} "
          f"FALLBACK_CHAIN={result['by_kind']['FALLBACK_CHAIN']} "
          f"PRODUCED_HERE={result['by_kind']['PRODUCED_HERE']} "
          f"PRODUCED_ELSEWHERE={result['by_kind']['PRODUCED_ELSEWHERE']} "
          f"UNPROVABLE={result['by_kind']['UNPROVABLE']} "
          f"(HOT={result['hot_count']})")

    # [P174] State the coverage before the findings, not after. The original
    # version printed a finding list and a footnote, and the footnote was where
    # "this check adjudicated nothing" went to be ignored.
    unmatched = sum(result["by_kind"].values())
    adj = result["orphan_adjudicable"]
    print(f"[orphan-reads] ORPHAN coverage: {adj}/{unmatched} unmatched reads "
          f"adjudicable" + (f" — VACUOUS, dynamic writes to "
                            f"{', '.join(result['dynamic_dicts'])} make absence "
                            f"unprovable there" if adj == 0 else ""))
    print()
    for o in result["findings"]:
        if o["kind"] in ("UNPROVABLE", "PRODUCED_ELSEWHERE", "FALLBACK_CHAIN",
                         "PRODUCED_HERE"):
            continue
        print(f"  [{o['kind']}/{o['severity']}] {o['file']}:{o['line']}  "
              f"{o['dict']}.get({o['key']!r}, {o['default']})")
    if result["dynamic_write_sites"]:
        print(f"\n  ({result['dynamic_site_count']} dynamic write sites exempted "
              f"— keys there cannot be proven absent)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
