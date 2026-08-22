"""[P372] Two observability defects, one P2 shape each.

DEFECT A — the [PROOF] line printed ``Macro=DISABLED`` while macro was LIVE.
  main.py is the ONLY writer of ``agent_signals["macro_crowd_context"]``. It
  publishes the adapter's macro dict — liveness field ``source_status ==
  "available"`` (P293, FRED provenance) — and writes the literal
  ``{"available": False}`` only in its ELSE branch. integration_v36's two
  stash sites read ``.get("available", False)``, a key the LIVE dict never
  carries, so ``macro_real`` was False on every tick the feed served data.
  Live evidence (2026-08-22): diag ``macro_status: 'available'`` and
  ``[PROOF] ... Macro=DISABLED`` in the same tick. Fixed at the reader with a
  pure helper that accepts the producer's shape.

DEFECT B — ``lead_lag_edge`` reads 0 in fusion since P287 namespaced B6.
  RECORDED, NOT REWIRED. P287 moved the BETA-CATCH-UP estimator to
  ``b6_lead_lag_edge`` precisely because it was OVERWRITING the exchange
  lead-lag key that micro/the LeadLagAlphaEngine own. The genuine owner
  (micro) is neutral today (``insufficient_samples``), and its neutral
  payload carries ``micro_is_valid: True`` so the bridge copies an honest
  0.0. Pointing the reader at ``b6_`` would re-create the collision P287
  ended; the pipeline never wrote the key on its real path. The only new
  fact worth pinning: ``b6_lead_lag_edge`` is WRITE-ONLY (zero readers) —
  wiring one is a live-behaviour decision, not a bugfix.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests._guard_pins import assert_text_pin

REPO = Path(__file__).resolve().parents[1]
V36_PATH = REPO / "integration" / "integration_v36.py"
MAIN_PATH = REPO / "main.py"
PIPE_PATH = REPO / "data_mgmt" / "market_data_pipeline.py"

V36_SRC = V36_PATH.read_text(encoding="utf-8-sig")
MAIN_SRC = MAIN_PATH.read_text(encoding="utf-8-sig")
PIPE_SRC = PIPE_PATH.read_text(encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# DEFECT A — the helper, the producer shapes, and the emitted line
# ---------------------------------------------------------------------------

from integration.integration_v36 import (  # noqa: E402
    HMATSv36Engine,
    TradeIntentV36,
    macro_context_is_live,
)
from defense.production_reliability import create_proof_log  # noqa: E402


# The dict main.py actually publishes when the adapter's FRED provenance
# says it has data (main.py ~L8125 + the L11817 writer).
LIVE_PRODUCER_SHAPE = {
    "event_window": {"active": False, "name": None, "t_minus_min": 0, "importance": 0},
    "macro_shock": 0.0,
    "macro_risk": 0.5,
    "source_status": "available",
    "provenance": {"fred": True},
}


@pytest.mark.parametrize(
    "mcc, expected",
    [
        (LIVE_PRODUCER_SHAPE, True),                    # the live shape — the bug
        ({"source_status": "available"}, True),
        ({"source_status": "AVAILABLE"}, True),
        ({"source_status": "defaults_only"}, False),    # P293: adapter returned hardcoded defaults
        ({"source_status": "unavailable"}, False),      # main.py's initial value
        ({"available": False}, False),                  # main.py's ELSE branch literal
        ({"available": True}, True),                    # legacy explicit shape
        ({"available": False, "source_status": "available"}, True),
        ({}, False),
        (None, False),
        (True, True),
        (False, False),
    ],
)
def test_macro_context_is_live_truth_table(mcc, expected):
    assert macro_context_is_live(mcc) is expected


def _engine_for_proof_log() -> HMATSv36Engine:
    """An engine shell with exactly the attributes _add_proof_log reads."""
    e = object.__new__(HMATSv36Engine)
    e.fusion_engine = object()
    e.drl_gate = object()
    e._last_veto_result = None
    e._last_deadlock_result = None
    e._last_sol_exit_signal = None
    e._proof_logs = []
    e._structured_proof_logs = []
    return e


def _macro_field(line: str) -> str:
    m = re.search(r"\| Macro=([A-Z]+) \|", line)
    assert m, line
    return m.group(1)


def test_proof_line_says_REAL_when_macro_is_live():
    """End to end through the REAL _add_proof_log with the stash set the way
    decide() now sets it, from the producer's actual dict."""
    e = _engine_for_proof_log()
    e._last_macro_active = macro_context_is_live(LIVE_PRODUCER_SHAPE)
    e._add_proof_log(TradeIntentV36(asset="ETH"))
    line = e._structured_proof_logs[-1].to_log_line()
    assert _macro_field(line) == "REAL", line


def test_proof_line_says_DISABLED_when_macro_is_genuinely_off():
    """main.py's ELSE-branch literal (feed unavailable / defaults_only) must
    still render DISABLED — the fix must not turn the label into a constant
    REAL (P174: a label that cannot vary is not a label)."""
    for off_shape in ({"available": False}, {"source_status": "defaults_only"}, None):
        e = _engine_for_proof_log()
        e._last_macro_active = macro_context_is_live(off_shape)
        e._add_proof_log(TradeIntentV36(asset="ETH"))
        line = e._structured_proof_logs[-1].to_log_line()
        assert _macro_field(line) == "DISABLED", (off_shape, line)


def test_create_proof_log_maps_macro_real_to_the_label():
    """The label is a pure function of macro_real — pin the mapping, since the
    reader-side fix rests on it."""
    assert _macro_field(create_proof_log(macro_real=True).to_log_line()) == "REAL"
    assert _macro_field(create_proof_log(macro_real=False).to_log_line()) == "DISABLED"


def test_both_decide_stash_sites_route_through_the_helper():
    """decide() stashes _last_macro_active at ENTRY (for early-return proof
    logs) and again in _build_fusion_signals. Both must go through the helper
    — fixing one leaves the other printing DISABLED on whichever path it
    serves. Pinned with unique anchors (P238/P350)."""
    assert_text_pin(
        V36_SRC,
        "self._last_macro_active = macro_context_is_live(_early_mcc)",
        near='_early_mcc = agent_signals.get("macro_crowd_context")',
        why="the ENTRY stash (early-return proof logs) no longer reads the producer's shape",
    )
    assert_text_pin(
        V36_SRC,
        "self._last_macro_active = macro_context_is_live(_mcc)",
        # leading newline: `_mcc = ...` is a SUBSTRING of `_early_mcc = ...` —
        # sibling ambiguity, the P357 class, caught by the helper's refusal
        near='\n        _mcc = agent_signals.get("macro_crowd_context")',
        why="the _build_fusion_signals stash no longer reads the producer's shape",
    )
    # The old expression — reading a key the producer never writes — is gone
    # from every assignment of the stash.
    for m in re.finditer(r"self\._last_macro_active\s*=\s*(.+)", V36_SRC):
        rhs = m.group(1)
        assert 'get("available", False)' not in rhs, (
            "a _last_macro_active assignment reads `.get(\"available\")` again — "
            "main.py publishes `source_status`, so this is the P372 defect "
            "re-created: " + rhs)


def test_the_producer_still_writes_the_shape_the_helper_accepts():
    """The reader-side fix is correct only while main.py keeps publishing the
    adapter dict keyed by `source_status`. If the writer changes shape, this
    fails and the helper must be re-derived against it — the P310 both-
    directions rule for a reader/writer contract."""
    assert_text_pin(
        MAIN_SRC,
        "agent_signals['macro_crowd_context'] = _mcc",
        near="_mcc = market_data.get('macro')",
        why="main.py no longer publishes the adapter macro dict under macro_crowd_context",
    )
    assert_text_pin(
        MAIN_SRC,
        "_mcc.get('source_status') == 'available'",
        near="_mcc = market_data.get('macro')",
        why="the writer's liveness field is no longer source_status — re-derive macro_context_is_live",
    )
    # and the genuine-off literal the DISABLED test relies on
    assert_text_pin(
        MAIN_SRC,
        "agent_signals['macro_crowd_context'] = {'available': False}",
        why="the writer's off-branch literal changed shape",
    )


def test_helper_is_marked_and_the_dict_branch_is_live():
    """The helper must honour BOTH dict shapes; a probe that drops the
    source_status branch turns the truth table red, and this pin names it."""
    src = V36_SRC
    i = src.index("def macro_context_is_live(")
    body = src[i:i + 1500]
    assert "[P372]" in body
    assert 'mcc.get("source_status"' in body
    assert 'mcc.get("available"' in body


# ---------------------------------------------------------------------------
# DEFECT B — recorded disposition, pinned so it does not rot silently
# ---------------------------------------------------------------------------

def _readers_of(key: str, src: str, path: Path):
    """Every read of `key` via .get()/subscript in a file, minus its writes."""
    out = []
    lines = src.splitlines()
    write_marker = "['" + key + "'] = "
    for m in re.finditer(r"\.get\(\s*['\"]" + re.escape(key) + r"['\"]", src):
        out.append((path.name, src.count("\n", 0, m.start()) + 1))
    for m in re.finditer(r"\[\s*['\"]" + re.escape(key) + r"['\"]\s*\](?!\s*=[^=])", src):
        ln = src.count("\n", 0, m.start()) + 1
        if write_marker in lines[ln - 1]:
            continue  # a mirror write (`market_data[k] = agent_signals[k]`) is not a reader
        out.append((path.name, ln))
    return out


def _py_files():
    for d in ("agents", "analytics", "core", "data_mgmt", "defense", "execution",
              "exchange", "integration", "orchestration", "risk", "signals"):
        yield from (REPO / d).rglob("*.py")
    yield MAIN_PATH


def test_b6_lead_lag_edge_is_written_and_read_by_nothing():
    """P287 namespaced the beta-catch-up estimator so it would stop
    OVERWRITING micro's exchange lead-lag. The consequence nobody recorded:
    the namespaced value has NO consumer anywhere — it is computed, logged
    ([LEAD_LAG] edge=-38.6bps live on 2026-08-22) and written to two dicts
    that nothing reads. Pinned as the DECIDED state (P318): if this fails a
    reader appeared, and that reader is feeding a DIFFERENT estimator into
    whatever channel it chose — the P287 collision one key over — and it
    needs its own P-entry, not a quiet wire."""
    writers = MAIN_SRC.count("['b6_lead_lag_edge'] = ")
    assert writers == 2, writers  # agent_signals + market_data mirror (main.py ~L11381/11385)
    readers = []
    for p in _py_files():
        if "__pycache__" in str(p):
            continue
        readers += _readers_of("b6_lead_lag_edge", p.read_text(encoding="utf-8-sig"), p)
    assert readers == [], (
        "b6_lead_lag_edge gained a reader %r — feeding B6's beta-catch-up edge "
        "into a lead_lag consumer is the mixed-provenance P287 ended; decide "
        "it in a P-entry (P141) rather than here" % (readers,))


def test_lead_lag_edge_is_owned_by_micro_and_b6_does_not_restore_the_collision():
    """The legacy key's only direct producers after P287: the micro bridge
    (copies the agent payload wholesale) and the setdefault. B6 must not be
    pointed back at it, and main.py's market_data read of the key must not
    be repointed at b6_ — either re-creates the collision."""
    assert "agent_signals['lead_lag_edge'] = round(" not in MAIN_SRC
    assert "market_data['lead_lag_edge'] = agent_signals" not in MAIN_SRC
    assert 'market_data.get("b6_lead_lag_edge"' not in MAIN_SRC
    assert "market_data.get('b6_lead_lag_edge'" not in MAIN_SRC
    # the micro bridge is the owner
    assert_text_pin(
        MAIN_SRC,
        "for _mk, _mv in _micro_sig.items():",
        near='if _micro_sig and _micro_sig.get("micro_is_valid", False):',
        why="the micro bridge no longer copies the agent payload — lead_lag_edge's owner moved",
    )
    assert_text_pin(
        MAIN_SRC,
        "agent_signals.setdefault('lead_lag_edge', 0.0)",
        why="the lead_lag_edge default site moved",
    )


def test_micro_neutral_payload_is_an_honest_zero_that_the_bridge_copies():
    """Question (3) of the audit: does micro's neutral payload overwrite
    anything? YES — it carries micro_is_valid=True, so the bridge copies its
    lead_lag_edge=0.0 over whatever main.py's market_data read set. That is
    correct (0.0 IS the exchange lead-lag while micro has no samples) and
    explains the live `lead_lag: edge_bps 0.0` while B6 logs -38.6bps."""
    from agents.microstructure_agent import _neutral_v6_payload
    p = _neutral_v6_payload("ETH")
    assert p["micro_is_valid"] is True
    assert p["lead_lag_edge"] == 0.0
    assert p["lead_lag_confidence"] == 0.0


def test_pipeline_never_writes_lead_lag_edge_on_its_real_path():
    """main.py:~9021 reads market_data['lead_lag_edge']; the pipeline's only
    write of that key sits in generate_verification_data (the DEGRADED/
    synthetic path, where it is a fabricated 35bps). So on a real tick that
    read is a structural 0.0 — which is why the micro bridge later in the
    same tick is the key's only live producer. If a real-path write appears,
    the ownership statement above is stale."""
    tree = ast.parse(PIPE_SRC)
    writer_funcs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    for k in sub.keys:
                        if isinstance(k, ast.Constant) and k.value == "lead_lag_edge":
                            writer_funcs.add(node.name)
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if (isinstance(t, ast.Subscript)
                                and isinstance(t.slice, ast.Constant)
                                and t.slice.value == "lead_lag_edge"):
                            writer_funcs.add(node.name)
    assert writer_funcs == {"generate_verification_data"}, writer_funcs


def test_the_reader_detector_still_sees_a_reader():
    """Anti-vacuity (P174): the no-consumer pin above rests on _readers_of;
    a detector that sees nothing would make that pin pass forever."""
    fake = ("x = agent_signals.get('b6_lead_lag_edge', 0.0)\n"
            "market_data['b6_lead_lag_edge'] = agent_signals['b6_lead_lag_edge']\n"
            "y = market_data['b6_lead_lag_edge']\n")
    found = _readers_of("b6_lead_lag_edge", fake, Path("fake.py"))
    assert [ln for _, ln in found] == [1, 3], found
