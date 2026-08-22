"""[P366] ROOT CAUSE for kraken_quant's silence: the converter and the twelve
strategies have never had a contract, and every mismatch fails SILENTLY.

The operator declined my symptom-level options ("repair 3 defects", "retire
the authority", "persist the buffers") and asked for research and a root-cause
fix. This is it, and the research found more than hand-inspection had.

--------------------------------------------------------------------------
THE BOUNDARY
--------------------------------------------------------------------------
`KrakenQuantAgentV6._convert_market_data` emits a FIXED 9-key dict:

    timestamp, prices, open_interest, funding_rate, liquidations,
    taker_ratio, liquidation_intensity, bid_depth, ask_depth

The twelve strategies were written against a richer `market_data` shape from
their original standalone context. **Nobody ever checked that what the
converter produces covers what the strategies read.** Enumerated by AST over
every `market_data.get("X")` and `market_data["X"]`, FOUR read keys that are
never produced:

    DarkPoolVolumeStrategy        market_data[<asset>]      FATAL
    ETFSpotCointegrationStrategy  close/open/volume_24h/... FATAL
    FundingDivergenceStrategy     predicted_funding         one branch dead
    LiquidationCascadeHunter      cvd                       BY DECISION

**This corrects P358c**, which reported 3 STRUCTURAL / 3 WARMUP from hand
inspection. `FundingDivergence` was filed as WARMUP only; it is also missing
an input, so one of its two signal paths can never run. `LiquidationCascade`
was never inspected at all because its regime has not occurred.

--------------------------------------------------------------------------
WHY IT IS SILENT, WHICH IS THE ACTUAL ROOT CAUSE
--------------------------------------------------------------------------
Every read is `market_data.get(key, <default>)` or `if asset in
market_data.get(key, {})`. **A missing key is therefore indistinguishable
from a measured zero** — the P2 collapse, twelve times over, on a
DECIDE-authority agent. Nothing logs, nothing raises, and the strategy simply
never fires. That is why this survived P215's diagnostic, P217's regime-map
fix, and P358's own investigation: each looked at behaviour ("0 fires") and
not at the boundary.

P310 built the producer/consumer contract for exactly this class and scoped
it, explicitly, to the shadow-ledger boundary — recording that it did not
cover others. This is another one.

--------------------------------------------------------------------------
WHAT THIS FILE DOES
--------------------------------------------------------------------------
Holds the boundary to a contract IN BOTH DIRECTIONS (P310): every key any
strategy reads must be produced, or be declared here with the reason it is
absent. It is a TEST, so it arms nothing — repairing any of these strategies
remains a P141 decision, and the declarations say so.
"""

import ast
import inspect
import pathlib
import textwrap

import pytest

import agents.kraken_quant_agent as kq

SRC = pathlib.Path(kq.__file__)


# ==========================================================================
# Reading the two sides
# ==========================================================================
def _tree():
    return ast.parse(SRC.read_text(encoding="utf-8", errors="replace"))


def produced_keys():
    """What the converter actually emits — read from the PRODUCER, never
    restated (P172), or this guard drifts from the thing it guards."""
    for n in ast.walk(_tree()):
        if isinstance(n, ast.FunctionDef) and n.name == "_convert_market_data":
            for r in ast.walk(n):
                if isinstance(r, ast.Return) and isinstance(r.value, ast.Dict):
                    return {k.value for k in r.value.keys
                            if isinstance(k, ast.Constant)}
    raise AssertionError("could not read _convert_market_data's key set")


def _literal_reads(node):
    out = set()
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "market_data"
                and n.args and isinstance(n.args[0], ast.Constant)):
            out.add(n.args[0].value)
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id == "market_data"
                and isinstance(n.slice, ast.Constant)):
            out.add(n.slice.value)
    return out


def _dynamic_reads(node):
    """`market_data.get(asset)` — a per-asset sub-dict the converter never
    builds. Invisible to a literal-key scan, and it is how DarkPoolVolume
    skips all three assets."""
    return {n.args[0].id for n in ast.walk(node)
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "market_data"
                and n.args and isinstance(n.args[0], ast.Name))}


def strategy_classes():
    out = {}
    for n in _tree().body:
        if isinstance(n, ast.ClassDef) and any(
                getattr(b, "id", getattr(b, "attr", "")) == "BaseStrategy"
                for b in n.bases):
            out[n.name] = n
    return out


# ==========================================================================
# The contract: unproduced keys must be DECLARED, with a reason
# ==========================================================================
# Each entry is (strategy, key) -> reason. A reason must name the severity and
# must NOT read as a to-do: repairing any of these ARMS a DECIDE-authority
# agent on a live account, which is a P141 decision (P358).
UNPRODUCED_BY_DECISION = {
    ("LiquidationCascadeHunter", "cvd"): (
        "OPTIONAL BY DECISION — the read is guarded by `if asset in "
        "market_data.get('cvd', {})` and the site says so verbatim "
        "('optional, graceful fallback if key missing'). Not a defect."),
    ("ETFSpotCointegrationStrategy", "close"): (
        "FATAL — btc_close falls to 0 and the strategy returns on its first "
        "check, so it can never fire (P358c). Repair = P141 arming."),
    ("ETFSpotCointegrationStrategy", "current_price"): (
        "FATAL — the fallback for `close`, equally absent (P358c)."),
    ("ETFSpotCointegrationStrategy", "open"): (
        "FATAL — same strategy, unreachable past the `close` check."),
    ("ETFSpotCointegrationStrategy", "volume"): (
        "FATAL — same strategy, unreachable past the `close` check."),
    ("ETFSpotCointegrationStrategy", "volume_24h"): (
        "FATAL — same strategy, unreachable past the `close` check."),
    ("FundingDivergenceStrategy", "predicted_funding"): (
        "ONE BRANCH DEAD — the smart-money short-build path is gated on "
        "`predicted_funding is not None`, so it can never run; the "
        "strategy's other path is live but warmup-bound (>=240 samples). "
        "[P366] corrects P358c, which filed this as WARMUP only."),
}

# Strategies whose per-asset sub-dict read cannot resolve, same rule.
DYNAMIC_UNPRODUCED_BY_DECISION = {
    "DarkPoolVolumeStrategy": (
        "FATAL — reads market_data[<asset>]['volume'/'close']; the converter "
        "returns a fixed key set with no per-asset sub-dict, so every asset "
        "is skipped (P358). Repair = P141 arming."),
}


def test_every_key_a_strategy_reads_is_produced_or_declared():
    """[P366] THE ROOT-CAUSE GUARD. Four mismatches existed and every one
    failed silently, because a missing key and a measured zero are the same
    value behind `.get(key, 0)` (P2). A fifth would too."""
    produced = produced_keys()
    undeclared = []
    for name, node in sorted(strategy_classes().items()):
        for key in sorted(_literal_reads(node)):
            if key in produced:
                continue
            if (name, key) in UNPRODUCED_BY_DECISION:
                continue
            undeclared.append(f"{name}.{key}")
    assert not undeclared, (
        f"strategies read keys the converter never produces, undeclared: "
        f"{undeclared}. A missing key is indistinguishable from a measured "
        f"zero here (P2), so this fails silently and the strategy just never "
        f"fires. Produce the key, or declare it in UNPRODUCED_BY_DECISION "
        f"with its severity."
    )


def test_every_per_asset_subdict_read_is_declared():
    """The half a literal-key scan cannot see."""
    undeclared = sorted(
        name for name, node in strategy_classes().items()
        if _dynamic_reads(node)
        and name not in DYNAMIC_UNPRODUCED_BY_DECISION)
    assert not undeclared, (
        f"{undeclared} read market_data[<asset>], which the converter never "
        f"builds — declare it or produce it"
    )


def test_no_declaration_is_a_parking_spot():
    """P310/P361's rule, and the one that keeps this honest: a declaration
    must still describe reality (the strategy exists and really does read
    that key) and must state a severity rather than a plan."""
    classes = strategy_classes()
    produced = produced_keys()
    for (name, key), reason in UNPRODUCED_BY_DECISION.items():
        assert name in classes, f"{name} no longer exists — stale declaration"
        assert key in _literal_reads(classes[name]), (
            f"{name} no longer reads {key!r} — delete the declaration rather "
            f"than leave coverage that is not"
        )
        assert key not in produced, (
            f"{key!r} IS now produced — {name} may have come alive at DECIDE "
            f"authority, which is a P141 decision, not a side effect"
        )
        assert any(s in reason for s in ("FATAL", "OPTIONAL", "ONE BRANCH")), (
            f"{name}.{key}: declaration states no severity"
        )
    for name, reason in DYNAMIC_UNPRODUCED_BY_DECISION.items():
        assert name in classes, f"{name} no longer exists"
        assert _dynamic_reads(classes[name]), (
            f"{name} no longer reads a per-asset sub-dict — stale"
        )


def test_the_converter_key_set_is_read_from_the_producer():
    """If the produced set were restated here, the guard would pass while the
    converter changed underneath it — the drift it exists to catch (P172)."""
    src = inspect.getsource(produced_keys)
    assert "_convert_market_data" in src and "ast" in src
    assert "timestamp" not in src, "the key set is hardcoded, not derived"


def test_the_guard_catches_a_NEW_mismatch():
    """Anti-vacuity (P174): a scan that cannot fire reports clean, and clean
    is what a healthy boundary also reports."""
    fake = textwrap.dedent('''
        class NewStrategy(BaseStrategy):
            def generate(self, market_data):
                return market_data.get('a_key_nobody_produces', 0)
    ''')
    node = ast.parse(fake).body[0]
    reads = _literal_reads(node)
    assert "a_key_nobody_produces" in reads
    assert "a_key_nobody_produces" not in produced_keys()


@pytest.mark.parametrize("name,key", [
    ("ETFSpotCointegrationStrategy", "close"),
    ("FundingDivergenceStrategy", "predicted_funding"),
])
def test_the_two_fatal_findings_are_real(name, key):
    """Pin the specific corrections this entry makes to P358c, so they cannot
    quietly stop being true."""
    classes = strategy_classes()
    assert key in _literal_reads(classes[name])
    assert key not in produced_keys()
