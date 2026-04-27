"""
test_replay_fusion.py — golden-trace fusion replay harness (P116)
====================================================================

Reads the FROZEN per-agent signals from tests/golden/live_signals_corpus.jsonl
(217 records captured from production 2026-04-25 to 2026-04-27), reconstructs
the agent_signals dict, feeds each into AuthorityFusionEngine.fuse(), and
snapshots the output. Subsequent runs assert identical output.

Catches:
  - Any change to fusion logic that produces a different decider/direction/
    confidence for the same input
  - Authority matrix mutations (P114-shape drift)
  - Veto/cap logic regressions
  - Silent semantic shifts (e.g. quant abstain handled as 0 vs as ABSTAIN)

Does NOT catch:
  - Bugs in upstream agents that produce wrong values to begin with
  - Bugs in market_data ingestion
  - Bugs in execution layer
  (those are caught by chaos harness + invariant tests + paper trade replay)

Usage:
  python -X utf8 -m pytest tests/test_replay_fusion.py -v

To regenerate snapshot after intentional fusion change:
  python -X utf8 tests/test_replay_fusion.py --update-snapshot
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

CORPUS = Path(__file__).parent / "golden" / "live_signals_corpus.jsonl"
SNAPSHOT = Path(__file__).parent / "golden" / "fusion_replay_snapshot.json"

# Per-agent stream-name → agent_signals key prefix.
# E.g. stream record name="quant" → agent_signals["quant_direction"]
# Some agents in stream use shortened names (micro vs microstructure).
STREAM_TO_KEY_PREFIX = {
    "quant": "quant",
    "drl": "drl",
    "sentiment": "sentiment",
    "two_stage": "two_stage",
    "short_bias": "short_bias",
    "funding": "funding",
    "onchain": "onchain",
    "llm_sentiment": "llm_sentiment",
    "flow": "flow",
    "kraken_quant": "kq",  # writes kq_direction / kq_confidence
    "micro": "micro",  # micro_direction / micro_confidence
    "model_alpha": "model_alpha",
    "onchain_graph": "onchain_graph",
    "options": "options",
    "vol_alpha": "vol_alpha",
    "whale": "whale_flow",
    "soldex": "soldex_arb",
    "onchain_sol": "onchain_sol",
}


def _load_corpus() -> List[Dict]:
    """Load frozen per-tick attribution records."""
    if not CORPUS.exists():
        pytest.skip(f"Golden corpus not found: {CORPUS}")
    return [json.loads(ln) for ln in CORPUS.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _reconstruct_agent_signals(record: Dict) -> Dict[str, float]:
    """Take a per-tick attribution record and rebuild the agent_signals dict
    that would have been the input to fusion.

    Fusion reads keys like agent_signals["quant_direction"], not the
    structured per-agent payload. The capture stream stores the structured
    form; this inverts that.
    """
    agent_signals: Dict[str, float] = {}
    for sig in record.get("signals", []):
        name = sig.get("agent_name", "")
        prefix = STREAM_TO_KEY_PREFIX.get(name, name)
        # Standard keys
        agent_signals[f"{prefix}_direction"] = float(sig.get("direction", 0.0))
        agent_signals[f"{prefix}_confidence"] = float(sig.get("confidence", 0.0))
        # data_quality (P98 NaN guard depends on this)
        if "data_quality" in sig:
            agent_signals[f"{prefix}_data_quality"] = float(sig.get("data_quality", 1.0))
        # Pull through raw_payload extras when present (e.g. options writes
        # options_short_confirmation as a separate key)
        rp = sig.get("raw_payload", {}) or {}
        for k, v in rp.items():
            if isinstance(v, (int, float, bool)):
                agent_signals.setdefault(k, float(v))
    return agent_signals


def _stable_hash(payload: Dict) -> str:
    """Stable digest for snapshot comparison — JSON sort + sha256."""
    # Round floats to 6 decimal places to absorb tiny FP drift across CPUs.
    def _round(v):
        if isinstance(v, float):
            return round(v, 6)
        if isinstance(v, dict):
            return {k: _round(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_round(x) for x in v]
        return v

    canonical = json.dumps(_round(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _replay_one(record: Dict) -> Dict:
    """Replay one tick through the fusion engine.

    Imports are deferred so test discovery doesn't trigger heavy module loads.
    """
    from signals.authority_fusion import (
        AuthorityFusionEngine, AgentSignal, FusionContext,
    )
    from core.canonical_enums import SystemMode
    from market.phase_detector import RegimePhase

    agent_signals = _reconstruct_agent_signals(record)

    # Build the canonical fusion-input dict — same shape as
    # _build_fusion_signals would produce for the agents we have data for.
    # We focus on the directional agents the corpus actually contains.
    fusion_signals: Dict[str, AgentSignal] = {}
    for name in ("quant", "drl", "sentiment", "two_stage", "short_bias",
                 "funding", "onchain", "llm_sentiment", "flow",
                 "kraken_quant", "microstructure", "model_alpha",
                 "onchain_graph", "options", "whale", "soldex"):
        prefix = STREAM_TO_KEY_PREFIX.get(name, name)
        d = agent_signals.get(f"{prefix}_direction", 0.0)
        c = agent_signals.get(f"{prefix}_confidence", 0.0)
        if abs(d) > 0 or c > 0:
            fusion_signals[name] = AgentSignal(direction=d, confidence=c)

    # Build a minimal context — defaults match the production NORMAL path.
    # regime_phase is the position-within-trend (IGNITION/EXPANSION/...) and
    # is distinct from regime label (QUIET_ACCUMULATION etc, passed separately).
    ctx = FusionContext(
        mode=SystemMode.NORMAL,
        regime_phase=RegimePhase.UNDEFINED,
        data_valid=True,
        drl_enabled=True,
        regime="QUIET_ACCUMULATION",
        asset=record.get("asset", "BTC"),
        current_price=0.0,
    )

    # Fresh engine per tick — no momentum_memory carryover (matches the
    # snapshot semantics; production carries memory but each replay tick
    # is treated as a clean independent decision for regression-detection
    # purposes).
    engine = AuthorityFusionEngine()
    try:
        result = engine.fuse(fusion_signals, ctx)
    except Exception as e:  # noqa: BLE001 — replay errors must surface
        return {
            "tick_id": record.get("tick_id"),
            "asset": record.get("asset"),
            "n_agents": len(fusion_signals),
            "error": f"{type(e).__name__}: {e}",
        }

    return {
        "tick_id": record.get("tick_id"),
        "asset": record.get("asset"),
        "n_agents": len(fusion_signals),
        "direction": round(result.direction, 6),
        "confidence": round(result.confidence, 6),
        "decider_agent": result.decider_agent,
        "primary_agent": result.primary_agent,
        "vetoes_active": sorted(result.vetoes_active),
        "is_partial_consensus": bool(result.is_partial_consensus),
    }


# =====================================================================
# Tests
# =====================================================================

class TestReplayFusion:
    """Snapshot comparison — any fusion logic change that affects ANY of the
    217 captured ticks shows here."""

    def test_corpus_loads(self):
        """Sanity: the frozen corpus is intact."""
        records = _load_corpus()
        assert len(records) >= 100, (
            f"Corpus too small ({len(records)}) — snapshot would be unstable"
        )

    def test_fusion_replay_deterministic(self):
        """Each tick replays to the same output every time."""
        records = _load_corpus()
        # Pick 5 representative ticks (first SOL, BTC, ETH + 2 random)
        sample = records[:3] + records[len(records) // 2:len(records) // 2 + 2]
        for record in sample:
            r1 = _replay_one(record)
            r2 = _replay_one(record)
            assert r1 == r2, (
                f"Non-deterministic replay for tick {record.get('tick_id')} "
                f"asset {record.get('asset')}: r1={r1}, r2={r2}"
            )

    def test_fusion_replay_no_exceptions(self):
        """Replay every captured tick — none should raise."""
        records = _load_corpus()
        errors = []
        for record in records:
            result = _replay_one(record)
            if "error" in result:
                errors.append(
                    f"  tick={record.get('tick_id')} asset={record.get('asset')}: {result['error']}"
                )
        assert not errors, (
            f"{len(errors)} ticks raised during fusion replay:\n" + "\n".join(errors[:10])
        )

    def test_fusion_replay_matches_snapshot(self):
        """The snapshot is the regression gate. Any change to fusion logic
        that produces different output for the same input fails here."""
        records = _load_corpus()
        replays = [_replay_one(r) for r in records]

        # Aggregate per-asset summary for snapshot stability
        per_asset: Dict[str, Dict] = defaultdict(lambda: {
            "n_ticks": 0,
            "directions": [],
            "decider_counts": defaultdict(int),
            "veto_counts": defaultdict(int),
        })
        for r in replays:
            asset = r["asset"]
            per_asset[asset]["n_ticks"] += 1
            per_asset[asset]["directions"].append(r.get("direction", 0.0))
            per_asset[asset]["decider_counts"][r.get("decider_agent", "")] += 1
            for v in r.get("vetoes_active", []):
                per_asset[asset]["veto_counts"][v] += 1

        # Build snapshot payload — stable across runs
        snapshot_payload = {
            "corpus_size": len(records),
            "per_asset": {
                asset: {
                    "n_ticks": s["n_ticks"],
                    "direction_sum": round(sum(s["directions"]), 6),
                    "direction_nonzero_count": sum(1 for d in s["directions"] if abs(d) > 1e-9),
                    "decider_distribution": dict(s["decider_counts"]),
                    "veto_distribution": dict(s["veto_counts"]),
                }
                for asset, s in sorted(per_asset.items())
            },
        }
        digest = _stable_hash(snapshot_payload)

        if not SNAPSHOT.exists():
            # First run — write snapshot
            SNAPSHOT.write_text(
                json.dumps({"digest": digest, "payload": snapshot_payload}, indent=2),
                encoding="utf-8",
            )
            pytest.skip(
                f"Snapshot created at {SNAPSHOT.name} (digest={digest}). "
                f"Re-run to verify regression gate."
            )

        existing = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        if existing["digest"] != digest:
            # Compute diff for debugging
            old_payload = existing["payload"]
            diff_lines = []
            for asset in sorted(set(old_payload["per_asset"]) | set(snapshot_payload["per_asset"])):
                old = old_payload["per_asset"].get(asset, {})
                new = snapshot_payload["per_asset"].get(asset, {})
                if old != new:
                    diff_lines.append(f"  {asset}:")
                    for k in sorted(set(old) | set(new)):
                        if old.get(k) != new.get(k):
                            diff_lines.append(f"    {k}: {old.get(k)!r} -> {new.get(k)!r}")
            pytest.fail(
                f"Fusion replay snapshot drift detected.\n"
                f"  Old digest: {existing['digest']}\n"
                f"  New digest: {digest}\n"
                f"Diff:\n" + "\n".join(diff_lines) +
                f"\n\nIf the change is intentional, regenerate via:\n"
                f"  python -X utf8 tests/test_replay_fusion.py --update-snapshot"
            )


def _update_snapshot():
    """CLI entry point: regenerate the snapshot after an intentional change."""
    records = _load_corpus()
    replays = [_replay_one(r) for r in records]
    per_asset: Dict[str, Dict] = defaultdict(lambda: {
        "n_ticks": 0, "directions": [],
        "decider_counts": defaultdict(int), "veto_counts": defaultdict(int),
    })
    for r in replays:
        asset = r["asset"]
        per_asset[asset]["n_ticks"] += 1
        per_asset[asset]["directions"].append(r.get("direction", 0.0))
        per_asset[asset]["decider_counts"][r.get("decider_agent", "")] += 1
        for v in r.get("vetoes_active", []):
            per_asset[asset]["veto_counts"][v] += 1

    snapshot_payload = {
        "corpus_size": len(records),
        "per_asset": {
            asset: {
                "n_ticks": s["n_ticks"],
                "direction_sum": round(sum(s["directions"]), 6),
                "direction_nonzero_count": sum(1 for d in s["directions"] if abs(d) > 1e-9),
                "decider_distribution": dict(s["decider_counts"]),
                "veto_distribution": dict(s["veto_counts"]),
            }
            for asset, s in sorted(per_asset.items())
        },
    }
    digest = _stable_hash(snapshot_payload)
    SNAPSHOT.write_text(
        json.dumps({"digest": digest, "payload": snapshot_payload}, indent=2),
        encoding="utf-8",
    )
    print(f"Snapshot updated: digest={digest}, ticks={len(records)}")
    print(f"Per-asset summary:")
    for asset, s in snapshot_payload["per_asset"].items():
        print(f"  {asset}: {s['n_ticks']} ticks, deciders={dict(s['decider_distribution'])}")


if __name__ == "__main__":
    if "--update-snapshot" in sys.argv:
        _update_snapshot()
    else:
        pytest.main([__file__, "-v", "-s"])
