#!/usr/bin/env python3
"""
Per-agent live attribution validator (P114 2026-04-27)
================================================================

Reads the last N tick records from signals_*.jsonl and validates the live
authority matrix against CLAUDE.md v6.8 (rows 1-25). For each agent reports:
  (a) writer firing rate (any non-zero direction OR confidence in window)
  (b) authority label matches matrix
  (c) data_quality distribution
  (d) per-asset routing matches docstring

Use as smoke gate BEFORE any matrix refactor + after any agent wiring change.

Usage:
    python scripts/agent_attribution_validate.py <signals_YYYYMMDD.jsonl> [n]

Cloud-side (preferred — runs against live attribution stream):
    ssh hmats "cat /var/lib/docker/volumes/hmats-logs/_data/attribution/signals_*.jsonl > /tmp/all_attr.jsonl && python3 /tmp/agent_validate.py /tmp/all_attr.jsonl 1000"

Exit codes:
    0 = all 16 directional agents present + authorities correct
    1 = structural failure (missing agent, wrong authority label)
    2 = usage error
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# Per CLAUDE.md authority matrix v6.8 (rows 1-25)
EXPECTED = {
    "quant":            "DECIDE",
    "regime":           "CONFIRM",
    "drl":              "DECIDE",
    "sentiment":        "ADVISE",
    "macro":            "CAP",
    "lead_lag":         "EXECUTE",
    "risk":             "VETO",
    "two_stage":        "CONFIRM",
    "short_bias":       "ADVISE",
    "funding":          "ADVISE",
    "onchain":          "ADVISE",
    "llm_sentiment":    "ADVISE",
    "flow":             "ADVISE",
    "structure":        "CONFIRM",
    "squeeze":          "ADVISE",
    "cvd":              "ADVISE",
    "risk_appetite":    "ADVISE",
    "kraken_quant":     "DECIDE",
    "microstructure":   "ADVISE",
    "model_alpha":      "ADVISE",
    "onchain_graph":    "ADVISE",
    "options":          "ADVISE",
    "vol_alpha":        "ADVISE",
    "whale":            "ADVISE",
    "soldex":           "ADVISE",
}

# Stream-name → matrix-name aliases (writer key vs canonical label)
STREAM_ALIASES = {
    "micro": "microstructure",
    # onchain_sol is a SOL-specific FEED, not the onchain_graph AGENT
}

# Agents intentionally absent from attribution stream (non-directional per
# CLAUDE.md "16-agent coverage" — risk/macro/lead_lag/cvd/structure are
# architecturally non-attribution producers).
NON_DIRECTIONAL = {
    "macro", "risk", "structure", "cvd", "lead_lag", "regime",
    "squeeze", "risk_appetite",
}

# Agents that ARE in attribution but emit `*_short_confirmation` /
# `*_data_quality` flags rather than ±direction by architectural design.
# Their direction stays 0.0 by spec — silence here is NOT a bug.
NON_DIRECTIONAL_IN_STREAM = {
    "options",   # options_short_confirmation (PCR sentiment, 0..1 scale)
    "vol_alpha", # always 0 direction; runs via intensity (matrix row 23)
}

# Agents gated by regime — silence is expected in listed regimes.
# Source: main.py:6849 [FIX-SHORT-BIAS] for short_bias.
REGIME_GATED = {
    "short_bias": {"QUIET_ACCUMULATION", "WEAK_CONSOLIDATION", "NEUTRAL_DRIFT"},
}

# Agents disabled by config — silence is expected, operator-controlled.
# Verified at runtime via startup log line "<engine> initialized (enabled=False)".
DISABLED_BY_CONFIG = {
    "onchain": "OnChainSentimentAlphaEngine initialized (enabled=False)",
}

# Agents whose writer requires upstream consensus/payload — silence in
# quiet regimes is the expected behavior, not a wiring bug.
CONSENSUS_GATED = {
    # main.py:6592 — only writes when strategic_check['prior_direction'] is truthy
    "two_stage": "writer at main.py:6592 requires StrategicCoordinator prior_direction",
}

# Per-asset routing — agents legitimately appear only on certain assets
ASSET_ROUTING = {
    "onchain":       {"BTC", "ETH"},     # BTC/ETH on-chain feed
    "onchain_graph": {"SOL"},            # SOL-specific
    "soldex":        {"SOL"},            # SOL-specific (DEX arb)
}


def load_recent(path: Path, n: int = 200) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for ln in lines[-n:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def validate(records: list[dict]) -> int:
    seen: dict = defaultdict(lambda: {"ticks": 0, "fired": 0, "authority": set(),
                                      "dq_zero": 0, "dq_full": 0, "dir_nonzero": 0,
                                      "assets": set()})
    for rec in records:
        asset = rec.get("asset", "?")
        for a in rec.get("signals", rec.get("agents", [])):
            raw_name = a.get("agent_name", "?")
            name = STREAM_ALIASES.get(raw_name, raw_name)
            s = seen[name]
            s["ticks"] += 1
            s["authority"].add(a.get("authority", "?"))
            s["assets"].add(asset)
            dq = a.get("data_quality", 0.0)
            if dq == 0.0:
                s["dq_zero"] += 1
            elif dq >= 0.99:
                s["dq_full"] += 1
            d = a.get("direction", 0.0)
            c = a.get("confidence", 0.0)
            if abs(d) > 0 or c > 0:
                s["fired"] += 1
            if abs(d) > 0:
                s["dir_nonzero"] += 1

    print()
    print(f"{'agent':22s} {'auth':18s} {'expect':10s} {'ticks':>6s} "
          f"{'fired%':>7s} {'dir!=0%':>8s} status")
    print("-" * 92)

    structural_failures = 0  # exit-blocking
    silent_agents: list = []  # warn-only
    missing_directional: list = []

    for name, want_auth in EXPECTED.items():
        if name not in seen:
            if name in NON_DIRECTIONAL:
                # Expected — non-attribution agent
                print(f"{name:22s} {'(non-attr)':18s} {want_auth:10s} "
                      f"{'-':>6s} {'-':>7s} {'-':>8s} OK_NON_DIR")
            else:
                missing_directional.append(name)
                print(f"{name:22s} {'-':18s} {want_auth:10s} "
                      f"{'0':>6s} {'-':>7s} {'-':>8s} MISSING")
                structural_failures += 1
            continue

        s = seen[name]
        auth_set = s["authority"]
        # Authority OK if want_auth is in the set (allows transitional dual-labels)
        auth_ok = want_auth in auth_set
        auth = ",".join(sorted(auth_set)) if auth_set else "-"

        fired_pct = 100.0 * s["fired"] / max(1, s["ticks"])
        dir_pct = 100.0 * s["dir_nonzero"] / max(1, s["ticks"])

        # Per-asset routing check
        expected_assets = ASSET_ROUTING.get(name)
        routing_ok = True
        if expected_assets is not None:
            extra = s["assets"] - expected_assets
            if extra:
                routing_ok = False

        status = "OK"
        if not auth_ok:
            status = f"WRONG_AUTH({auth})"
            structural_failures += 1
        elif not routing_ok:
            status = f"WRONG_ROUTE(saw={s['assets']})"
            structural_failures += 1
        elif s["fired"] == 0:
            if name in NON_DIRECTIONAL_IN_STREAM:
                status = "OK_NON_DIR_INSTR"  # non-directional by architecture
            elif name in REGIME_GATED:
                status = f"OK_REGIME_GATED({','.join(sorted(REGIME_GATED[name]))})"
            elif name in DISABLED_BY_CONFIG:
                status = "OK_DISABLED_BY_CONFIG"
            elif name in CONSENSUS_GATED:
                status = "OK_CONSENSUS_GATED"
            else:
                status = "SILENT_DATA_QUIET"  # writer wired + active, no signal in window
                silent_agents.append(name)

        print(f"{name:22s} {auth:18s} {want_auth:10s} {s['ticks']:>6d} "
              f"{fired_pct:>6.1f}% {dir_pct:>7.1f}% {status}")

    # Surface unaliased unknown agents (not in matrix, not in alias map)
    unknown = sorted(set(seen) - set(EXPECTED) - set(STREAM_ALIASES.values()))
    if unknown:
        print(f"\n[INFO] Stream contains agents not in matrix (likely SOL-specific feeds): {unknown}")

    print()
    print(f"Records analyzed:       {len(records)}")
    print(f"Directional in matrix:  {len(EXPECTED) - len(NON_DIRECTIONAL)}")
    print(f"Found in attribution:   {len(set(seen) & set(EXPECTED))}")
    print(f"Structural failures:    {structural_failures}")
    print(f"Silent (in stream, never fired in window): {len(silent_agents)} {silent_agents}")
    if missing_directional:
        print(f"MISSING directional:    {missing_directional}")

    if structural_failures == 0:
        print("\n[OK] Authority matrix structurally valid — refactor can proceed safely from a wiring standpoint.")
        if silent_agents:
            print(f"[WARN] {len(silent_agents)} agents silent in window — investigate before declaring 'expected behavior' baseline.")
    else:
        print(f"\n[FAIL] {structural_failures} structural failures — DO NOT refactor until resolved.")

    return 0 if structural_failures == 0 else 1


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    records = load_recent(path, n)
    if not records:
        print(f"[ERROR] No records loaded from {path}", file=sys.stderr)
        sys.exit(2)
    sys.exit(validate(records))


if __name__ == "__main__":
    main()
