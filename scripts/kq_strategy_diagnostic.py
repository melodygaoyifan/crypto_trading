"""Kraken-Quant 12-strategy firing diagnostic.

Reads data/kq_firing_stats.json (persisted each 4H heartbeat by main.py) and
reports which of the 12 internal strategies have fired vs never fired, grouped
by their regime bucket.

Usage (on cloud, from host):
    docker exec hmats-trader python -X utf8 scripts/kq_strategy_diagnostic.py

Usage (local):
    python -X utf8 scripts/kq_strategy_diagnostic.py
    python -X utf8 scripts/kq_strategy_diagnostic.py --path /opt/hmats/data/kq_firing_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Canonical 12 strategies (must match agents/kraken_quant_agent.py:2118-2137)
CANONICAL = {
    "BEAR": [
        "LiquidationCascadeHunter",
        "HurstExponentStrategy",
        "ShannonEntropyStrategy",
        "VarianceRiskPremiumStrategy",
    ],
    "BULL": [
        "FundingDivergenceStrategy",
        "ETFSpotCointegrationStrategy",
        "RelativeStrengthStrategy",
        "OrderBookImbalanceStrategy",
    ],
    "SIDEWAYS": [
        "KalmanCointegrationStrategy",
        "OrnsteinUhlenbeckStrategy",
        "DarkPoolVolumeStrategy",
        "DeltaNeutralFundingStrategy",
    ],
}


KNOWN_SILENT_CAUSES = {
    # [P358] Measured causes for reachable strategies that never fire. A
    # cause here is NOT permission to repair it: every one of these would let
    # a DECIDE-authority agent start emitting directions on a live account,
    # which is a P141 activation. `tests/test_p358_kraken_quant_silence.py`
    # pins the underlying defects, so an entry that goes stale fails there
    # rather than quietly misdescribing a strategy that has been fixed.
    # KEYS ARE THE NAMES THE AGENT REPORTS, not the class names — my first
    # cut used the class names and silently covered 2 of 4 (P310/P2), found
    # only by running this against real producer output (P264/P309).
    "OrderBookImbalance": (
        "STRUCTURAL",
        "cannot fire at ANY threshold: main.py sets bid_depth = ask_depth = "
        "orderbook_depth_1pct_usd/2, so (bid-ask)/(bid+ask) is identically 0 "
        "(P358). Repair = P141 arming."),
    "DarkPoolVolumeStrategy": (
        "STRUCTURAL",
        "reads market_data[asset]['volume'/'close']; the converter returns a "
        "fixed key set with no per-asset sub-dict, so both are 0 and every "
        "asset is skipped (P358/P2). Repair = P141 arming."),
    "ETFSpotCointegration": (
        "STRUCTURAL",
        "reads market_data['close'/'open'/'volume_24h']; the converter "
        "produces none of them, so btc_close==0 and it returns on its first "
        "check (P358c/P2). Repair = P141 arming."),
    # [P358c] WARMUP clocks measured, not assumed. Buffers are fed ~3x per 4H
    # tick (generate_signal runs once per asset and appends for all three),
    # were in-memory with no persistence — a deploy reset them (P301/P316).
    # [P390] RelativeStrength + KalmanCointegration now PERSIST via
    # strategies/_warmup_state, so their clocks are CUMULATIVE uptime;
    # FundingDivergence remains unpersisted. Days = samples/3 ticks x 4h.
    "RelativeStrengthStrategy": (
        "WARMUP",
        ">=50 price samples ~= 17 ticks ~= 2.8 days of CUMULATIVE uptime "
        "(P358c clock; buffers persisted across restarts since P390)."),
    "KalmanCointegration_SOL_ETH": (
        "WARMUP",
        ">=50 price AND >=30 spread samples (the spread only accrues once "
        "prices are full) ~= 27 ticks ~= 4.5 days of CUMULATIVE uptime "
        "(P358c clock; buffers + Kalman state persisted since P390)."),
    "FundingDivergenceStrategy": (
        "WARMUP",
        ">=240 price samples ~= 80 ticks ~= 13.3 days of uninterrupted "
        "uptime — the longest clock of the six (P358c/P301/P316)."),
}


def load_stats(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print(f"[KQ-DIAG] stats file not found: {path}")
        print("  (Waiting for first 4H heartbeat after instrumentation deploy.)")
        sys.exit(2)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[KQ-DIAG] failed to parse {path}: {exc}")
        sys.exit(3)


# [P215] Regime enum VALUE -> NAME. The agent used to key its telemetry by value
# ('chop') while this script looked it up by name ('SIDEWAYS'), so every lookup
# missed and the whole table printed fabricated zeros. The agent now writes
# names; this map keeps stats files written before that fix readable, so an old
# snapshot reports what it actually recorded instead of silently reading as
# "everything dead" — which is the very failure being fixed.
_VALUE_TO_NAME = {"bear": "BEAR", "bull": "BULL", "chop": "SIDEWAYS"}


def _by_name(d: Dict[str, Any]) -> Dict[str, Any]:
    return {_VALUE_TO_NAME.get(str(k), str(k)): v for k, v in d.items()}


def render(stats: Dict[str, Any]) -> None:
    ts = stats.get("ts", "?")
    tick = stats.get("tick", "?")
    uptime_h = stats.get("uptime_sec", 0) / 3600.0
    regime_ticks = _by_name(stats.get("regime_ticks", {}) or {})
    by_regime = _by_name(stats.get("by_regime", {}) or {})
    never_fired = set(stats.get("never_fired", []) or [])
    archived = set(stats.get("archived", []) or [])

    total_ticks = sum(regime_ticks.values()) or 1
    print("=" * 72)
    print("  Kraken-Quant 12-Strategy Firing Diagnostic")
    print("=" * 72)
    print(f"  snapshot: ts={ts}  tick={tick}  uptime={uptime_h:.1f}h")
    print(f"  regime ticks: " + " ".join(
        f"{k}={v} ({100*v/total_ticks:.0f}%)" for k, v in sorted(regime_ticks.items())
    ))
    print("")

    alive = 0
    dead = 0
    for regime_name, canonical_names in CANONICAL.items():
        rows = by_regime.get(regime_name, []) or []
        by_name = {r.get("name"): r for r in rows}
        # [P215] Iterate the names the AGENT reported, not this hardcoded list.
        # Three runtime names differ from CANONICAL — KalmanCointegration_SOL_ETH
        # vs KalmanCointegrationStrategy, ETFSpotCointegration(+Strategy),
        # OrderBookImbalance(+Strategy) — so looking up by the literal missed and
        # printed "[!] not invoked despite regime active" for a strategy that HAD
        # been invoked. Exactly the mismatch this entry already fixed one level
        # up; a hardcoded mirror of a runtime list drifts. CANONICAL is kept only
        # to notice a bucket that lost a strategy entirely.
        expected = [r.get("name") for r in rows] or canonical_names
        _missing = [n for n in canonical_names
                    if n not in by_name and not any(
                        n.rstrip("Strategy") in str(k) for k in by_name)]
        if rows and _missing:
            print(f"  [!] expected in {regime_name} but absent from the agent's "
                  f"report: {_missing}")
        rticks = regime_ticks.get(regime_name, 0)
        bucket_hit = "*" if rticks == 0 else ""
        print(f"  [{regime_name}]   regime_ticks={rticks}{bucket_hit}")
        print(f"  {'strategy':<34} {'attempts':>9} {'fires':>7} {'fire%':>7}  status")
        print("  " + "-" * 68)
        for s_name in expected:
            r = by_name.get(s_name) or {}
            att = int(r.get("attempts", 0))
            fires = int(r.get("fires", 0))
            rate = (100.0 * fires / att) if att > 0 else 0.0
            if s_name in archived:
                # [P215] A P157 decision, not a fault. Previously these were
                # indistinguishable from dead strategies (both attempts=0).
                status = "ARCHIVED (P157 decision — cannot fire by design)"
            elif att == 0:
                if rticks == 0:
                    status = "never-active (regime not seen)"
                else:
                    status = "[!] not invoked despite regime active"
            elif fires == 0:
                # [P358] "0 fires" had THREE distinct causes and one label, so
                # a reader could not tell a strategy that CANNOT fire from one
                # that is starved from one that simply declined — the P199/P216
                # conflation, in the tool built to explain the silence. Naming
                # the known ones leaves the residue genuinely unexplained,
                # which is the set worth investigating.
                _cause = KNOWN_SILENT_CAUSES.get(s_name)
                status = f"[{_cause[0]}] {_cause[1]}" if _cause else (
                    "[DEAD] 0 fires — cause UNKNOWN, worth a trace")
                dead += 1
            elif rate < 1.0:
                status = "[LOW] <1% fire rate"
                alive += 1
            else:
                status = "[ALIVE]"
                alive += 1
            print(f"  {s_name:<34} {att:>9} {fires:>7} {rate:>6.1f}%  {status}")
        print("")

    print("-" * 72)
    # [P215] Report against the strategies that CAN fire, not a flat 12. With 4
    # archived and the market in one regime bucket, "x/12" understates by
    # counting strategies that are excluded by design or unreachable by regime.
    _reachable = [r.get("name")
                  for b, rows in by_regime.items() if regime_ticks.get(b, 0) > 0
                  for r in (rows or []) if r.get("name") not in archived]
    print(f"  Summary: {alive} alive, {dead} never fired while active")
    print(f"  Of 12 strategies: {len(archived)} archived (P157), "
          f"{len(_reachable)} reachable in the regimes actually seen "
          f"{sorted(b for b in CANONICAL if regime_ticks.get(b, 0) > 0)}")
    if _reachable:
        print(f"  Reachable now: {sorted(_reachable)}")
    if never_fired:
        print(f"  Never fired: {sorted(never_fired)}")
    # Hint for regimes we never reached
    missing_buckets = [b for b in CANONICAL if regime_ticks.get(b, 0) == 0]
    if missing_buckets:
        print(f"  Regime buckets never active: {missing_buckets}")
        print("  (Strategies in those buckets have had ZERO chance to fire.)")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        default="data/kq_firing_stats.json",
        help="Path to persisted firing stats (default: data/kq_firing_stats.json)",
    )
    args = ap.parse_args()
    stats = load_stats(Path(args.path))
    render(stats)


if __name__ == "__main__":
    main()
