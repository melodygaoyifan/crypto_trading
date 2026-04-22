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


def load_stats(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print(f"[KQ-DIAG] stats file not found: {path}")
        print("  (Waiting for first 4H heartbeat after instrumentation deploy.)")
        sys.exit(2)
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"[KQ-DIAG] failed to parse {path}: {exc}")
        sys.exit(3)


def render(stats: Dict[str, Any]) -> None:
    ts = stats.get("ts", "?")
    tick = stats.get("tick", "?")
    uptime_h = stats.get("uptime_sec", 0) / 3600.0
    regime_ticks = stats.get("regime_ticks", {}) or {}
    by_regime = stats.get("by_regime", {}) or {}
    never_fired = set(stats.get("never_fired", []) or [])

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
    for regime_name, expected in CANONICAL.items():
        rows = by_regime.get(regime_name, []) or []
        by_name = {r.get("name"): r for r in rows}
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
            if att == 0:
                if rticks == 0:
                    status = "never-active (regime not seen)"
                else:
                    status = "[!] not invoked despite regime active"
            elif fires == 0:
                status = "[DEAD] 0 fires"
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
    print(f"  Summary: {alive}/12 alive,  {dead}/12 never fired while active")
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
