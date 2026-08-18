"""Unified 16-agent 5-DIM audit.

Runs across attribution + shadow ledger to produce per-agent breakdown:
  DIM 1: Latency          (pass-through — not instrumented, same as 8-agent audit)
  DIM 2: VETO stats       (gate-level only; per-agent VETO not recorded today)
  DIM 3: Signal quality   (direction/confidence/active%/data_quality per agent)
  DIM 4: PnL attribution  (from shadow_ledger FILL/CLOSE when primary_agent recorded)
  DIM 5: Ghost detection  (matrix inventory vs actual signals seen)

Agents previously NOT in the attribution tracker will show N=0 until one 4H
tick runs after the attribution expansion (main.py:8299). Those are labelled
PENDING, not DEAD, so the two states aren't confused.

Usage:
    docker exec hmats-trader python -X utf8 scripts/agent_audit_16.py
    python -X utf8 scripts/agent_audit_16.py --base /opt/hmats --days 14
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Canonical 16-agent inventory. Matches main.py:8299-8366 _attr_collected
# and authority_fusion.py AUTHORITY_MATRIX_NORMAL direction-producing subset.
# ---------------------------------------------------------------------------
AGENTS: List[Dict] = [
    # Core (pre-expansion, has historical data)
    {"name": "quant",         "auth": "DECIDE",  "scope": "all", "phase": "core"},
    {"name": "short_bias",    "auth": "ADVISE",  "scope": "all", "phase": "core"},
    {"name": "sentiment",     "auth": "ADVISE",  "scope": "all", "phase": "core"},
    {"name": "micro",         "auth": "TRIGGER", "scope": "all", "phase": "core"},
    {"name": "model_alpha",   "auth": "ADVISE",  "scope": "all", "phase": "core"},
    {"name": "kraken_quant",  "auth": "ADVISE",  "scope": "all", "phase": "core"},
    {"name": "vol_alpha",     "auth": "ADVISE",  "scope": "all", "phase": "core"},
    {"name": "onchain_sol",   "auth": "ADVISE",  "scope": "SOL", "phase": "core"},
    # [ATTR-EXPAND] Newly tracked — historical N=0 expected
    {"name": "drl",           "auth": "ADVISE",  "scope": "all", "phase": "new"},
    {"name": "two_stage",     "auth": "CONFIRM", "scope": "all", "phase": "new"},
    {"name": "funding",       "auth": "ADVISE",  "scope": "all", "phase": "new"},
    {"name": "llm_sentiment", "auth": "ADVISE",  "scope": "all", "phase": "new"},
    {"name": "flow",          "auth": "ADVISE",  "scope": "all", "phase": "new"},
    {"name": "soldex",        "auth": "ADVISE",  "scope": "SOL", "phase": "new"},
    {"name": "onchain_graph", "auth": "ADVISE",  "scope": "SOL", "phase": "new"},
    {"name": "onchain",       "auth": "ADVISE",  "scope": "BTC/ETH", "phase": "new"},
]

# Agents intentionally NOT tracked (architectural non-direction — NOT bugs)
SKIPPED_MATRIX_AGENTS = {
    "regime": "classifier (no direction)",
    "macro": "CAP (leverage ceiling only)",
    "lead_lag": "EXECUTE (timing only)",
    "risk": "VETO (no direction)",
    "structure": "boolean confirmation",
    "squeeze": "safety score",
    "cvd": "divergence (one-sided)",
    "risk_appetite": "subset of macro",
    "options": "asymmetric short-confirm",
    "whale": "magnitude only",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_load_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    try:
        with open(path, "r", errors="ignore", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                line = line.replace(": Infinity,", ": 1e12,").replace(": -Infinity,", ": -1e12,")
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# DIM 2: VETO (gate-level)
# ---------------------------------------------------------------------------
def dim2_veto(ledger_dir: Path, days: int) -> None:
    print("\n" + "=" * 78)
    print(f"  DIM 2  VETO stats  (last {days} days, gate-level)")
    print("=" * 78)
    if not ledger_dir.exists():
        print("  (shadow_ledger dir missing)")
        return
    files = sorted(ledger_dir.glob("ledger_*.jsonl"))[-days:]
    intents = vetos = 0
    gates: Counter = Counter()
    for f in files:
        for rec in _safe_load_jsonl(f):
            et = rec.get("entry_type", "")
            if et == "INTENT":
                intents += 1
            elif et == "GATE_REJECT":
                vetos += 1
                reason = (rec.get("data", {}).get("rejection_reason", "") or "")[:70]
                if ":" in reason:
                    gate = reason.split(":", 1)[0].replace("[v3.6.1]", "").strip()
                    gates[gate] += 1
    total = intents + vetos
    rate = 100 * vetos / max(total, 1)
    print(f"  intents={intents}  vetos={vetos}  rate={rate:.1f}%")
    if gates:
        print(f"\n  top gates:")
        for g, c in gates.most_common(6):
            print(f"    {g[:42]:<42} {c:>5}  ({100*c/max(vetos,1):.0f}%)")
    print("\n  NOTE: per-agent VETO attribution is not recorded today — the system")
    print("  only captures which GATE rejected, not which agent's signal caused it.")


# ---------------------------------------------------------------------------
# DIM 3: Signal quality (per agent)
# ---------------------------------------------------------------------------
def dim3_signal_quality(attr_dir: Path, days: int) -> Dict[str, dict]:
    print("\n" + "=" * 78)
    print(f"  DIM 3  Per-agent signal quality  (last {days} days)")
    print("=" * 78)
    stats: Dict[str, dict] = {a["name"]: {"dirs": [], "confs": [], "dqs": []} for a in AGENTS}
    if not attr_dir.exists():
        print("  (attribution dir missing)")
        return stats
    files = sorted(attr_dir.glob("signals_*.jsonl"))[-days:]
    for f in files:
        for rec in _safe_load_jsonl(f):
            for sig in rec.get("signals", []):
                name = sig.get("agent_name", "")
                if name not in stats:
                    continue
                d = sig.get("direction", 0.0)
                c = sig.get("confidence", 0.0)
                dq = sig.get("data_quality", 0.0)
                if isinstance(d, (int, float)):
                    stats[name]["dirs"].append(float(d))
                if isinstance(c, (int, float)):
                    stats[name]["confs"].append(float(c))
                if isinstance(dq, (int, float)):
                    stats[name]["dqs"].append(float(dq))

    print(f"  {'agent':<15} {'phase':<6} {'auth':<8} {'scope':<8} "
          f"{'N':>6} {'dirMean':>8} {'dirStd':>7} {'act%':>5} {'confμ':>6} {'dqμ':>5} status")
    print("  " + "-" * 92)
    for a in AGENTS:
        name = a["name"]
        s = stats[name]
        n = len(s["dirs"])
        if n == 0:
            status = "PENDING (no history)" if a["phase"] == "new" else "[!] DEAD — tracked but zero output"
            print(f"  {name:<15} {a['phase']:<6} {a['auth']:<8} {a['scope']:<8} "
                  f"{n:>6} {'—':>8} {'—':>7} {'—':>5} {'—':>6} {'—':>5} {status}")
            continue
        dmean = statistics.mean(s["dirs"])
        dstd = statistics.stdev(s["dirs"]) if n > 1 else 0.0
        act = 100 * sum(1 for x in s["dirs"] if abs(x) > 0.01) / n
        cmean = statistics.mean(s["confs"]) if s["confs"] else 0.0
        dqmean = statistics.mean(s["dqs"]) if s["dqs"] else 0.0

        flags = []
        if act < 5 and dqmean < 0.1:
            flags.append("DEAD")
        elif act < 10:
            flags.append("rare")
        if dstd < 0.05 and n > 20:
            flags.append("stuck")
        if cmean < 0.2 and act > 30:
            flags.append("low-conf")
        status = " ".join(flags) if flags else "OK"
        print(f"  {name:<15} {a['phase']:<6} {a['auth']:<8} {a['scope']:<8} "
              f"{n:>6} {dmean:>+8.3f} {dstd:>7.3f} {act:>4.0f}% {cmean:>6.3f} {dqmean:>5.2f} {status}")

    print("\n  legend: phase=core → historic data; new → post-expand (N=0 expected until next tick)")
    return stats


# ---------------------------------------------------------------------------
# DIM 4: Per-agent PnL / hit-rate
# ---------------------------------------------------------------------------
def dim4_pnl(attr_dir: Path, ledger_dir: Path, days: int) -> None:
    print("\n" + "=" * 78)
    print(f"  DIM 4  Per-agent hit-rate (from attribution outcomes)")
    print("=" * 78)
    if not attr_dir.exists():
        print("  (attribution dir missing)")
    else:
        out_files = sorted(attr_dir.glob("outcomes_*.jsonl"))[-days:]
        per: Dict[str, dict] = defaultdict(lambda: {"n": 0, "active": 0, "correct": 0})
        for f in out_files:
            for rec in _safe_load_jsonl(f):
                for o in rec.get("outcomes", []):
                    name = o.get("agent_name", "")
                    d = float(o.get("direction", 0.0) or 0.0)
                    s = per[name]
                    s["n"] += 1
                    if abs(d) > 0.05:
                        s["active"] += 1
                        if o.get("correct"):
                            s["correct"] += 1
        print(f"  {'agent':<15} {'phase':<6} {'N':>6} {'active':>8} {'hit':>8}  verdict")
        print("  " + "-" * 60)
        for a in AGENTS:
            name = a["name"]
            s = per.get(name, {"n": 0, "active": 0, "correct": 0})
            n = s["n"]; ac = s["active"]; co = s["correct"]
            if n == 0:
                verdict = "PENDING" if a["phase"] == "new" else "no data"
                print(f"  {name:<15} {a['phase']:<6} {n:>6} {ac:>8} {'—':>8}  {verdict}")
                continue
            if ac < 5:
                verdict = "low-sample"
                hr_s = "—"
            else:
                hr = co / ac
                hr_s = f"{hr*100:.1f}%"
                if hr >= 0.55:
                    verdict = "GOOD"
                elif hr >= 0.45:
                    verdict = "random"
                else:
                    verdict = "BAD (worse than coin)"
            print(f"  {name:<15} {a['phase']:<6} {n:>6} {ac:>8} {hr_s:>8}  {verdict}")

    # Shadow-ledger FILL PnL grouped by primary_agent (if recorded)
    print("\n  --- shadow_ledger PnL grouped by primary_agent ---")
    if not ledger_dir.exists():
        print("  (no shadow_ledger)")
        return
    pnl_per: Dict[str, List[float]] = defaultdict(list)
    files = sorted(ledger_dir.glob("ledger_*.jsonl"))[-days:]
    for f in files:
        for rec in _safe_load_jsonl(f):
            if rec.get("entry_type") not in ("FILL", "CLOSE", "EXIT"):
                continue
            d = rec.get("data", {})
            pnl = d.get("realized_pnl") or d.get("net_pnl") or d.get("pnl")
            pa = d.get("primary_agent") or d.get("signal_source") or d.get("decision_source")
            if isinstance(pnl, (int, float)) and pa:
                pnl_per[str(pa)].append(float(pnl))
    if not pnl_per:
        print("  (no FILL records tag primary_agent — $-PnL attribution not yet wired)")
    else:
        print(f"  {'primary_agent':<20} {'trades':>7} {'total $':>12} {'avg $':>10} {'win%':>6}")
        for agent, pnls in sorted(pnl_per.items(), key=lambda x: -sum(x[1])):
            tot = sum(pnls); avg = tot / len(pnls)
            wr = 100 * sum(1 for p in pnls if p > 0) / len(pnls)
            print(f"  {agent:<20} {len(pnls):>7} {tot:>+12.2f} {avg:>+10.2f} {wr:>5.0f}%")


# ---------------------------------------------------------------------------
# DIM 5: Ghost / coverage inventory
# ---------------------------------------------------------------------------
def dim5_ghost(attr_dir: Path, days: int, fusion_src: Path) -> None:
    print("\n" + "=" * 78)
    print("  DIM 5  Ghost / coverage inventory")
    print("=" * 78)
    # Parse matrix
    matrix: Dict[str, str] = {}
    if fusion_src.exists():
        try:
            src = fusion_src.read_text(errors="ignore", encoding="utf-8")
            m = re.search(r"AUTHORITY_MATRIX_NORMAL\s*=\s*\{(.*?)\n\}", src, re.DOTALL)
            if m:
                for name, auth in re.findall(r'["\']([a-z_]+)["\']\s*:\s*Authority\.(\w+)', m.group(1)):
                    matrix[name] = auth
        except Exception:
            pass
    writers: set = set()
    if attr_dir.exists():
        for f in sorted(attr_dir.glob("signals_*.jsonl"))[-days:]:
            for rec in _safe_load_jsonl(f):
                for sig in rec.get("signals", []):
                    n = sig.get("agent_name", "")
                    if n:
                        writers.add(n)
    tracked_names = {a["name"] for a in AGENTS}
    print(f"  authority_matrix entries:        {len(matrix):>3}")
    print(f"  direction-producing (tracked):   {len(tracked_names):>3}")
    print(f"  architectural non-direction:     {len(SKIPPED_MATRIX_AGENTS):>3}")
    print(f"  seen writing signals (historic): {len(writers):>3}")

    expected_in_attr = tracked_names
    present = expected_in_attr & writers
    absent = expected_in_attr - writers
    surprise = writers - expected_in_attr - set(matrix.keys())

    print(f"\n  tracked + seen:   {sorted(present)}")
    print(f"\n  tracked + absent: {sorted(absent)}")
    print("    (new-phase agents here = PENDING, fire from next tick onward)")
    if surprise:
        print(f"\n  unexpected writers (not in matrix, not tracked): {sorted(surprise)}")

    print(f"\n  matrix agents skipped on purpose (non-direction):")
    for n, reason in SKIPPED_MATRIX_AGENTS.items():
        in_matrix = "in-matrix" if n in matrix else "NOT-in-matrix"
        print(f"    {n:<18} {in_matrix:<14} — {reason}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/opt/hmats",
                    help="HMATS root (default: /opt/hmats for container)")
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()
    base = Path(args.base)
    attr = base / "logs" / "attribution"
    ledger = base / "data" / "shadow_ledger"
    fusion = base / "signals" / "authority_fusion.py"

    print("=" * 78)
    print(f"  HMATS 16-Agent 5-DIM Audit")
    print(f"  base={base}  window={args.days}d")
    print("=" * 78)
    print("\n  DIM 1  Latency: per-agent timing is NOT instrumented (known gap).")
    print("  → total tick time available via [HEARTBEAT] logs; per-agent latency unavailable.")

    dim2_veto(ledger, args.days)
    dim3_signal_quality(attr, args.days)
    dim4_pnl(attr, ledger, args.days)
    dim5_ghost(attr, args.days, fusion)

    print("\n" + "=" * 78)
    print("  Done. PENDING rows will start populating after the next 4H tick")
    print("  once main.py:8299 (expanded _attr_collected) runs live.")
    print("=" * 78)


if __name__ == "__main__":
    main()
