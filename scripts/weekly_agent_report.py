"""Weekly agent performance + VETO rate report → Discord.

Reads:
  - logs/attribution/outcomes_*.jsonl   → per-agent hit-rate, calibration
  - data/shadow_ledger/ledger_*.jsonl   → GATE_REJECT breakdown, true VETO rate

Run inside container:
  docker exec hmats-engine python3 scripts/weekly_agent_report.py [--days 7] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    # shadow ledger uses Infinity literal (non-standard JSON)
                    line = line.replace(": Infinity,", ": 1e12,").replace(": -Infinity,", ": -1e12,")
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return out


def collect_files(base: Path, prefix: str, days: int) -> List[Path]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    files = []
    for p in sorted(base.glob(f"{prefix}*.jsonl")):
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            if mtime >= cutoff:
                files.append(p)
        except Exception:
            continue
    return files


def analyze_agents(outcome_files: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Per-agent hit-rate, avg confidence, sample count, regime breakdown."""
    agent_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "n": 0, "correct": 0, "conf_sum": 0.0, "active_n": 0,
        "by_regime": defaultdict(lambda: {"n": 0, "correct": 0}),
    })

    for path in outcome_files:
        for rec in load_jsonl(path):
            regime = rec.get("regime", "UNKNOWN")
            for o in rec.get("outcomes", []):
                name = o.get("agent_name", "?")
                s = agent_stats[name]
                s["n"] += 1
                dir_val = o.get("direction", 0.0) or 0.0
                conf = o.get("confidence", 0.0) or 0.0
                s["conf_sum"] += conf
                if abs(dir_val) > 0.05:  # "active" = had directional opinion
                    s["active_n"] += 1
                    if o.get("correct"):
                        s["correct"] += 1
                    r = s["by_regime"][regime]
                    r["n"] += 1
                    if o.get("correct"):
                        r["correct"] += 1

    result = {}
    for name, s in agent_stats.items():
        n = s["n"]
        active_n = s["active_n"]
        hit_rate = (s["correct"] / active_n) if active_n > 0 else 0.0
        avg_conf = (s["conf_sum"] / n) if n > 0 else 0.0
        active_pct = (active_n / n * 100) if n > 0 else 0.0
        regime_breakdown = {
            r: {"hit_rate": rs["correct"] / rs["n"] if rs["n"] > 0 else 0.0, "n": rs["n"]}
            for r, rs in s["by_regime"].items() if rs["n"] >= 3
        }
        result[name] = {
            "ticks": n,
            "active_n": active_n,
            "active_pct": active_pct,
            "hit_rate": hit_rate,
            "avg_conf": avg_conf,
            "by_regime": regime_breakdown,
        }
    return result


def analyze_vetos(ledger_files: List[Path]) -> Dict[str, Any]:
    """Precise VETO rate + gate breakdown from shadow ledger."""
    intent_count = 0
    veto_count = 0
    gate_reasons = Counter()
    gate_source = Counter()  # which gate rejected most
    by_asset = defaultdict(lambda: {"intents": 0, "vetos": 0})

    for path in ledger_files:
        for rec in load_jsonl(path):
            entry_type = rec.get("entry_type", "")
            asset = rec.get("asset", "?")
            if entry_type == "INTENT":
                intent_count += 1
                by_asset[asset]["intents"] += 1
            elif entry_type == "GATE_REJECT":
                veto_count += 1
                by_asset[asset]["vetos"] += 1
                reason = (rec.get("data", {}).get("rejection_reason", "") or "")[:80]
                gate_reasons[reason] += 1
                # Extract gate name from reason prefix like "[v3.6.1] Alpha gate:" → "Alpha gate"
                if ":" in reason:
                    gate_part = reason.split(":", 1)[0]
                    gate_part = gate_part.replace("[v3.6.1]", "").replace("[", "").replace("]", "").strip()
                    gate_source[gate_part] += 1

    total = intent_count + veto_count
    veto_rate = (veto_count / total * 100) if total > 0 else 0.0

    return {
        "total_intents": intent_count,
        "total_vetos": veto_count,
        "veto_rate_pct": veto_rate,
        "top_reasons": gate_reasons.most_common(5),
        "by_gate": gate_source.most_common(5),
        "by_asset": dict(by_asset),
    }


def format_report(agent_stats: Dict, veto_stats: Dict, days: int) -> tuple[str, Dict]:
    """Returns (discord_msg, details_dict)."""
    lines = []
    lines.append(f"**HMATS Weekly Agent Report** (last {days}d)")

    # --- VETO section ---
    vr = veto_stats["veto_rate_pct"]
    lines.append(
        f"\n**VETO Stats**  intents={veto_stats['total_intents']}  "
        f"vetos={veto_stats['total_vetos']}  rate={vr:.1f}%"
    )
    if veto_stats["by_gate"]:
        lines.append("Top gates:")
        for gate, cnt in veto_stats["by_gate"][:5]:
            pct = cnt / max(veto_stats["total_vetos"], 1) * 100
            lines.append(f"  • {gate[:40]}: {cnt} ({pct:.0f}%)")

    # --- Agent section ---
    lines.append("\n**Agent Hit-Rates** (|dir|>0.05)")
    # Sort by active_n descending, filter agents with at least 5 active signals
    sorted_agents = sorted(
        [(n, s) for n, s in agent_stats.items() if s["active_n"] >= 5],
        key=lambda x: -x[1]["hit_rate"],
    )
    if sorted_agents:
        for name, s in sorted_agents:
            emoji = "🟢" if s["hit_rate"] >= 0.55 else ("🟡" if s["hit_rate"] >= 0.45 else "🔴")
            lines.append(
                f"  {emoji} {name:<14} hit={s['hit_rate']:.1%} "
                f"active={s['active_pct']:.0f}% (n={s['active_n']}) "
                f"conf={s['avg_conf']:.2f}"
            )
    else:
        lines.append("  (no agents with ≥5 active signals)")

    # --- Flag dead agents ---
    dead = [n for n, s in agent_stats.items() if s["ticks"] >= 10 and s["active_pct"] < 5]
    if dead:
        lines.append(f"\n⚠️ **Dead agents** (active<5%): {', '.join(dead)}")

    msg = "\n".join(lines)

    details = {
        "VETO Rate": f"{vr:.1f}% ({veto_stats['total_vetos']}/{veto_stats['total_intents']+veto_stats['total_vetos']})",
        "Top Gate": (veto_stats["by_gate"][0][0][:40] if veto_stats["by_gate"] else "n/a"),
        "Best Agent": (f"{sorted_agents[0][0]} ({sorted_agents[0][1]['hit_rate']:.1%})" if sorted_agents else "n/a"),
        "Worst Agent": (f"{sorted_agents[-1][0]} ({sorted_agents[-1][1]['hit_rate']:.1%})" if sorted_agents else "n/a"),
    }
    return msg, details


def push_discord(webhook: str, msg: str, details: Dict) -> bool:
    """Send to Discord webhook as embed."""
    try:
        import urllib.request
        fields = [{"name": k, "value": v[:1024], "inline": True} for k, v in details.items()]
        payload = {
            "username": "HMATS",
            "embeds": [{
                "title": "📊 Weekly Agent Report",
                "description": msg[:3900],
                "color": 0x3498DB,
                "fields": fields,
            }],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook, data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "HMATS/6.8 (https://github.com)",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except Exception as e:
        print(f"Discord push failed: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="Look-back window in days")
    parser.add_argument("--dry-run", action="store_true", help="Print only, no Discord push")
    parser.add_argument("--base", type=str, default="/opt/hmats", help="Base path (default /opt/hmats)")
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists():
        base = Path.cwd()  # local fallback

    attr_dir = base / "logs" / "attribution"
    ledger_dir = base / "data" / "shadow_ledger"

    outcome_files = collect_files(attr_dir, "outcomes_", args.days) if attr_dir.exists() else []
    ledger_files = collect_files(ledger_dir, "ledger_", args.days) if ledger_dir.exists() else []

    print(f"Analyzing {len(outcome_files)} outcome files, {len(ledger_files)} ledger files...")

    agent_stats = analyze_agents(outcome_files)
    veto_stats = analyze_vetos(ledger_files)

    msg, details = format_report(agent_stats, veto_stats, args.days)
    print("\n" + "=" * 60)
    print(msg)
    print("\nDetails:", json.dumps(details, indent=2))
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY_RUN] skipping Discord push")
        return 0

    webhook = os.environ.get("DISCORD_WEBHOOK_URL") or os.environ.get("DISCORD_API", "")
    if not webhook:
        print("⚠️ DISCORD_WEBHOOK_URL / DISCORD_API not set — skipping push")
        return 0

    ok = push_discord(webhook, msg, details)
    print(f"Discord push: {'✅ OK' if ok else '❌ FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
