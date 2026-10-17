"""[P1-7] Phase 1 shadow-period report -> reports/HMATS_PHASE1_SHADOW.md

    docker exec hmats-engine python -X utf8 scripts/phase1_shadow_report.py \
        --log /opt/hmats/logs/hmats.log --out /opt/hmats/data/evidence_reports/HMATS_PHASE1_SHADOW.md
    python -X utf8 scripts/phase1_shadow_report.py --fills-json fills.json --data-dir <pulled data> --log <pulled hmats.log>

One section per shadow filter:
  * [P1-2] realized friction gate — data/realized_friction_shadow.jsonl
  * [P1-4] contract floor         — "[P1-4-FLOOR]" lines in the engine log
For each: WOULD_SUPPRESS count, and for the round trips the suppressed
decisions actually opened (entries within 60 min after a WOULD_SUPPRESS tick,
same asset): sum commission and sum realized gross PnL, via the Phase 0
round-trip model (analytics/unit_economics/roundtrips.py).

SWITCH-TO-ENFORCE CRITERION (Phase 1 spec, pre-committed): enforce only if the
suppressed fills' sum realized gross PnL < their sum commission (they were net
losers). Otherwise the verdict is KEEP_SHADOW with the reason. Fewer than 48h of
shadow data -> NOT_YET. The report never flips a flag (P141).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _shadow_rows(data_dir):
    rows = []
    try:
        with open(os.path.join(data_dir, "realized_friction_shadow.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except ValueError:  # noqa: silent-swallow — torn line skipped
                    continue
    except FileNotFoundError:  # noqa: silent-swallow — absence is reported as zero shadow rows
        pass
    return rows


_FLOOR = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[P1-4-FLOOR\] (\w+): .*WOULD_SUPPRESS")


def _floor_rows(log_path):
    rows = []
    if not log_path:
        return rows
    for p in sorted({log_path, log_path + ".1", log_path + ".2"}):
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    m = _FLOOR.search(line)
                    if m:
                        t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(
                            tzinfo=dt.timezone.utc).timestamp()
                        rows.append({"ts": t, "asset": m.group(2), "verdict": "WOULD_SUPPRESS"})
        except FileNotFoundError:  # noqa: silent-swallow — rotated file absent
            continue
    return rows


def _attribute(rows, rts):
    sup = [r for r in rows if r.get("verdict") == "WOULD_SUPPRESS"]
    ids = {}
    for r in sup:
        for i, rt in enumerate(rts):
            if rt["asset"] == r["asset"] and r["ts"] <= rt["entry_ts"] <= r["ts"] + 3600:
                ids[i] = rt
    hit = list(ids.values())
    return dict(would_suppress=len(sup), round_trips=len(hit),
                sum_commission_usd=sum(x["fee_usd"] for x in hit),
                sum_realized_gross_usd=sum(x["gross_pnl_usd"] for x in hit))


def _verdict(stat, hours):
    if hours < 48:
        return "NOT_YET", f"only {hours:.1f}h of shadow data (< 48h)"
    if stat["round_trips"] == 0:
        return "KEEP_SHADOW", "no suppressed decision opened a round trip — nothing to judge"
    if stat["sum_realized_gross_usd"] < stat["sum_commission_usd"]:
        return "ENFORCE_ELIGIBLE", ("suppressed fills were net losers: gross "
                                    f"${stat['sum_realized_gross_usd']:.2f} < commission "
                                    f"${stat['sum_commission_usd']:.2f} (operator decision)")
    return "KEEP_SHADOW", ("suppressed fills earned more than their commission: gross "
                           f"${stat['sum_realized_gross_usd']:.2f} >= ${stat['sum_commission_usd']:.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("HMATS_DATA_DIR", os.path.join(REPO, "data")))
    ap.add_argument("--log", default=None)
    ap.add_argument("--fills-json", default=None)
    ap.add_argument("--since", default=None, help="ISO start of the shadow period (default: first shadow row)")
    ap.add_argument("--out", default=os.path.join(REPO, "reports", "HMATS_PHASE1_SHADOW.md"))
    a = ap.parse_args(argv)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "wue", os.path.join(REPO, "scripts", "weekly_unit_economics.py"))
    wue = importlib.util.module_from_spec(spec); spec.loader.exec_module(wue)
    from analytics.unit_economics.roundtrips import build_round_trips, normalize_fills
    try:
        if a.fills_json:
            with open(a.fills_json, encoding="utf-8") as fh:
                raw = json.load(fh)
            fills = raw.get("fills", raw) if isinstance(raw, dict) else raw
        else:
            fills = wue._pull_fills()
    except Exception as e:  # noqa: silent-swallow — refusal printed, exit 2
        print(f"REFUSING: could not read fills ({type(e).__name__}: {e})"); return 2
    cs = wue._contract_sizes()
    rts, _ = build_round_trips(normalize_fills(fills, cs, wue.PIDS), cs)
    sections = []
    for name, rows in (("[P1-2] realized all-in friction gate", _shadow_rows(a.data_dir)),
                       ("[P1-4] contract-granularity floor", _floor_rows(a.log))):
        start = (dt.datetime.fromisoformat(a.since).timestamp() if a.since
                 else min((r["ts"] for r in rows if "ts" in r), default=time.time()))
        rows = [r for r in rows if r.get("ts", 0) >= start]
        hours = (time.time() - start) / 3600.0
        st = _attribute(rows, rts)
        v, why = _verdict(st, hours)
        counts = {}
        for r in rows:
            counts[r.get("verdict", "?")] = counts.get(r.get("verdict", "?"), 0) + 1
        sections.append(
            f"## {name}\n\n- shadow period: {hours:.1f}h since "
            f"{dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat()}\n"
            f"- verdict counts: {json.dumps(counts)}\n"
            f"- WOULD_SUPPRESS decisions: {st['would_suppress']}; round trips they opened: "
            f"{st['round_trips']}; sum commission ${st['sum_commission_usd']:.2f}; "
            f"sum realized gross PnL ${st['sum_realized_gross_usd']:.2f}\n"
            f"- **verdict: {v}** — {why}\n")
    md = ("# HMATS Phase 1 — shadow report\n\nGenerated "
          f"{dt.datetime.now(dt.timezone.utc).isoformat()}. Switch criterion: suppressed fills' "
          "sum realized gross PnL < sum commission. This report never changes a flag.\n\n"
          + "\n".join(sections))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
