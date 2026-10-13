#!/usr/bin/env python3
"""[P423] Realized P&L per DECIDER — the attribution the sleeve never had.

WHY: `data/trade_attribution.jsonl` stopped on 2026-06-10 (the Kraken path),
and the sleeve's fill ledger (`data/fill_quality.jsonl`, P290) recorded cost
but not CAUSE — no row said which seat (regimebook / whale / skew / etf
de-risk / trend) asked for the position, or why it was closed (signal flip,
risk veto, venue stop, watchdog, FORCE_FLAT). So "does the live decider make
money" was unanswerable from the live book. P423 stamps every sleeve fill
with the driver's order context (`decider`, `src`, `intent_direction`,
`alpha_bps`, `position_before`, `cause`) and this reader turns the ledger
into realized round trips attributed to the ENTRY decider, with the exit
cause tallied separately — the MiFID RTS 6/24 shape (an algo id on every
order) applied to a two-leg book.

METHOD: per asset, walk FILLED rows in time order tracking the signed
position. A leg that INCREASES |position| opens (or adds to) a lot tagged
with that leg's decider; a leg that DECREASES |position| closes contracts
against the open lot at its average entry price and realizes
    (exit_px - entry_px) * sign * contracts * contract_size - fees.
A flip is a close followed by an open. Realized P&L is attributed to the
lot's ENTRY decider; the closing leg's decider/cause is tallied under
"exit cause". An UNRESOLVED leg (no fill price, P290) still moves the
position count but makes every round trip it touches UNPRICED — counted,
never fabricated (P2).

Three tag states are kept apart because they mean different things:
  * `pre-P423 (untagged)`  — the row predates the stamp (no `decider` key)
  * `ledger_error:no_ctx` — the row has the key and it is null: a post-P423
                            fill the driver did not label (a defect to fix)
  * a decider name        — attributed

Stdlib-only; reads the hmats-data volume in-container (P213) or an scp'd
copy via --fill-file. REFUSES (exit 2) on a missing file; a thin ledger is
reported with n, never suppressed into a verdict. n<10 per cell is not a
verdict and is marked so.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# [P423] Local restatement of CoinbaseAdapter._CONTRACT_SIZE_FALLBACK keyed by
# asset (this reader must stay stdlib / importable without the adapter);
# pinned equal to the adapter's table by tests/test_p423_gaps.py (P192).
CONTRACT_SIZE = {"BTC": 0.01, "ETH": 0.1, "SOL": 5.0, "XRP": 500.0,
                 "ADA": 1000.0, "LTC": 5.0, "DOGE": 5000.0, "BNB": 1.0}
MIN_CELL_N = 10
UNTAGGED = "pre-P423 (untagged)"
NO_CTX = "ledger_error:no_ctx"


def _refuse(msg: str) -> None:
    print(f"REFUSING TO REPORT: {msg}", file=sys.stderr)
    sys.exit(2)


def tag_of(row: Dict[str, Any]) -> str:
    if "decider" not in row:
        return UNTAGGED
    d = row.get("decider")
    return str(d) if d else NO_CTX


def cause_of(row: Dict[str, Any]) -> str:
    if "cause" not in row:
        return UNTAGGED
    c = row.get("cause")
    return str(c) if c else NO_CTX


def load_fills(path: Path, window_days: Optional[int]) -> List[Dict[str, Any]]:
    if not path.exists():
        _refuse(f"{path} does not exist here. The sleeve writes it on the LIVE "
                f"data volume — run in-container or pass --fill-file. "
                f"'File missing' is not 'no fills'.")
    cutoff = (datetime.now(timezone.utc).timestamp() - window_days * 86400
              if window_days else None)
    rows = []
    bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:  # noqa: silent-swallow — counted and reported on stderr after the loop (P293 per-record JSONL rule)
            bad += 1
            continue
        if cutoff is not None and float(r.get("ts") or 0) < cutoff:
            continue
        if str(r.get("status") or "") not in ("filled", "unresolved"):
            continue
        rows.append(r)
    rows.sort(key=lambda r: float(r.get("ts") or 0))
    if bad:
        print(f"  note: {bad} unparseable ledger line(s) skipped", file=sys.stderr)
    return rows


def attribute(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure: returns {"round_trips": [...], "unpriced": n, "no_ctx": n,
    "untagged": n, "open_lots": {asset: lot}}."""
    trips: List[Dict[str, Any]] = []
    lots: Dict[str, Dict[str, Any]] = {}   # asset -> open lot
    pos: Dict[str, int] = defaultdict(int)
    unpriced = no_ctx = untagged = drift = 0
    for r in rows:
        a = str(r.get("asset") or "")
        n = int(r.get("contracts") or 0)
        if not a or n <= 0:
            continue
        side = str(r.get("side") or "").upper()
        delta = n if side == "BUY" else -n
        px = r.get("fill_avg_price")
        px = float(px) if px is not None else None
        fee = float(r.get("fees_usd") or 0.0)
        tag = tag_of(r)
        if tag == UNTAGGED:
            untagged += 1
        elif tag == NO_CTX:
            no_ctx += 1
        cur = pos[a]
        # [P423] re-anchor on the venue snapshot the recorder stamped BEFORE
        # the leg: pre-P423 rows have none (maker rows carry INTENDED
        # contracts, unresolved rows may never have filled), so the walked
        # position drifts from the book — count the drift, trust the venue.
        pb = r.get("position_before")
        if pb is not None:
            try:
                pb = int(pb)
                if pb != cur:
                    drift += 1
                    if pb == 0:
                        lots.pop(a, None)
                    cur = pb
            except (TypeError, ValueError):  # noqa: silent-swallow — a malformed snapshot leaves the walked position in place (P223 value coercion)
                pass
        cs = CONTRACT_SIZE.get(a, 1.0)
        remaining = delta
        # 1. closing part (opposite sign to the current position)
        if cur != 0 and (cur > 0) != (remaining > 0):
            close_n = min(abs(remaining), abs(cur))
            lot = lots.get(a)
            sign = 1 if cur > 0 else -1
            if lot is None or lot.get("entry_px") is None or px is None:
                pnl = None
                unpriced += 1
            else:
                pnl = (px - lot["entry_px"]) * sign * close_n * cs
            trips.append({
                "asset": a, "contracts": close_n, "sign": sign,
                "entry_decider": (lot or {}).get("decider", UNTAGGED),
                "entry_ts": (lot or {}).get("ts"), "exit_ts": r.get("ts"),
                "entry_px": (lot or {}).get("entry_px"), "exit_px": px,
                "gross_usd": pnl,
                "fees_usd": (((lot or {}).get("fees_usd", 0.0) * close_n / max(abs(cur), 1))
                             + fee * close_n / n),
                "exit_decider": tag, "exit_cause": cause_of(r),
                "exit_src": r.get("src"),
                "liquidity": r.get("liquidity"), "urgent": bool(r.get("urgent")),
            })
            cur += close_n * (1 if remaining > 0 else -1)
            remaining += close_n * (-1 if remaining > 0 else 1)
            if cur == 0:
                lots.pop(a, None)
            elif lot is not None:
                lot["contracts"] = abs(cur)
                lot["fees_usd"] = lot.get("fees_usd", 0.0) * abs(cur) / (abs(cur) + close_n)
        # 2. opening / adding part
        if remaining != 0:
            lot = lots.get(a)
            if lot is None:
                lots[a] = {"decider": tag, "ts": r.get("ts"), "entry_px": px,
                           "contracts": abs(remaining), "fees_usd": fee * abs(remaining) / n,
                           "src": r.get("src")}
            else:
                # add: average the entry price; an unpriced add poisons the lot
                old_n = lot["contracts"]
                if lot.get("entry_px") is None or px is None:
                    lot["entry_px"] = None
                else:
                    lot["entry_px"] = ((lot["entry_px"] * old_n + px * abs(remaining))
                                       / (old_n + abs(remaining)))
                lot["contracts"] = old_n + abs(remaining)
                lot["fees_usd"] = lot.get("fees_usd", 0.0) + fee * abs(remaining) / n
            cur += remaining
        pos[a] = cur
    return {"round_trips": trips, "unpriced": unpriced, "no_ctx": no_ctx,
            "untagged": untagged, "drift": drift, "open_lots": lots,
            "positions": dict(pos)}


def summarize(res: Dict[str, Any]) -> Dict[str, Any]:
    by: Dict[str, Dict[str, Any]] = {}
    for t in res["round_trips"]:
        key = f"{t['asset']}|{t['entry_decider']}"
        c = by.setdefault(key, {"asset": t["asset"], "entry_decider": t["entry_decider"],
                                "n": 0, "priced": 0, "gross_usd": 0.0, "fees_usd": 0.0,
                                "wins": 0, "contracts": 0, "exit_causes": defaultdict(int)})
        c["n"] += 1
        c["contracts"] += t["contracts"]
        c["fees_usd"] += t["fees_usd"] or 0.0
        c["exit_causes"][t["exit_cause"]] += 1
        if t["gross_usd"] is not None:
            c["priced"] += 1
            c["gross_usd"] += t["gross_usd"]
            if t["gross_usd"] - (t["fees_usd"] or 0.0) > 0:
                c["wins"] += 1
    for c in by.values():
        c["net_usd"] = c["gross_usd"] - c["fees_usd"]
        c["exit_causes"] = dict(c["exit_causes"])
        c["verdict_grade"] = "n<%d: not a verdict" % MIN_CELL_N if c["priced"] < MIN_CELL_N else "ok"
    return by


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fill-file", default="data/fill_quality.jsonl")
    ap.add_argument("--window-days", type=int, default=0,
                    help="0 = the whole ledger (default: round trips span weeks)")
    ap.add_argument("--json", default=None, help="optional report path")
    args = ap.parse_args(argv)
    rows = load_fills(Path(args.fill_file), args.window_days or None)
    if not rows:
        _refuse("no filled/unresolved rows in the window — not 'no P&L'.")
    res = attribute(rows)
    by = summarize(res)
    print(f"Sleeve realized attribution by ENTRY decider — {len(rows)} legs, "
          f"{len(res['round_trips'])} round trips "
          f"({res['unpriced']} unpriced: an unresolved leg touched them)")
    print(f"  tag coverage: untagged(pre-P423)={res['untagged']}  "
          f"ledger_error:no_ctx={res['no_ctx']}  (no_ctx>0 is a DEFECT: a post-P423 "
          f"fill the driver did not label)")
    print(f"\n{'asset':5} {'entry decider':28} {'rt':>3} {'priced':>6} {'gross$':>9} "
          f"{'fees$':>8} {'net$':>9} {'win%':>5}  grade")
    for key in sorted(by):
        c = by[key]
        win = 100.0 * c["wins"] / c["priced"] if c["priced"] else float("nan")
        print(f"{c['asset']:5} {c['entry_decider'][:28]:28} {c['n']:3d} {c['priced']:6d} "
              f"{c['gross_usd']:+9.2f} {c['fees_usd']:8.2f} {c['net_usd']:+9.2f} "
              f"{win:5.0f}  {c['verdict_grade']}")
        print(f"      exit causes: {c['exit_causes']}")
    if res["open_lots"]:
        print("\nopen lots (not yet realized): " + ", ".join(
            f"{a}={l['contracts']}ct by {l['decider']}" for a, l in res["open_lots"].items()))
    print("\nRead honestly: attribution is by the ENTRY decider; the exit cause "
          "says who/what closed it. Pre-P423 rows carry no tag by construction — "
          "this table becomes evidence as post-P423 fills accrue (P230).")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(),
             "legs": len(rows), "cells": by, "unpriced": res["unpriced"],
             "no_ctx": res["no_ctx"], "untagged": res["untagged"],
             "open_lots": res["open_lots"]}, indent=1, default=str),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
