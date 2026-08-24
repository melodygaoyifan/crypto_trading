"""[P395] Forward-accumulation of the FREE new-data direction signals so they
become out-of-sample-probeable in months — the P389c pattern applied to options
(Deribit put/call) and on-chain (Blockchair), under the operator's fixed-fund /
Coinbase+Kraken constraint.

WHY THIS AND NOT A BACKTEST NOW: the two free new-data leads the research graded
worth pursuing (P388) — options positioning (Deribit put/call OI + volume, the
direction signal, BTC/ETH) and on-chain settlement (Blockchair volume + largest
transfer, BTC/ETH) — are LIVE SNAPSHOTS with NO free history. So they cannot be
screened today; the only honest acquisition path without paying for history
(Tardis/Amberdata options-chain; Glassnode/CryptoQuant on-chain) is to bank them
forward, exactly like the P389c positioning cron. In ~6-12 months there is enough
to run the P386 hold-aware Rung-0 probe on real OOS data.

HONEST CEILING (stated so nobody over-invests): under the fixed ~$11k / flat-CDE-
fee constraint the required IC to clear cost stays ~0.07-0.11 (P385/P392). A new
signal must roughly DOUBLE the current 0.04 AND clear that floor. Positioning
(OI/LSR) already FAILED that OOS (P391); DVOL already screened dead. Put/call and
on-chain are the remaining free untested slices — a low-cost long shot, not a
likely win. This script banks the data; the probe (months out) decides.

OUTPUT: training/training_data/signal_snapshots/newdata_snapshots.jsonl — one row
per UTC day, merge/dedup by date (P266: re-runs GROW the archive, never lose a
row). gitignored (training_data/), operator-local + server-cron; scp to merge.

CADENCE: daily (these are 24h-rolling snapshots; a daily row captures each day's
value). Add to the P389c server cron. Never raises; a failed source leaves its
fields absent (P2 — absence is not a fabricated zero), the other source still banks.
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get(
    "HMATS_SNAPSHOT_DIR",
    str(REPO / "training" / "training_data" / "signal_snapshots")))
OUT = OUT_DIR / "newdata_snapshots.jsonl"


def _fetch_deribit() -> dict:
    """Options positioning: put/call (OI + volume) + DVOL, BTC/ETH. Free, no SOL."""
    try:
        sys.path.insert(0, str(REPO))
        from data_mgmt.feeds.deribit_feed import get_deribit_feed

        async def _run():
            return await get_deribit_feed().fetch()

        snap = asyncio.run(_run())
        out = {}
        for cur, m in (snap.metrics or {}).items():
            if not m.is_usable():
                continue
            out[cur] = {
                "put_call_ratio_oi": m.put_call_ratio_oi,
                "put_call_ratio_volume": m.put_call_ratio_volume,
                "total_oi_calls": m.total_oi_calls,
                "total_oi_puts": m.total_oi_puts,
                "mean_mark_iv": m.mean_mark_iv,
                "dvol": m.dvol,
                "underlying_price": m.underlying_price,
                "instrument_count": m.instrument_count,
            }
        return out
    except Exception as e:  # a feed failure must never lose the on-chain row
        print(f"  [deribit] fetch failed: {e}")
        return {}


def _fetch_onchain() -> dict:
    """On-chain settlement: 24h volume USD + largest transfer, BTC/ETH. Free."""
    try:
        sys.path.insert(0, str(REPO))
        from data_mgmt.feeds.blockchair_onchain import get_blockchair_onchain_feed

        d = get_blockchair_onchain_feed().fetch() or {}
        out = {}
        for cur, v in d.items():
            vol = getattr(v, "onchain_volume_24h_usd", 0.0)
            if not vol:
                continue
            out[cur] = {
                "onchain_volume_24h_usd": vol,
                "transaction_count": getattr(v, "transaction_count", 0),
                "largest_transaction_usd": getattr(v, "largest_transaction_usd", 0.0),
            }
        return out
    except Exception as e:
        print(f"  [onchain] fetch failed: {e}")
        return {}


def _load_existing() -> dict:
    """date(str) -> row, so a re-run overwrites the same UTC day (idempotent)."""
    rows = {}
    if OUT.exists():
        for ln in OUT.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
                if r.get("date"):
                    rows[r["date"]] = r
            except Exception:
                continue
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # timestamp passed in (Date.now-free convention not required here — cron job)
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y-%m-%d")

    deribit = _fetch_deribit()
    onchain = _fetch_onchain()
    if not deribit and not onchain:
        print("BOTH sources failed — nothing banked (archive untouched)")
        return 1

    rows = _load_existing()
    rows[day] = {
        "date": day,
        "fetched_at": now.isoformat(),
        "options": deribit,   # {BTC:{pcr_oi,...}, ETH:{...}}  (absent => not fetched, P2)
        "onchain": onchain,   # {BTC:{onchain_volume_24h_usd,...}, ETH:{...}}
    }
    tmp = OUT.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for d in sorted(rows):
            f.write(json.dumps(rows[d]) + "\n")
    os.replace(tmp, OUT)

    print(f"BANKED {day}: options={sorted(deribit)} onchain={sorted(onchain)} "
          f"-> {OUT.name} ({len(rows)} days total, "
          f"{sorted(rows)[0]} -> {sorted(rows)[-1]})")
    if deribit:
        for cur, m in deribit.items():
            print(f"  {cur} pcr_oi={m['put_call_ratio_oi']:.3f} "
                  f"pcr_vol={m['put_call_ratio_volume']:.3f} dvol={m['dvol']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
