"""[P1-6] Weekly unit-economics KPI report from the venue's own ledger.

    docker exec hmats-engine python -X utf8 scripts/weekly_unit_economics.py
    python -X utf8 scripts/weekly_unit_economics.py --fills-json fills.json   # offline

Per asset and per asset x regime: alpha_per_rt_bps, all_in_cost_per_rt_bps,
net_per_rt_bps, fills_per_rt, median_hold_bars, fee_share_of_alpha, n_rt.
Round trips come from analytics/unit_economics/roundtrips.py — the Phase 0
algorithm, one implementation. Live mode makes READ-ONLY calls only
(get_fills, get_product). Writes data/evidence_reports/unit_economics_<date>.md.
Exit 0 written / 2 refusal (no fills, no credentials, unreadable input).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PIDS = {"BIP-20DEC30-CDE": "BTC", "ETP-20DEC30-CDE": "ETH", "SLP-20DEC30-CDE": "SOL",
        "XPP-20DEC30-CDE": "XRP", "BNB-20DEC30-CDE": "BNB"}


def _contract_sizes():
    from exchange.coinbase_adapter import CoinbaseAdapter
    return {a: float(CoinbaseAdapter._CONTRACT_SIZE_FALLBACK[p])
            for p, a in PIDS.items() if p in CoinbaseAdapter._CONTRACT_SIZE_FALLBACK}


def _pull_fills():
    from coinbase.rest import RESTClient
    from exchange.coinbase_adapter import _plain
    kf = os.environ.get("COINBASE_KEY_FILE") or os.path.join(REPO, "data", ".coinbase_key.json")
    c = RESTClient(key_file=kf)
    fills, cursor = [], None
    for _ in range(60):
        kw = {"limit": 250}
        if cursor:
            kw["cursor"] = cursor
        r = _plain(c.get_fills(**kw))
        fills.extend(r.get("fills", []) or [])
        cursor = r.get("cursor") or None
        if not cursor:
            break
    return fills


def _decision_mids(data_dir):
    out = {}
    try:
        with open(os.path.join(data_dir, "fill_quality.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:  # noqa: silent-swallow — torn line skipped
                    continue
                if r.get("order_id") and r.get("decision_mid"):
                    out.setdefault(r["order_id"], float(r["decision_mid"]))
    except FileNotFoundError:  # noqa: silent-swallow — no ledger = spread UNAVAILABLE, said in the report
        pass
    return out


def _funding(data_dir):
    out = {}
    try:
        with open(os.path.join(data_dir, "cde_funding_history.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    out.setdefault(r["asset"], []).append((float(r["ts"]), float(r["funding_8h"]) / 8.0))
                except (ValueError, KeyError, TypeError):  # noqa: silent-swallow — malformed row skipped
                    continue
    except FileNotFoundError:  # noqa: silent-swallow — no history = funding UNAVAILABLE
        pass
    return out


def _regimes(data_dir):
    out = {}
    pat = re.compile(r"'regime': '([A-Z_0-9]+)'")
    for f in glob.glob(os.path.join(data_dir, "diagnostics", "diag_*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
            m = pat.search(str((d.get("components", {}).get("regime_hysteresis") or {}).get("output", "")))
            if m and d.get("tick_time"):
                t = dt.datetime.fromisoformat(d["tick_time"].replace("Z", "+00:00")).timestamp()
                out.setdefault(d["asset"], []).append((t, m.group(1)))
        except Exception:  # noqa: silent-swallow — an unreadable diag file is one missing regime label
            continue
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fills-json", default=None)
    ap.add_argument("--data-dir", default=os.environ.get("HMATS_DATA_DIR", os.path.join(REPO, "data")))
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from analytics.unit_economics.roundtrips import (build_round_trips, normalize_fills,
                                                     render_markdown)
    try:
        if a.fills_json:
            with open(a.fills_json, encoding="utf-8") as fh:
                raw = json.load(fh)
            fills = raw.get("fills", raw) if isinstance(raw, dict) else raw
        else:
            fills = _pull_fills()
    except Exception as e:  # noqa: silent-swallow — refusal is printed and exits 2
        print(f"REFUSING: could not read fills ({type(e).__name__}: {e})")
        return 2
    if not fills:
        print("REFUSING: zero fills returned — 'no data' is not a KPI")
        return 2
    cs = _contract_sizes()
    F = normalize_fills(fills, cs, PIDS, _decision_mids(a.data_dir))
    rts, open_pos = build_round_trips(F, cs, _funding(a.data_dir), _regimes(a.data_dir))
    since = time.time() - a.days * 86400
    md = render_markdown(rts, open_pos, since,
                         f"HMATS unit economics — {dt.date.today().isoformat()}")
    out = a.out or os.path.join(a.data_dir, "evidence_reports",
                                f"unit_economics_{dt.date.today().strftime('%Y%m%d')}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
