#!/usr/bin/env python3
"""[P271] september_check — the whole September evidence read as ONE command.

WHY THIS EXISTS
---------------
The system's profitability question resolves in early September through
several independent instruments (the P237 tripwire, six P166 forward-ledger
candidates). Each was operator-remembered, multi-step, and split across the
server and this machine — and a decision procedure that depends on someone
re-running tools by hand quietly becomes "never" (the P230/P235 lesson).
This script makes the weekly trajectory read and the September verdicts one
executable step. It CHANGES NOTHING — it reads, scores, and reports; every
action it recommends is a documented human step
(docs/SEPTEMBER_DECISION_TREE.md).

WHAT IT DOES (operator machine, needs ssh hmats + the training venv)
--------------------------------------------------------------------
 1. pulls the shadow ledgers + evidence reports from the server volume
 2. refreshes the local 4H OHLCV series (monthly archives + daily extension)
 3. builds Kraken-sourced OHLCV parquets for the P271 breadth assets — the
    scorer otherwise reports `ohlcv_missing` for them, which reads exactly
    like "no signal" (the P199/P264 trap: a ledger is only working when the
    tool that judges it can read today's rows end-to-end)
 4. runs compute_shadow_ic over ALL candidate prefixes
 5. runs the P237 tripwire checker on the pulled reports
 6. prints a per-candidate countdown to its 30d P166 read date

Run weekly (and on the decision dates):
    python -X utf8 scripts/september_check.py
    python -X utf8 scripts/september_check.py --window-days 30   # the exam

Exit codes (P287): 3 = the P237 tripwire FIRED (grep-able, outranks scorer
failures); 2 = refusal (no fresh pull / scorer refusal); 1 = scorer failure;
0 = ok. A tripwire REFUSAL is surfaced in the RESULT line, not the rc.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

LEDGER_DIR = REPO / "data" / "strategy_shadow_pulled"
REPORTS_DIR = REPO / "data" / "evidence_reports_pulled"
OHLCV_DIR = REPO / "training" / "training_data" / "drl_training"

# candidate -> (ledger prefix, first-live date, what a PASS flips)
# [P287] This roster must cover EVERY accruing candidate the scorer reads —
# the pre-P287 list omitted the five P277 families and mlpshadow, i.e. the
# one command that prints "days left" was silent on six exam dates (the
# exact operator-must-remember gap this script exists to close, P230/P271).
CANDIDATES = {
    "regimebook":     ("regimebook",     "2026-08-10", "regimebook_mode enforce (P256 seat)"),
    "regimebook_adj": ("regimebook",     "2026-08-10", "ADJ params into the seat"),
    "derivflow":      ("derivflow",      "2026-08-08", "liq-basis candidate -> own P-entry"),
    "ma_filter":      ("ma_filter",      "2026-08-08", "coinbase_ma_filter_enforce true"),
    "volskip":        ("regimebook",     "2026-08-16", "volskip into the ETH seat path"),
    "etfflow":        ("etfflow",        "2026-08-16", "ETF-flow candidate -> own P-entry"),
    "breadth books":  ("regimebook",     "2026-08-16", "widening decision (routing + books)"),
    # [P277] enhancement families (all first-live at the P277 deploy)
    "stablecoinflow": ("stablecoinflow", "2026-08-16", "P277 candidate -> own P-entry"),
    "oidiv_confirm":  ("oidiv",          "2026-08-16", "P277 candidate -> own P-entry"),
    "oidiv_fade":     ("oidiv",          "2026-08-16", "P277 candidate -> own P-entry"),
    "calbasis":       ("calbasis",       "2026-08-16", "P277 candidate -> own P-entry"),
    "xsmom":          ("xsmom",          "2026-08-16", "P277 candidate -> own P-entry"),
    "eventfilter":    ("eventfilter",    "2026-08-16", "enforcement -> own P-entry (P277)"),
    # [P284/P285c] the trained-model Rung-3 shadow; seat WITHDRAWN (P285c),
    # ledger accrues as free evidence toward the ~09-15 read
    "mlpshadow":      ("mlpshadow",      "2026-08-16", "P285c-withdrawn; read is informational"),
    # [P289] P288 trend-rule challengers — PARTIAL-certified (transfer-
    # validated on virgin era + 5 never-fitted assets; BTC-2018 crash-dodge
    # miss recorded). First-live at the next deploy after 2026-08-16.
    "donchian":       ("donchian",       "2026-08-17", "challenger seat/widening -> own P-entry (P288 PARTIAL)"),
    "emaens":         ("emaens",         "2026-08-17", "challenger seat/widening -> own P-entry (P288 PARTIAL)"),
}

KRAKEN_PAIRS_BREADTH = {"XRP": "XRPUSD", "ADA": "ADAUSD", "LTC": "LTCUSD",
                        "DOGE": "XDGUSD", "BNB": "BNBUSD"}


def _run(cmd: list, **kw) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, **kw).returncode


def pull_from_server() -> bool:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    # tar-pipe: scp brace-globbing over ssh is shell-dependent (learned the
    # hard way in P270's evidence pull)
    for remote, local in (
            ("/var/lib/docker/volumes/hmats-data/_data/strategy_shadow",
             LEDGER_DIR),
            ("/var/lib/docker/volumes/hmats-data/_data/evidence_reports",
             REPORTS_DIR)):
        rc = subprocess.run(
            f'ssh hmats "cd {remote} && tar cf - ." | tar xf - -C "{local}"',
            shell=True).returncode
        if rc != 0:
            print(f"  !! pull failed for {remote} (rc={rc})")
            ok = False
    return ok


def refresh_prices() -> int:
    return _run([sys.executable, "-X", "utf8",
                 str(REPO / "training" / "scripts" / "refresh_ohlcv_4h.py")])


def build_breadth_ohlcv() -> int:
    """Kraken public OHLC -> {ASSET}_4H_ohlcv.parquet for the breadth
    assets. ~720 bars (~120d) — enough for every 30d window this script
    scores. Merges with any existing file so history GROWS past Kraken's
    own window (the P266 CoinGlass lesson: a bounded-depth API feeding an
    overwriting fetcher caps history forever)."""
    n_failed = 0
    try:
        import pandas as pd
    except ImportError:
        print("  !! pandas missing — breadth OHLCV skipped")
        return len(KRAKEN_PAIRS_BREADTH)
    for asset, pair in KRAKEN_PAIRS_BREADTH.items():
        try:
            url = (f"https://api.kraken.com/0/public/OHLC?pair={pair}"
                   f"&interval=240")
            with urllib.request.urlopen(url, timeout=20) as r:
                d = json.loads(r.read().decode())
            if d.get("error"):
                raise RuntimeError(str(d["error"])[:80])
            res = d.get("result", {})
            key = next((k for k in res if k != "last"), None)
            rows = res.get(key, [])
            if len(rows) < 50:
                raise RuntimeError(f"only {len(rows)} rows")
            # drop the LAST row: Kraken returns the in-progress candle
            # (P253c class — scoring against a partial bar skews forward
            # returns on exactly the newest records)
            rows = rows[:-1]
            df = pd.DataFrame(
                [{"timestamp": pd.Timestamp(int(r[0]), unit="s", tz="UTC"),
                  "open": float(r[1]), "high": float(r[2]),
                  "low": float(r[3]), "close": float(r[4]),
                  "volume": float(r[6])} for r in rows])
            out = OHLCV_DIR / f"{asset}_4H_ohlcv.parquet"
            if out.exists():
                old = pd.read_parquet(out)
                df = (pd.concat([old, df])
                      .drop_duplicates(subset="timestamp", keep="last")
                      .sort_values("timestamp").reset_index(drop=True))
            OHLCV_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
            print(f"  breadth OHLCV {asset}: {len(df)} bars -> {out.name}")
        except Exception as e:
            n_failed += 1
            print(f"  !! breadth OHLCV {asset} FAILED: "
                  f"{type(e).__name__}: {e}")
    return n_failed


def run_scorer(window_days: int) -> int:
    """[P332] Score BOTH ways: per-asset for diagnosis, POOLED for the verdict.

    Until now this ran the per-asset read only, and P293g measured that such a
    read CANNOT certify at the 16h horizon inside a 30-day window: it needs
    IC >= 0.302 where the economic bar is ~0.13. So every 16h September verdict
    would have been computed on a clock that cannot fire — the P174 shape, on
    the reads the whole roster is waiting for. P299 built --pool-assets for
    exactly this and nothing wired it in (P170: a mechanism nothing calls).

    Measured on the pulled ledgers the day this landed, pooling roughly triples
    n: ma_filtered 116 -> 348, regimebook 101 -> 573, regimebook_adj -> 300.

    BOTH are printed rather than swapping one for the other, because they
    answer different questions and only one of them can be a verdict:
      * POOLED governs promote/kill for a POOLABLE family (declared same-rule,
        standardized per asset before pooling — P299).
      * PER-ASSET is diagnosis: it shows WHERE a family works, which is what
        P307c needed to explain two labs that appeared to disagree.
    """
    rc_asset = _run([sys.executable, "-X", "utf8", "-m",
                     "analytics.shadow_ic.compute_shadow_ic",
                     "--ledger-dir", str(LEDGER_DIR),
                     "--window-days", str(window_days)])
    print("")
    print("=" * 74)
    print("  POOLED READ — this is the one the P166 gate can fire on for a")
    print("  POOLABLE family; the per-asset table above is diagnosis (P299).")
    print("=" * 74)
    rc_pooled = _run([sys.executable, "-X", "utf8", "-m",
                      "analytics.shadow_ic.compute_shadow_ic",
                      "--ledger-dir", str(LEDGER_DIR),
                      "--window-days", str(window_days),
                      "--pool-assets"])
    # A refusal in EITHER read is a refusal: a missing pooled read must not be
    # invisible behind a healthy per-asset one (P199 — "no data" is not "no
    # signal", one level up).
    return rc_asset or rc_pooled


def run_tripwire() -> int:
    return _run([sys.executable, "-X", "utf8",
                 str(REPO / "analytics" / "calibration" / "tripwire_check.py"),
                 "--reports-dir", str(REPORTS_DIR)])


def countdown() -> None:
    today = datetime.now(timezone.utc).date()
    print("\n" + "=" * 74)
    print(f"  SEPTEMBER COUNTDOWN (today {today})   "
          f"decision tree: docs/SEPTEMBER_DECISION_TREE.md")
    print("=" * 74)
    print(f"  {'candidate':<16} {'ledger since':<13} {'30d read date':<14} "
          f"{'days left':<10} on PASS")
    for name, (prefix, since, action) in CANDIDATES.items():
        d0 = datetime.strptime(since, "%Y-%m-%d").date()
        read = d0.toordinal() + 30
        left = read - today.toordinal()
        read_s = datetime.fromordinal(read).date().isoformat()
        print(f"  {name:<16} {since:<13} {read_s:<14} "
              f"{max(0, left):<10} {action}")
    print(f"  {'trend tripwire':<16} {'weekly crons':<13} {'2026-09-01':<14} "
          f"{max(0, datetime(2026, 9, 1).date().toordinal() - today.toordinal()):<10} "
          f"per-asset trend_assets removal (P237)")


def _final_rc(scorer_rc: int, tripwire_rc: int) -> int:
    """[P287] Exit-code contract, pure so it is testable:
      3 = the P237 tripwire FIRED (tripwire_check's own grep-able contract,
          previously discarded — a FIRED tripwire survived only as
          scrollback). FIRED outranks a scorer failure: the deactivation
          decision is the more urgent signal.
      else the scorer's rc (2 = its refusal, 1 = failure, 0 = ok).
    A tripwire REFUSAL (rc 2, e.g. no reports pulled) does NOT fail the
    run — it is surfaced in the RESULT line instead, because the scorer
    half of the run is still valid evidence."""
    if tripwire_rc == 3:
        return 3
    return scorer_rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-days", type=int, default=10,
                    help="scoring window (10 = weekly trajectory read; "
                         "30 = the P166 exam)")
    ap.add_argument("--no-pull", action="store_true",
                    help="score already-pulled ledgers")
    args = ap.parse_args()
    if not args.no_pull:
        print("[1/5] pulling ledgers + reports from the server...")
        if not pull_from_server():
            print("REFUSING to score without a fresh pull — stale local "
                  "ledgers score yesterday's system (P255: local data/ is "
                  "never live evidence). Use --no-pull to override.")
            return 2
    print("[2/5] refreshing home-asset OHLCV...")
    prices_rc = refresh_prices()
    print("[3/5] building breadth-asset OHLCV from Kraken...")
    breadth_failed = build_breadth_ohlcv()
    print(f"[4/5] scoring all candidates (window={args.window_days}d)...")
    rc = run_scorer(args.window_days)
    print("[5/5] tripwire status...")
    tw_rc = run_tripwire()
    countdown()
    # [P287] one unmissable RESULT line — step failures used to be `!!`
    # noise mid-scroll above a normal-looking verdict table, and the
    # tripwire's exit code was dropped entirely.
    tw_txt = {0: "not fired", 2: "REFUSED (reports unreadable/missing)",
              3: "*** FIRED ***"}.get(tw_rc, f"rc={tw_rc}")
    print(f"\nRESULT: scorer rc={rc} | tripwire {tw_txt} | "
          f"price refresh {'OK' if prices_rc == 0 else 'FAILED'} | "
          f"breadth OHLCV failures {breadth_failed}"
          + ("  <- some price series are stale; forward joins shrink "
             "accordingly" if (prices_rc != 0 or breadth_failed) else ""))
    return _final_rc(rc, tw_rc)


if __name__ == "__main__":
    sys.exit(main())
