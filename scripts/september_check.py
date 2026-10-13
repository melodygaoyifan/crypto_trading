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
 3. builds Kraken-sourced OHLCV EXTENSION parquets ({ASSET}_4H_ohlcv_kraken)
    for the P271 breadth assets — the scorer otherwise reports
    `ohlcv_missing` for them, which reads exactly like "no signal" (the
    P199/P264 trap: a ledger is only working when the tool that judges it can
    read today's rows end-to-end). [P420] It never writes the PRIMARY
    {ASSET}_4H_ohlcv.parquet — that is the seat-alpha calibration input.
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
# [P421e] The ledger of reads TAKEN (committed). The countdown had no notion
# of a read having happened: after the 19 September reads were executed on
# 2026-10-12 it kept printing "19 READ(S) ARE DUE" every Monday forever — an
# alarm that fires on the healthy state is wallpaper (P202), and the one read
# that genuinely needs a human would be indistinguishable from the 18 that do
# not. A read is recorded here ONLY by run_scorer when its window covered the
# candidate's full accrual (the P421 trap: the 30d-from-today default read
# n=0..noise while the full-span read was decisive).
READS_TAKEN_PATH = REPO / "analytics" / "shadow_ic" / "reads_taken.json"
SCORER_REPORTS_DIR = REPO / "analytics" / "shadow_ic" / "reports"
SEAT_REPORT_PATH = SCORER_REPORTS_DIR / "seat_read_regimebook_latest.json"

# [P420] TWO-FILE CONVENTION for the breadth price series, pinned by test:
#   {ASSET}_4H_ohlcv.parquet         PRIMARY — Binance-sourced, built ONLY by
#                                     training/scripts/refresh_ohlcv_4h.py from
#                                     raw/{ASSET}_60m.parquet. It is the INPUT of
#                                     training/seat_alpha_calibration.py, i.e. the
#                                     producer behind the LIVE XRP/BNB gate
#                                     constants (core.seat_alpha).
#   {ASSET}_4H_ohlcv_kraken.parquet  EXTENSION — the ~120d Kraken window this
#                                     script fetches; the scorer reads it as a
#                                     UNION with the primary (primary wins).
# Before P420 this script MERGED its Kraken rows INTO the primary (keep="last"),
# so ~560 Binance bars per asset were overwritten with Kraken prints and the
# seat-alpha producer drifted from its own shipped table (BNB validation
# +36.6 -> +51.1, XRP -21.9 -> -14.8) the first time it ran after the P412c
# commit. A reader of live evidence must NEVER write the calibration input.
PRIMARY_OHLCV_SUFFIX = "_4H_ohlcv.parquet"
KRAKEN_OHLCV_SUFFIX = "_4H_ohlcv_kraken.parquet"


def kraken_ohlcv_path(asset: str) -> Path:
    """The EXTENSION file this script owns — never the primary (P420)."""
    return OHLCV_DIR / f"{asset}{KRAKEN_OHLCV_SUFFIX}"

# candidate -> (ledger prefix, first-live date, what a PASS flips)
# [P287] This roster must cover EVERY accruing candidate the scorer reads —
# the pre-P287 list omitted the five P277 families and mlpshadow, i.e. the
# one command that prints "days left" was silent on six exam dates (the
# exact operator-must-remember gap this script exists to close, P230/P271).
CANDIDATES = {
    "regimebook":     ("regimebook",     "2026-08-10", "regimebook_mode enforce (P256 seat)"),
    "regimebook_adj": ("regimebook",     "2026-08-10", "ADJ params into the seat"),
    "derivflow":      ("derivflow",      "2026-08-08", "liq-basis candidate -> own P-entry"),
    # [P361] ma_filter's action is NO LONGER a plain arm-on-PASS: P324 measured
    # it NOT EARNED at the pre-committed bar, P337 found its disagreements
    # marked entries that did BETTER against the decider it actually filters,
    # P348 showed no obtainable sample makes it significant, and P356 DISARMED
    # it by operator instruction. The read still happens (evidence is free),
    # but a PASS now has to argue with four entries.
    "ma_filter":      ("ma_filter",      "2026-08-08",
                       "re-arm ONLY vs P324/P337/P348 (disarmed P356)"),
    # [P361] whale_filter was accruing since 08-18 with NO read date at all —
    # the roster's own comment demands every scorer-read candidate, and this
    # one was missing, so its ledger would have grown unexamined (P199/P230,
    # the gap this script exists to close).
    "whale_filter":   ("whale_filter",   "2026-08-18",
                       "re-arm ONLY vs P324/P348 (disarmed P356)"),
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
    # [P420] three streams were accruing with NO read date (the P361 gap,
    # again): the LIVE skew decider's own A/B (P407j), the WS2 conviction
    # shadow (its reader is scripts/conviction_sizing_review.py, PnL-based —
    # the shadow-IC scorer does not judge a sizing overlay), and the breadth
    # regime books, which since this fix carry strategy "regimebook_breadth"
    # (fork-4 contract) and read the scorer's kraken-extension series.
    "skewetf":        ("skewetf",        "2026-08-25",
                       "agree-gate vs skew-override on BTC/ETH (P407i/j) -> own P-entry"),
    "convsize":       ("convsize",       "2026-08-27",
                       "re-arm fusion_conviction_to_sleeve ONLY vs P417 (reader: conviction_sizing_review.py)"),
    # ledger-since = the deploy date of P420 (the rows are written under the
    # new strategy name from that deploy on); set it when the deploy lands.
    "regimebook_breadth": ("regimebook",  "TBD-at-deploy",
                       "breadth book verdict per asset; XRP/BNB already routed, ADA/LTC/DOGE stay out (P412c)"),
}

# [P421e] Which scorer `strategy` names carry each candidate's verdict. A
# ledger PREFIX is not a strategy name: "regimebook" prefixes five families
# (regimebook, _adj, _volskip, _fgshort, _breadth) and "derivflow" holds two
# strategies, so a prefix match would lump a candidate's verdict with its
# neighbours'. An empty tuple = the scorer does not judge this candidate (its
# reader is named in the CANDIDATES action text).
VERDICT_STRATEGIES = {
    "regimebook":     ("regimebook",),
    "regimebook_adj": ("regimebook_adj",),
    "derivflow":      ("liquidation_squeeze", "liquidation_exhaustion"),
    "ma_filter":      ("ma_filtered",),
    "whale_filter":   ("whale_filtered",),
    "volskip":        ("regimebook_volskip",),
    "etfflow":        ("etfflow",),
    "breadth books":  ("regimebook_breadth",),
    "stablecoinflow": ("stablecoinflow",),
    "oidiv_confirm":  ("oidiv_confirm",),
    "oidiv_fade":     ("oidiv_fade",),
    "calbasis":       ("calbasis",),
    "xsmom":          ("xsmom",),
    "eventfilter":    ("eventfilter",),
    "mlpshadow":      ("mlpshadow",),
    "donchian":       ("donchian",),
    "emaens":         ("emaens",),
    "skewetf":        ("skewetf_skew", "skewetf_etf", "skewetf_agree"),
    "convsize":       (),
    "regimebook_breadth": ("regimebook_breadth",),
}

# [P421e] Candidates with a PRE-COMMITTED next read beyond the P166 clock.
# Stamped onto the ledger entry so the countdown shows the date and the
# instrument that owns it, instead of a human remembering (P333).
NEXT_READS = {
    "calbasis": "2026-11-10 (scripts/calbasis_check.py, P421 criterion)",
}


def load_reads_taken(path: Path = None) -> dict:
    """[P421e] {candidate: {read_on, window_days, verdicts, reports, ...}}.
    An unreadable ledger reads as EMPTY (every overdue read shouts) — the
    fail direction a missing alarm must have (P2)."""
    path = Path(path) if path else READS_TAKEN_PATH
    try:
        d = json.loads(path.read_text(encoding="utf-8-sig"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _verdict_summary(report: dict, strategies: tuple) -> dict:
    """Per strategy: the POOLED record's verdict when one exists, else the
    per-asset verdicts. Reads the scorer's own fields; computes nothing."""
    out = {}
    recs = (report or {}).get("per_strategy") or []
    for st in strategies:
        rows = [r for r in recs if r.get("strategy") == st]
        pooled = [r for r in rows if r.get("asset") == "POOLED"]
        if pooled:
            r = pooled[0]
            ic = r.get("ic_per_horizon") or {}
            out[st] = {"scope": "POOLED", "verdict": r.get("verdict"),
                       "ic": {h: round(float(v), 4) for h, v in ic.items()},
                       "n": r.get("n_per_horizon")}
        elif rows:
            out[st] = {"scope": "per-asset",
                       "verdict": {r.get("asset"): r.get("verdict") for r in rows}}
        else:
            out[st] = {"scope": "absent", "verdict": None}
    return out


def record_reads_taken(window_days: int, per_asset_report: Path,
                       pooled_report: Path, today=None,
                       path: Path = None) -> tuple:
    """[P421e] Record every candidate whose FULL accrual the window covered.

    Returns (recorded, skipped) candidate-name lists. A candidate whose ledger
    started more than `window_days` ago is SKIPPED, never recorded: a short
    window is the exact mistake P421 made first (30d default, n=0..noise) and
    recording it as "read" would silence the alarm on an exam that was not
    sat. An unstarted clock (TBD-since) is skipped too.
    """
    path = Path(path) if path else READS_TAKEN_PATH
    today = today or datetime.now(timezone.utc).date()
    try:
        per_asset = json.loads(Path(per_asset_report).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        per_asset = {}
    try:
        pooled = json.loads(Path(pooled_report).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        pooled = {}
    ledger = load_reads_taken(path)
    recorded, skipped = [], []
    for name, (prefix, since, _action) in CANDIDATES.items():
        try:
            d0 = datetime.strptime(since, "%Y-%m-%d").date()
        except ValueError:
            skipped.append(name)
            continue
        span = (today - d0).days
        if window_days < span:
            skipped.append(name)
            continue
        strategies = VERDICT_STRATEGIES.get(name, (prefix,))
        summary = _verdict_summary(pooled, strategies) if strategies else {}
        # a strategy with no POOLED row falls back to the per-asset report
        for st, v in list(summary.items()):
            if v.get("scope") == "absent":
                summary[st] = _verdict_summary(per_asset, (st,))[st]
        entry = {"read_on": today.isoformat(), "window_days": int(window_days),
                 "ledger_since": since, "span_days": span,
                 "verdicts": summary,
                 "reports": {"per_asset": str(per_asset_report),
                             "pooled": str(pooled_report)}}
        if not strategies:
            entry["note"] = ("not judged by the shadow-IC scorer; "
                             "see the action column for its reader")
        nxt = NEXT_READS.get(name) or (ledger.get(name) or {}).get("next_read")
        if nxt:
            entry["next_read"] = nxt
        ledger[name] = entry
        recorded.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    return recorded, skipped


def _verdict_line(entry: dict) -> str:
    vs = entry.get("verdicts") or {}
    if not vs:
        return entry.get("note") or "read taken"
    parts = []
    for st, v in vs.items():
        verdict = v.get("verdict")
        if isinstance(verdict, dict):
            verdict = "/".join(f"{a}:{x}" for a, x in verdict.items())
        ic = v.get("ic") or {}
        ic_s = ("" if not ic else
                " ic " + "/".join(f"{ic[h]:+.2f}" for h in sorted(ic, key=int)))
        parts.append(f"{st} {verdict}{ic_s}")
    s = "; ".join(parts)
    nxt = entry.get("next_read")
    return s + (f"  | next read {nxt}" if nxt else "")

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


def _naive_utc(series):
    """Coerce a timestamp series to tz-NAIVE UTC — the canonical convention the
    home OHLCV builder (refresh_ohlcv_4h) and the scorer both use. Kraken rows
    were built tz-AWARE while the merged-in history is naive, so concat/sort
    raised 'Cannot compare tz-naive and tz-aware timestamps'. Naive-in stays
    naive (assumed UTC); aware-in is converted to UTC then made naive, so both
    sides of a merge are always the same kind."""
    import pandas as pd
    s = pd.to_datetime(series, utc=True)   # naive->assume-UTC, aware->to-UTC
    return s.dt.tz_localize(None)          # drop tz -> naive UTC


def build_breadth_ohlcv() -> int:
    """Kraken public OHLC -> {ASSET}_4H_ohlcv_kraken.parquet for the breadth
    assets. ~720 bars (~120d) — enough for every 30d window this script
    scores. Merges with any existing KRAKEN file so history GROWS past
    Kraken's own window (the P266 CoinGlass lesson: a bounded-depth API
    feeding an overwriting fetcher caps history forever).

    [P420] Writes the EXTENSION file only. The PRIMARY {ASSET}_4H_ohlcv.parquet
    is the seat-alpha calibration input (core.seat_alpha via
    training/seat_alpha_calibration.py --verify) and is built exclusively by
    refresh_ohlcv_4h.py from the Binance raw archive; merging Kraken prints
    into it is how the live XRP/BNB gate constants drifted from their own
    producer. The scorer unions the two (primary wins)."""
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
                [{"timestamp": pd.Timestamp(int(r[0]), unit="s"),   # naive UTC (P414b)
                  "open": float(r[1]), "high": float(r[2]),
                  "low": float(r[3]), "close": float(r[4]),
                  "volume": float(r[6])} for r in rows])
            df["timestamp"] = _naive_utc(df["timestamp"])
            out = kraken_ohlcv_path(asset)   # [P420] never the primary
            if out.exists():
                old = pd.read_parquet(out)
                old["timestamp"] = _naive_utc(old["timestamp"])  # P414b: never mix tz
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


def run_scorer(window_days: int, record: bool = True) -> int:
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
    SCORER_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    per_asset_path = SCORER_REPORTS_DIR / f"shadow_ic_{stamp}_perasset.json"
    pooled_path = SCORER_REPORTS_DIR / f"shadow_ic_{stamp}_pooled.json"
    rc_asset = _run([sys.executable, "-X", "utf8", "-m",
                     "analytics.shadow_ic.compute_shadow_ic",
                     "--ledger-dir", str(LEDGER_DIR),
                     "--window-days", str(window_days),
                     "--output", str(per_asset_path)])
    print("")
    print("=" * 74)
    print("  POOLED READ — this is the one the P166 gate can fire on for a")
    print("  POOLABLE family; the per-asset table above is diagnosis (P299).")
    print("=" * 74)
    rc_pooled = _run([sys.executable, "-X", "utf8", "-m",
                      "analytics.shadow_ic.compute_shadow_ic",
                      "--ledger-dir", str(LEDGER_DIR),
                      "--window-days", str(window_days),
                      "--pool-assets", "--output", str(pooled_path)])
    # A refusal in EITHER read is a refusal: a missing pooled read must not be
    # invisible behind a healthy per-asset one (P199 — "no data" is not "no
    # signal", one level up).
    rc = rc_asset or rc_pooled
    if rc == 0 and record:
        # [P421e] Record the reads this run actually SAT — only the candidates
        # whose full accrual the window covered. The rest keep shouting.
        recorded, skipped = record_reads_taken(window_days, per_asset_path,
                                               pooled_path)
        print(f"\n[P421e] reads recorded in {READS_TAKEN_PATH.name}: "
              f"{', '.join(recorded) or '(none)'}")
        if skipped:
            print(f"[P421e] NOT recorded (window {window_days}d shorter than "
                  f"the ledger span, or clock unstarted): {', '.join(skipped)}"
                  f" -> re-run with --window-days >= the span to take those reads")
    return rc


def run_seat_read(window_days: int) -> int:
    """[P421e] Score the regimebook DECIDE candidate from its OWN ledger.

    scripts/seat_check.py scores the seat from agent_ic_review's `quant`
    series, which is the live decider's claims AFTER the seats ran. When the
    book has said flat all window that series is a constant, its IC is
    undefined, and the controller correctly REFUSES — so the one verdict it
    was built to reach (flat wins by comparison, P271) was unreachable from a
    flat book. The pooled regimebook ledger is the book's own claim series,
    the exact thing the September exam scored; scored at 1 and 4 bars (the
    controller's two horizons) it is the evidence the controller needs. The
    verdict is printed, NEVER applied (P141).
    """
    SCORER_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rc = _run([sys.executable, "-X", "utf8", "-m",
               "analytics.shadow_ic.compute_shadow_ic",
               "--ledger-dir", str(LEDGER_DIR),
               "--window-days", str(window_days),
               "--horizons", "1,4", "--prefixes", "regimebook",
               "--pool-assets", "--output", str(SEAT_REPORT_PATH)])
    if rc not in (0, 3):
        print(f"[P421e] seat read: scorer rc={rc}; seat controller not run")
        return 2
    return _run([sys.executable, "-X", "utf8",
                 str(REPO / "scripts" / "seat_check.py"),
                 "--shadow-ic-report", str(SEAT_REPORT_PATH),
                 "--ledger-dir", str(LEDGER_DIR)])


def run_tripwire() -> int:
    return _run([sys.executable, "-X", "utf8",
                 str(REPO / "analytics" / "calibration" / "tripwire_check.py"),
                 "--reports-dir", str(REPORTS_DIR)])


def countdown(return_due: bool = False, today_override=None):
    """Print the per-candidate countdown; with `return_due`, report whether any
    read is DUE.

    [P333] Pure date arithmetic over the CANDIDATES table — no ledgers, no
    parquets, no network — which is why it can run in-container while the rest
    of this script cannot (the scorer needs the operator's OHLCV, P213).

    That matters because the September reads had DATES and no alarm. The P237
    trend tripwire is croned and shouts on its own date; the other five read
    dates (09-07 through 09-16) lived only in a doc and in this script, which
    nobody runs unless they remember to. "A decision procedure that depends on
    someone re-running a tool by hand quietly becomes never" is this repo's own
    finding (P230), and it is the reason the P199 shadow-IC gate sat unrun for
    months.
    """
    today = (today_override
             or datetime.now(timezone.utc).date())
    due = []
    taken = load_reads_taken()
    print("\n" + "=" * 74)
    print(f"  SEPTEMBER COUNTDOWN (today {today})   "
          f"decision tree: docs/SEPTEMBER_DECISION_TREE.md")
    print("=" * 74)
    print(f"  {'candidate':<16} {'ledger since':<13} {'30d read date':<14} "
          f"{'days left':<10} on PASS")
    for name, (prefix, since, action) in CANDIDATES.items():
        try:
            d0 = datetime.strptime(since, "%Y-%m-%d").date()
        except ValueError:
            # [P420] a candidate whose clock has not started (ledger-since is
            # set at deploy) is listed, never silently dropped — an unstarted
            # clock is not a clock that cannot fire (P199).
            print(f"  {name:<16} {since:<13} {'(clock unstarted)':<14} "
                  f"{'-':<10} {action}")
            continue
        read = d0.toordinal() + 30
        left = read - today.toordinal()
        read_s = datetime.fromordinal(read).date().isoformat()
        # [P421e] A read that was TAKEN is reported as taken — date, window
        # and the scorer's own verdict — and never counted DUE again.
        # A read taken AFTER the reference date does not count: a --today
        # override must see the world as it stood on that date.
        t = taken.get(name)
        if t and t.get("read_on") and str(t["read_on"]) <= today.isoformat():
            print(f"  {name:<16} {since:<13} {t['read_on']:<14} "
                  f"{'TAKEN':<10} {_verdict_line(t)}")
            continue
        print(f"  {name:<16} {since:<13} {read_s:<14} "
              f"{max(0, left):<10} {action}")
        if left <= 0:
            due.append((name, read_s, action))
    # [P420] the P237 prescription (remove an asset from trend_assets) was
    # RETIRED by P299 — tripwire_check.py exits 0/2 only and prints
    # "do NOT edit trend_assets"; the streak is evidence for the seat
    # controller's comparison (scripts/seat_check.py). The old line here still
    # told the operator to do the retired thing.
    print(f"  {'trend tripwire':<16} {'weekly crons':<13} {'2026-09-01':<14} "
          f"{max(0, datetime(2026, 9, 1).date().toordinal() - today.toordinal()):<10} "
          f"DETECTION ONLY: streak feeds seat_check; NO trend_assets edit (P299)")
    if due:
        print("")
        print("!" * 74)
        print(f"  {len(due)} READ(S) ARE DUE. They do NOT run themselves — the")
        print("  scorer needs the operator's OHLCV parquets (P213). Run:")
        print("      python -X utf8 scripts/september_check.py --window-days 30")
        print("  then apply docs/SEPTEMBER_DECISION_TREE.md. Promotion is a")
        print("  config change a human makes (P141); this only tells you the")
        print("  window closed.")
        for name, read_s, action in due:
            print(f"    - {name}: read date {read_s} -> {action}")
        print("!" * 74)
    if return_due:
        return due
    return None


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
    ap.add_argument("--today", default=None,
                    help="ISO date override (testing the DUE path)")
    ap.add_argument("--countdown-only", action="store_true",
                    help="[P333] print the countdown and exit 3 if any read is "
                         "DUE; runs anywhere (no ledgers/parquets/network)")
    ap.add_argument("--no-pull", action="store_true",
                    help="score already-pulled ledgers")
    args = ap.parse_args()
    if args.countdown_only:
        # [P333] The alarm half: no pull, no refresh, no scoring — so it runs
        # in-container on a cron. Exit 3 = a read window has CLOSED and a human
        # must run the real exam; 0 = nothing due yet.
        _t = (datetime.strptime(args.today, "%Y-%m-%d").date()
              if args.today else None)
        due = countdown(return_due=True, today_override=_t)
        return 3 if due else 0
    if not args.no_pull:
        print("[1/6] pulling ledgers + reports from the server...")
        if not pull_from_server():
            print("REFUSING to score without a fresh pull — stale local "
                  "ledgers score yesterday's system (P255: local data/ is "
                  "never live evidence). Use --no-pull to override.")
            return 2
    print("[2/6] refreshing home-asset OHLCV...")
    prices_rc = refresh_prices()
    print("[3/6] building breadth-asset OHLCV from Kraken...")
    breadth_failed = build_breadth_ohlcv()
    print(f"[4/6] scoring all candidates (window={args.window_days}d)...")
    rc = run_scorer(args.window_days)
    print("[5/6] tripwire status...")
    tw_rc = run_tripwire()
    print("[6/6] seat controller on the pooled regimebook ledger (P421e)...")
    seat_rc = run_seat_read(args.window_days)
    countdown()
    # [P287] one unmissable RESULT line — step failures used to be `!!`
    # noise mid-scroll above a normal-looking verdict table, and the
    # tripwire's exit code was dropped entirely.
    # [P421e] rc 0 is NOT "nothing happened": P299 made the checker
    # detection-only, so 0 means "streak reported above" (it printed 4/4
    # GATE-CLOSED on all three assets while this line said not fired).
    tw_txt = {0: "detection-only, see streak lines (P299)", 2: "REFUSED (reports unreadable/missing)",
              3: "*** FIRED ***"}.get(tw_rc, f"rc={tw_rc}")
    seat_txt = {0: "incumbent holds", 2: "REFUSED",
                3: "*** SWITCH recommended ***"}.get(seat_rc, f"rc={seat_rc}")
    print(f"\nRESULT: scorer rc={rc} | tripwire {tw_txt} | seat {seat_txt} | "
          f"price refresh {'OK' if prices_rc == 0 else 'FAILED'} | "
          f"breadth OHLCV failures {breadth_failed}"
          + ("  <- some price series are stale; forward joins shrink "
             "accordingly" if (prices_rc != 0 or breadth_failed) else ""))
    return _final_rc(rc, tw_rc)


if __name__ == "__main__":
    sys.exit(main())
