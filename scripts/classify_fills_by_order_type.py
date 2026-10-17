"""[P1R-3] Where the maker share comes from — fills x order type x intent x month.

    python -X utf8 scripts/classify_fills_by_order_type.py --fills data/phase0/fills_raw.parquet

Read-only over the Phase 0 frame (the venue's own fills joined to
data/fill_quality.jsonl). Answers, from the data:
  1. which intent classes the TAKER fills come from;
  2. whether a post-only request actually reached the venue as maker
     (every fill whose fill-quality label is "maker" was placed post-only
     by the maker ladder — if the flag were dropped some would come back
     TAKER);
  3. how often the maker ladder timed out into a taker cross;
  4. what V15's 98.7% measured (it is reproduced as the LIMIT ORDER-TYPE
     share, not a fill-liquidity share).
Writes reports/phase1r_maker_attribution.md. Exit 2 if the fills are absent.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Maker-first went live 2026-08-15 23:xx UTC (P270 activation, ee1dd9d); the
# fill-quality ledger (liquidity labels) from 2026-08-16 (P290).
MAKER_FIRST_FROM = "2026-08-16"
CURRENT_CONFIG_FROM = {"ETH/SOL stack (P416/P417/P419)": "2026-08-27",
                       "BTC skew seat (P424)": "2026-10-13"}


def _md(df) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    idx = [" / ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in df.index]
    cols = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    out = ["| " + " | ".join([str(df.index.name or "")] + cols) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for i, (_, row) in zip(idx, df.iterrows()):
        out.append("| " + " | ".join([i] + [str(v) for v in row.tolist()]) + " |")
    return chr(10).join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fills", default=os.path.join(REPO, "data", "phase0", "fills_raw.parquet"))
    ap.add_argument("--out", default=os.path.join(REPO, "reports", "phase1r_maker_attribution.md"))
    a = ap.parse_args(argv)
    if not os.path.exists(a.fills):
        print(f"REFUSING: {a.fills} missing (Phase 0 output)"); return 2
    import pandas as pd
    F = pd.read_parquet(a.fills)
    F["month"] = F["iso"].str[:7]
    F["day"] = F["iso"].str[:10]
    F["path"] = F["fq_liquidity"].fillna("pre_ledger")
    L = []
    w = L.append
    n = len(F); mk = int((F.liquidity_indicator == "MAKER").sum())
    w("# HMATS Phase 1R — R-3 maker-share attribution\n")
    w(f"{n} venue fills 2026-06-13 → 2026-10-15; MAKER {mk} ({mk / n:.1%}), TAKER {n - mk}.\n")
    w("## Liquidity × month\n")
    w(pd.crosstab(F.month, F.liquidity_indicator, margins=True).pipe(_md) + "\n")
    w("## Liquidity × intent class (Phase 0 inference)\n")
    w(pd.crosstab(F.cls, F.liquidity_indicator, margins=True).pipe(_md) + "\n")
    w("## Liquidity × placement path (fill_quality label) × order type\n")
    w(pd.crosstab([F.path, F.order_type], F.liquidity_indicator, margins=True).pipe(_md) + "\n")
    post = F[F.day >= MAKER_FIRST_FROM]
    pm = int((post.liquidity_indicator == "MAKER").sum())
    w(f"## Before vs after maker-first ({MAKER_FIRST_FROM})\n")
    pre = F[F.day < MAKER_FIRST_FROM]
    w(f"- before: {len(pre)} fills, MAKER {int((pre.liquidity_indicator == 'MAKER').sum())} "
      f"(maker-first was OFF: every order was a 0.2%-through LIMIT, i.e. a taker by construction)\n")
    w(f"- after: {len(post)} fills, MAKER {pm} ({pm / max(len(post), 1):.1%})\n")
    lab = post[post.fq_liquidity.notna()]
    mk_lab = lab[lab.fq_liquidity == "maker"]
    w("\n## Q2 — does post-only reach the venue?\n")
    w(f"- fills placed by the maker ladder (post-only LIMIT): {len(mk_lab)}; venue verdict MAKER on "
      f"{int((mk_lab.liquidity_indicator == 'MAKER').sum())}, TAKER on "
      f"{int((mk_lab.liquidity_indicator == 'TAKER').sum())}.\n")
    w("- code path: CoinbaseSleeve._maker_first_attempt / _maker_reprice_step build "
      "OrderRequest(post_only=True) -> CoinbaseAdapter.place_order -> "
      "client.limit_order_gtc(post_only=bool(request.post_only)); the P270 probe "
      "showed the venue REJECTS a crossing post-only order.\n")
    tc = int((lab.fq_liquidity == "taker_cross").sum())
    w("\n## Q3 — maker-ladder timeouts\n")
    w(f"- ladder outcomes with a fill (after {MAKER_FIRST_FROM}): maker {len(mk_lab)}, timed out into "
      f"a taker cross {tc} -> fallback rate {tc / max(len(mk_lab) + tc, 1):.1%} of ladder fills.\n")
    w("\n## Q1 — where the TAKER fills come from (after maker-first)\n")
    tk = post[post.liquidity_indicator == "TAKER"]
    w(pd.crosstab(tk.path, tk.cls, margins=True).pipe(_md) + "\n")
    w("\n## Q4 — V15's 98.7%\n")
    lim = int((F.order_type == "LIMIT").sum())
    w(f"- LIMIT order-type share of these fills: {lim}/{n} = {lim / n:.1%}; LIMIT share of ORDERS: "
      f"{F.drop_duplicates('order_id').order_type.eq('LIMIT').mean():.1%}.\n")
    w("- V15 / analytics/sixty_day_review/review_aggregator.maker_fee_ratio = n_limit / n_classified "
      "(ORDER TYPE), and training/backtest_framework FeeSchedule.maker_pct=0.987 cites it. A "
      "0.2%-through LIMIT is a TAKER fill; the ratio counted intent, not the venue's verdict "
      "(the P166 finding). The venue liquidity share is the 34.6% above.\n")
    for k, d in CURRENT_CONFIG_FROM.items():
        c = F[F.day >= d]
        cm = int((c.liquidity_indicator == "MAKER").sum())
        w(f"\n## Current configuration — {k}, since {d}\n")
        w(f"- {len(c)} fills, MAKER {cm} ({cm / max(len(c), 1):.1%})"
          + (" — INSUFFICIENT (n<10)" if len(c) < 10 else "") + "\n")
        w(pd.crosstab([c.path, c.cls], c.liquidity_indicator).pipe(_md) + "\n")
    md = "\n".join(L)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
