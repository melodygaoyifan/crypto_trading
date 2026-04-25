"""
ALPHA_GATE rejection post-mortem.

For each alpha-gate rejection in the lookback window, dump the full
gate_details so we can see EXACTLY which input was below threshold,
whether DRL substitution fired, and what the effective alpha direction
was at decision time.

Output: pareto of WHICH FIELD IN GATE_DETAILS was at fault, plus N
sample raw rejections so the operator can eyeball the data.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_DIR = REPO_ROOT / "data" / "shadow_ledger"


def _iter_ledger(days: int) -> Iterable[Dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    if not LEDGER_DIR.exists():
        return
    for ledger_file in sorted(LEDGER_DIR.glob("ledger_*.jsonl")):
        try:
            file_date = datetime.strptime(
                ledger_file.stem.replace("ledger_", ""), "%Y%m%d"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if file_date < cutoff - timedelta(days=1):
            continue
        with open(ledger_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = entry.get("timestamp")
                if not ts:
                    continue
                try:
                    entry_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                except (ValueError, AttributeError):
                    continue
                if entry_dt < cutoff:
                    continue
                yield entry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--n-samples", type=int, default=10)
    args = p.parse_args()

    alpha_rejects = []
    for entry in _iter_ledger(args.days):
        if entry.get("entry_type") != "GATE_REJECT":
            continue
        reason = (entry.get("data") or {}).get("rejection_reason", "")
        if "alpha" not in reason.lower() and "min_alpha" not in reason.lower():
            continue
        alpha_rejects.append(entry)

    print(f"Found {len(alpha_rejects)} alpha-gate rejections in last {args.days} days\n")

    # Pareto by gate_details signature
    drl_dir_buckets = Counter()
    drl_conf_buckets = Counter()
    alpha_diff_buckets = Counter()
    strategy_to_drl = defaultdict(list)
    sub_path_evidence = Counter()  # was DRL substitution happening?

    for r in alpha_rejects:
        d = (r.get("data") or {}).get("gate_details") or {}
        drl_dir = d.get("drl_action") or 0.0
        drl_conf = d.get("drl_confidence") or 0.0
        alpha_est = d.get("alpha_estimated_bps") or 0.0
        alpha_thr = d.get("alpha_threshold_bps") or 0.0
        strategy = d.get("strategy") or "?"

        # Bucket DRL strength
        if abs(drl_dir) < 0.3:
            drl_dir_buckets["weak (<0.3)"] += 1
        elif abs(drl_dir) < 0.5:
            drl_dir_buckets["moderate (0.3-0.5)"] += 1
        elif abs(drl_dir) < 0.7:
            drl_dir_buckets["strong (0.5-0.7)"] += 1
        else:
            drl_dir_buckets["very strong (>=0.7)"] += 1

        if drl_conf < 0.2:
            drl_conf_buckets["very low (<0.2)"] += 1
        elif drl_conf < 0.3:
            drl_conf_buckets["low (0.2-0.3) — BELOW P20 threshold"] += 1
        elif drl_conf < 0.4:
            drl_conf_buckets["moderate (0.3-0.4)"] += 1
        elif drl_conf < 0.5:
            drl_conf_buckets["high (0.4-0.5)"] += 1
        else:
            drl_conf_buckets["very high (>=0.5)"] += 1

        diff = alpha_est - alpha_thr
        if alpha_thr <= 0:
            alpha_diff_buckets["threshold_zero_or_neg"] += 1
        elif alpha_est == 0 and alpha_thr > 0:
            alpha_diff_buckets["alpha_estimated_is_ZERO (P20 substitution NOT firing)"] += 1
        elif diff < -50:
            alpha_diff_buckets["very far below threshold (>50bps gap)"] += 1
        elif diff < -20:
            alpha_diff_buckets["far below (20-50bps gap)"] += 1
        elif diff < 0:
            alpha_diff_buckets["below (<20bps gap)"] += 1
        else:
            alpha_diff_buckets["above threshold (??)"] += 1

        strategy_to_drl[strategy].append((drl_dir, drl_conf))

        # P20 substitution evidence: when quant abstained AND drl was strong,
        # alpha_estimated_bps should reflect DRL's contribution. If alpha=0
        # despite strong drl, substitution didn't fire.
        quant_abstained = strategy == "hold" or alpha_est == 0
        drl_strong = abs(drl_dir) >= 0.5 and drl_conf >= 0.3
        if quant_abstained and drl_strong:
            if alpha_est > 0:
                sub_path_evidence["FIRED (alpha>0 with strong DRL)"] += 1
            else:
                sub_path_evidence["MISSING (alpha=0 with strong DRL ← BUG)"] += 1
        elif quant_abstained and not drl_strong:
            sub_path_evidence["NOT_APPLICABLE (drl below threshold)"] += 1

    print("=" * 78)
    print("DRL DIRECTION DISTRIBUTION at alpha-gate reject")
    print("=" * 78)
    for bucket, count in drl_dir_buckets.most_common():
        pct = 100 * count / max(1, len(alpha_rejects))
        print(f"  {bucket:30} {count:5}  {pct:5.1f}%")
    print()

    print("=" * 78)
    print("DRL CONFIDENCE DISTRIBUTION at alpha-gate reject")
    print("=" * 78)
    for bucket, count in drl_conf_buckets.most_common():
        pct = 100 * count / max(1, len(alpha_rejects))
        print(f"  {bucket:50} {count:5}  {pct:5.1f}%")
    print()

    print("=" * 78)
    print("ALPHA vs THRESHOLD DISTRIBUTION")
    print("=" * 78)
    for bucket, count in alpha_diff_buckets.most_common():
        pct = 100 * count / max(1, len(alpha_rejects))
        print(f"  {bucket:55} {count:5}  {pct:5.1f}%")
    print()

    print("=" * 78)
    print("P20 DRL SUBSTITUTION EVIDENCE")
    print("=" * 78)
    for bucket, count in sub_path_evidence.most_common():
        print(f"  {bucket:60} {count:5}")
    print()

    print("=" * 78)
    print(f"SAMPLE RAW REJECTIONS (last {args.n_samples})")
    print("=" * 78)
    for r in alpha_rejects[-args.n_samples:]:
        d = (r.get("data") or {}).get("gate_details") or {}
        reason = (r.get("data") or {}).get("rejection_reason", "")[:70]
        print(f"  {r.get('timestamp', '?')} {r.get('asset', '?')} {d.get('strategy', '?')}")
        print(f"    reason: {reason}")
        print(f"    drl_dir={d.get('drl_action')} drl_conf={d.get('drl_confidence')}")
        print(f"    alpha_est_bps={d.get('alpha_estimated_bps')} thresh_bps={d.get('alpha_threshold_bps')}")
        print(f"    regime={d.get('regime')} mode={d.get('mode')}")
        print()


if __name__ == "__main__":
    main()
