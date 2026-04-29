#!/usr/bin/env python3
"""
compute_strategy_correlation.py — pairwise agent correlation matrix (P131, v3 1.4)
================================================================================

v3 Track A item 1.4 (P1-6). Reads synced IC signal records and computes
pairwise Pearson correlation across the 12 fusion-agent direction signals.

**Scope clarification vs v1 P1-6**:
  v1 wanted 12×12 correlation among kraken_quant strategies. The IC
  stream's `agent_signals` block tracks the 12 FUSION AGENTS (quant,
  drl, sentiment, microstructure, whale, etc.), not the per-strategy
  kraken_quant breakdown. Per-kraken_quant-strategy correlation requires
  a different log source (kq_firing_stats.jsonl from P128) which only
  has 1 day of data; deferred to a separate follow-up.

  This script answers the related question: "are the 12 fusion agents
  12 independent alpha sources or collinear?" Same shape, slightly
  different resolution.

**Output**:
  - `analytics/ic/reports/agent_correlation_matrix_<date>.json` — summary
    + high-corr pairs + effective independent source count via PCA
  - `analytics/ic/reports/agent_corr_matrix_<asset>_<date>.csv` — raw
    matrix per asset

**Decision rule per v3 prompt**:
  - effective_independent_sources <= 4: 12 agents are <=4 alpha sources
    in disguise. v1 "add more strategies" thesis dies.
  - effective_independent_sources >= 8: real diversity. v1 thesis lives,
    but needs IC heatmap to decide WHICH categories.
  - 5-7: gray zone, manual pair-by-pair review.

Read-only analysis. No trade impact.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_synced_ic_signals(sync_dir: Path) -> pd.DataFrame:
    """Load all ic_signals_*.jsonl files under sync_dir/ic_signals/.

    Returns long-form DataFrame: one row per (timestamp, asset, agent)
    with columns [timestamp, asset, agent, direction, confidence].
    """
    ic_dir = sync_dir / "ic_signals"
    if not ic_dir.exists():
        raise FileNotFoundError(f"IC signals dir missing: {ic_dir}")

    rows: List[Dict] = []
    for fpath in sorted(ic_dir.glob("ic_signals_*.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = rec.get("timestamp")
                asset = rec.get("asset", "?")
                agents = rec.get("agent_signals") or {}
                for agent_name, agent_data in agents.items():
                    if not isinstance(agent_data, dict):
                        continue
                    rows.append({
                        "timestamp": ts,
                        "asset": asset,
                        "agent": agent_name,
                        "direction": float(agent_data.get("direction", 0.0) or 0.0),
                        "confidence": float(agent_data.get("confidence", 0.0) or 0.0),
                    })

    if not rows:
        raise ValueError(f"No IC signal records found under {ic_dir}")
    return pd.DataFrame(rows)


def compute_correlation_per_asset(
    df: pd.DataFrame, signal_col: str = "direction"
) -> Dict[str, pd.DataFrame]:
    """Per-asset pairwise Pearson correlation across agents."""
    out: Dict[str, pd.DataFrame] = {}
    for asset in sorted(df["asset"].unique()):
        sub = df[df["asset"] == asset]
        # Pivot: rows = timestamp, cols = agent, values = direction
        pivot = sub.pivot_table(
            index="timestamp", columns="agent", values=signal_col, aggfunc="last"
        )
        # Drop columns with zero variance (would produce NaN in corr)
        nonzero_var = [c for c in pivot.columns if pivot[c].std() > 0]
        if len(nonzero_var) < 2:
            out[asset] = pd.DataFrame()
            continue
        out[asset] = pivot[nonzero_var].corr(method="pearson")
    return out


def report_per_asset(corr: pd.DataFrame, threshold: float = 0.7) -> Dict:
    """High-corr pairs + effective independent source count via PCA."""
    if corr.empty or len(corr) < 2:
        return {
            "n_agents": int(len(corr)),
            "high_corr_pairs": [],
            "effective_independent_sources": 0,
            "eigenvalue_spectrum": [],
            "note": "INSUFFICIENT_DATA: <2 agents with non-zero variance",
        }

    n = len(corr)
    high_corr_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            rho = corr.iloc[i, j]
            if pd.notna(rho) and abs(rho) > threshold:
                high_corr_pairs.append({
                    "a": str(corr.index[i]),
                    "b": str(corr.columns[j]),
                    "rho": round(float(rho), 4),
                })

    # PCA: count eigenvalues > 1.0 (Kaiser criterion / unit-variance random
    # matrix noise floor). Replace NaN with 0 for the eigvalsh computation.
    matrix = corr.fillna(0).values
    try:
        eigvals = np.linalg.eigvalsh(matrix)
        effective_independent = int(np.sum(eigvals > 1.0))
        eigvals_list = [round(float(v), 4) for v in eigvals.tolist()]
    except np.linalg.LinAlgError:
        effective_independent = 0
        eigvals_list = []

    return {
        "n_agents": int(n),
        "agents": list(corr.index),
        "high_corr_pairs_threshold": threshold,
        "high_corr_pairs": sorted(high_corr_pairs, key=lambda x: -abs(x["rho"])),
        "effective_independent_sources": effective_independent,
        "eigenvalue_spectrum": eigvals_list,
    }


def write_outputs(per_asset_corr: Dict[str, pd.DataFrame], reports: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sync_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Per-asset CSV matrices
    for asset, corr in per_asset_corr.items():
        if not corr.empty:
            corr.to_csv(out_dir / f"agent_corr_matrix_{asset}_{sync_date}.csv")

    # Summary JSON
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "P131 v3 1.4",
        "scope": "fusion-agent direction correlation (NOT per-kraken_quant-strategy)",
        "per_asset": reports,
    }
    out_path = out_dir / f"agent_correlation_matrix_{sync_date}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[P131] Summary written: {out_path}")
    return out_path


def print_human_report(reports: Dict):
    """Operator-readable summary."""
    print()
    print("=" * 70)
    print("FUSION-AGENT CORRELATION MATRIX REPORT")
    print("=" * 70)
    for asset, info in reports.items():
        print(f"\n{asset}:")
        if "note" in info:
            print(f"  {info['note']}")
            continue
        n = info["n_agents"]
        eff = info["effective_independent_sources"]
        print(f"  Agents (non-zero variance):    {n}")
        print(f"  Effective independent sources: {eff}")
        # Decision rule
        if eff <= 4:
            print(f"  -> v1 'add more strategies' thesis WEAKENED")
            print(f"     12 agents collapse to <={eff} alpha sources")
        elif eff >= 8:
            print(f"  -> Real agent diversity confirmed (eff>={eff})")
        else:
            print(f"  -> GRAY ZONE ({eff}/12); manual review of pair list below")

        pairs = info["high_corr_pairs"]
        if pairs:
            print(f"  High-correlation pairs (|rho| > {info['high_corr_pairs_threshold']}):")
            for p in pairs[:10]:
                print(f"    {p['a']:20s} <-> {p['b']:20s}  rho = {p['rho']:+.3f}")
        else:
            print(f"  No pairs with |rho| > {info['high_corr_pairs_threshold']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sync-dir", type=Path,
        default=Path(f"data/audit_sync/{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"),
        help="Path to sync directory (output of scripts/sync_audit_data.sh)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("analytics/ic/reports"),
        help="Output directory for matrix CSVs + summary JSON",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7,
        help="Pairwise |rho| threshold for high-corr flag",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.sync_dir.exists():
        print(f"FAIL: sync dir not found: {args.sync_dir}", file=sys.stderr)
        print(f"Run scripts/sync_audit_data.sh first.", file=sys.stderr)
        sys.exit(2)

    df = load_synced_ic_signals(args.sync_dir)
    print(f"[P131] Loaded {len(df):,} signal records "
          f"({df['agent'].nunique()} unique agents, "
          f"{df['asset'].nunique()} assets, "
          f"{df['timestamp'].nunique()} unique timestamps)")

    per_asset_corr = compute_correlation_per_asset(df, signal_col="direction")
    reports = {asset: report_per_asset(corr, threshold=args.threshold)
               for asset, corr in per_asset_corr.items()}

    out_path = write_outputs(per_asset_corr, reports, args.out_dir)
    if not args.quiet:
        print_human_report(reports)

    sys.exit(0)


if __name__ == "__main__":
    main()
