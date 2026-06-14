"""
Combinatorial Purged CV + CSCV Probability of Backtest Overfitting (PBO).

This is the validation METHODOLOGY the DRL training lacks. `train_drl_full.py`
selects `best_fold = max(folds, key=mean_reward)` — picking the best of N folds
is textbook selection bias (López de Prado's #1 cause of inflated backtests:
backtest Sharpe 9-10 -> live break-even). The fix is to (a) generate MANY
out-of-sample paths via combinatorial splits and (b) estimate the probability
that the in-sample-best configuration is actually overfit (PBO).

Methods (deep-research R3, verified 3-0; memory crypto-quant-research-2026-06):
  - cscv_pbo: Bailey & López de Prado, "The Probability of Backtest Overfitting"
    (J. Computational Finance, 2017). PBO = fraction of combinatorial IS/OOS
    splits where the IS-best config ranks below the OOS median. PBO < 0.5 = the
    selection generalizes; PBO -> 1 = the "best" is overfit noise.
  - combinatorial_purged_splits: C(n_groups, k) train/test index sets with a
    purge+embargo gap (no train bar within `embargo` of any test bar).

Analytics-only; pure python (itertools + math); no sklearn/scipy.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import List, Sequence, Tuple, Dict, Any


def _sharpe(rets: Sequence[float]) -> float:
    n = len(rets)
    if n < 3:
        return 0.0
    m = sum(rets) / n
    var = sum((x - m) ** 2 for x in rets) / n
    sd = math.sqrt(var)
    return (m / sd) if sd > 1e-12 else 0.0


def combinatorial_purged_splits(
    n_obs: int, n_groups: int = 6, n_test_groups: int = 2, embargo: int = 0
) -> List[Tuple[List[int], List[int]]]:
    """Yield (train_idx, test_idx) for every C(n_groups, n_test_groups) choice of
    test groups, purging train indices within `embargo` bars of any test group."""
    bounds = [round(i * n_obs / n_groups) for i in range(n_groups + 1)]
    groups = [list(range(bounds[i], bounds[i + 1])) for i in range(n_groups)]
    out = []
    for test_combo in combinations(range(n_groups), n_test_groups):
        test_idx = sorted(i for g in test_combo for i in groups[g])
        test_set = set(test_idx)
        # purge+embargo: drop train bars within `embargo` of any test bar
        banned = set(test_idx)
        if embargo > 0:
            for t in test_idx:
                for d in range(1, embargo + 1):
                    banned.add(t - d)
                    banned.add(t + d)
        train_idx = [i for i in range(n_obs) if i not in banned and i not in test_set]
        out.append((train_idx, test_idx))
    return out


def cscv_pbo(perf_matrix: List[Sequence[float]], n_splits: int = 10) -> Dict[str, Any]:
    """Probability of Backtest Overfitting via CSCV.

    perf_matrix: list of N configs, each a list of T per-period returns (aligned).
    Returns PBO + OOS degradation stats. PBO is the share of combinatorial
    in-sample/out-of-sample partitions in which the IS-best config underperforms
    the OOS median (logit < 0)."""
    n_configs = len(perf_matrix)
    if n_configs < 2:
        return {"error": "need >=2 configs for PBO"}
    T = min(len(r) for r in perf_matrix)
    if n_splits % 2 != 0:
        n_splits += 1
    # contiguous blocks
    bnd = [round(i * T / n_splits) for i in range(n_splits + 1)]
    blocks = [list(range(bnd[i], bnd[i + 1])) for i in range(n_splits)]
    logits = []
    oos_best = []
    oos_median_of_best = []
    half = n_splits // 2
    for is_combo in combinations(range(n_splits), half):
        is_rows = [i for b in is_combo for i in blocks[b]]
        oos_rows = [i for b in range(n_splits) if b not in is_combo for i in blocks[b]]
        is_sr = [_sharpe([perf_matrix[c][i] for i in is_rows]) for c in range(n_configs)]
        oos_sr = [_sharpe([perf_matrix[c][i] for i in oos_rows]) for c in range(n_configs)]
        nstar = max(range(n_configs), key=lambda c: is_sr[c])  # IS-best (the selection)
        # rank of the IS-best in OOS (1..N); relative rank omega in (0,1)
        rank = 1 + sum(1 for c in range(n_configs) if oos_sr[c] < oos_sr[nstar])
        omega = rank / (n_configs + 1)
        omega = min(1 - 1e-9, max(1e-9, omega))
        logits.append(math.log(omega / (1 - omega)))
        oos_best.append(oos_sr[nstar])
        srt = sorted(oos_sr)
        oos_median_of_best.append(srt[len(srt) // 2])
    pbo = sum(1 for x in logits if x < 0) / len(logits)
    return {
        "pbo": pbo,
        "n_partitions": len(logits),
        "n_configs": n_configs,
        "mean_oos_sharpe_of_is_best": sum(oos_best) / len(oos_best),
        "median_oos_sharpe_all": sum(oos_median_of_best) / len(oos_median_of_best),
        "verdict": ("OVERFIT_SELECTION" if pbo >= 0.5 else
                    "ROBUST_SELECTION" if pbo <= 0.2 else "MARGINAL"),
    }
