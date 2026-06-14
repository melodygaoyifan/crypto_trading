"""
Sharpe validation harness (Layer 3 of the alpha/beta end-to-end fix).

Purpose: objectively answer "is this model overfit?" BEFORE trusting it live —
the discipline that should have caught the DRL backtest Sharpe +9.22 -> live
break-even gap (CLAUDE.md P143; the live v5.1 promotion has ZERO validation).

Methods (deep-research R3, verified 3-0; memory crypto-quant-research-2026-06):
  - Probabilistic Sharpe Ratio (PSR): probability the TRUE Sharpe exceeds a
    benchmark, correcting for sample length, skew, and kurtosis. Bailey & Lopez
    de Prado. THIS IS THE ROBUST CORE.
  - Deflated Sharpe Ratio (DSR): PSR against the EXPECTED-MAXIMUM Sharpe under N
    independent trials (multiple-testing / selection-bias haircut).
    CAVEAT: the deep-research pass REFUTED (0-3) the *exact* analytic E[SR_max]
    formula, so DSR here is provided as a SENSITIVITY indicator, clearly flagged,
    NOT as a precise gate. Lean on PSR + the empirical live-vs-backtest gap.

Analytics-only: this module never touches trading. Pure functions over a returns
array; no live dependencies; no scipy (uses an erf-based normal CDF).
"""

from __future__ import annotations

import math
from typing import Sequence, Dict, Any, Optional


def _ncdf(x: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation). p in (0,1)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _moments(returns: Sequence[float]):
    n = len(returns)
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / n
    sd = math.sqrt(var) if var > 0 else 0.0
    if sd == 0:
        return n, mean, sd, 0.0, 3.0
    skew = sum((r - mean) ** 3 for r in returns) / n / sd ** 3
    kurt = sum((r - mean) ** 4 for r in returns) / n / sd ** 4  # non-excess (normal=3)
    return n, mean, sd, skew, kurt


def sharpe(returns: Sequence[float]) -> float:
    """Per-period (non-annualized) Sharpe."""
    n, mean, sd, _, _ = _moments(returns)
    return mean / sd if sd > 0 else 0.0


def annualized_sharpe(returns: Sequence[float], periods_per_year: float) -> float:
    return sharpe(returns) * math.sqrt(periods_per_year)


def probabilistic_sharpe_ratio(returns: Sequence[float], sr_benchmark: float = 0.0) -> Dict[str, Any]:
    """PSR = P(true per-period SR > sr_benchmark). sr_benchmark is per-period.
    Returns the probability + the inputs. >0.95 = confident the edge is real."""
    n, mean, sd, skew, kurt = _moments(returns)
    if n < 3 or sd == 0:
        return {"psr": 0.0, "sr": 0.0, "n": n, "note": "insufficient/degenerate"}
    sr = mean / sd
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2))
    z = (sr - sr_benchmark) * math.sqrt(n - 1.0) / denom
    return {"psr": _ncdf(z), "sr": sr, "skew": skew, "kurt": kurt, "n": n,
            "sr_benchmark": sr_benchmark}


def expected_max_sharpe(n_trials: int, sr_variance_across_trials: float) -> float:
    """Expected maximum per-period Sharpe under n_trials (selection-bias benchmark).
    *** CAVEAT: the exact analytic form was REFUTED (0-3) in the research pass.
    Treat the DSR built on this as a SENSITIVITY indicator, not a precise gate. ***
    """
    if n_trials < 2 or sr_variance_across_trials <= 0:
        return 0.0
    emc = 0.5772156649015329  # Euler-Mascheroni
    sd = math.sqrt(sr_variance_across_trials)
    return sd * ((1 - emc) * _ppf(1 - 1.0 / n_trials) + emc * _ppf(1 - 1.0 / (n_trials * math.e)))


def deflated_sharpe_ratio(returns: Sequence[float], n_trials: int,
                          sr_variance_across_trials: float) -> Dict[str, Any]:
    """DSR = PSR against the expected-max-Sharpe benchmark (multiple-testing).
    SENSITIVITY indicator (see expected_max_sharpe caveat)."""
    bench = expected_max_sharpe(n_trials, sr_variance_across_trials)
    out = probabilistic_sharpe_ratio(returns, sr_benchmark=bench)
    out["dsr"] = out.pop("psr")
    out["expected_max_sr_benchmark"] = bench
    out["_caveat"] = "exact E[SR_max] formula refuted 0-3; treat as sensitivity, not a gate"
    return out


def backtest_vs_live(claimed_backtest_sharpe_annual: float,
                     live_returns: Sequence[float], periods_per_year: float,
                     n_trials: Optional[int] = None) -> Dict[str, Any]:
    """Compare a claimed backtest Sharpe to the live realized Sharpe + PSR.
    A large positive gap (backtest >> live) with low live PSR = overfit signature."""
    live_ann = annualized_sharpe(live_returns, periods_per_year)
    psr = probabilistic_sharpe_ratio(live_returns, sr_benchmark=0.0)
    gap = claimed_backtest_sharpe_annual - live_ann
    # an annualized backtest Sharpe > ~4 is itself a strong overfit red flag
    implausible = claimed_backtest_sharpe_annual >= 4.0
    verdict = "OVERFIT_SUSPECT" if (gap > 3.0 and psr["psr"] < 0.95) else (
        "LIVE_UNCONFIRMED" if psr["psr"] < 0.95 else "LIVE_CONFIRMED")
    return {
        "claimed_backtest_sharpe_annual": claimed_backtest_sharpe_annual,
        "live_sharpe_annual": live_ann,
        "gap": gap,
        "live_psr_gt0": psr["psr"],
        "live_n": psr["n"],
        "backtest_implausible_flag": implausible,
        "verdict": verdict,
    }
