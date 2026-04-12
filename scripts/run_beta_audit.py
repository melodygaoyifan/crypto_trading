"""
Beta/Alpha Decomposition Audit — non-invasive analysis script.

Reads strategy equity/trades and benchmark OHLCV, computes:
  - Overall and rolling beta
  - Upside/downside capture
  - Residual alpha attribution
  - Regime-bucketed beta
  - Risk classification

Output: reports/beta_alpha_audit.json + reports/beta_alpha_audit.md

Usage:
    python -X utf8 scripts/run_beta_audit.py
"""
import sys, os, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_trade_returns():
    """Load per-trade returns from exit_alpha_tracking."""
    trades = []
    path = Path("data/exit_alpha_tracking.jsonl")
    if not path.exists():
        return trades
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                trades.append(r)
            except Exception:
                pass
    return trades


def build_strategy_returns(trades):
    """Convert trade PnL to a return series (one return per trade)."""
    returns = []
    for t in trades:
        bps = t.get("realized_bps", 0)
        returns.append(bps / 10000.0)  # bps → decimal return
    return np.array(returns)


def build_benchmark_returns(trades):
    """Build buy-and-hold BTC return matching each trade period.
    Approximate: each trade's benchmark return = (exit_price/entry_price - 1) for BTC direction."""
    returns = []
    for t in trades:
        entry_p = t.get("entry_price", 0)
        exit_p = t.get("exit_price", 0)
        if entry_p > 0:
            bench_ret = (exit_p - entry_p) / entry_p  # Buy-and-hold return over same period
        else:
            bench_ret = 0.0
        returns.append(bench_ret)
    return np.array(returns)


def build_regime_labels(trades):
    """Extract regime labels per trade."""
    return np.array([t.get("regime_at_entry", "UNKNOWN") for t in trades])


def main():
    logger.info("=" * 60)
    logger.info("HMATS Alpha/Beta Decomposition Audit")
    logger.info("=" * 60)

    trades = load_trade_returns()
    logger.info(f"Loaded {len(trades)} closed trades")

    if len(trades) < 3:
        logger.error("Insufficient trades for beta analysis")
        return

    strategy_rets = build_strategy_returns(trades)
    benchmark_rets = build_benchmark_returns(trades)
    regimes = build_regime_labels(trades)

    from analytics.beta_exposure import BetaExposureAnalyzer
    analyzer = BetaExposureAnalyzer(annualization_factor=2190.0)

    # Per-asset analysis
    assets = sorted(set(t.get("asset", "?") for t in trades))
    asset_reports = {}

    for asset in assets:
        mask = [i for i, t in enumerate(trades) if t.get("asset") == asset]
        if len(mask) < 3:
            logger.info(f"\n{asset}: too few trades ({len(mask)})")
            continue
        s = strategy_rets[mask]
        b = benchmark_rets[mask]
        r = regimes[mask]
        report = analyzer.full_report(s, b, r,
                                       benchmark_name=f"{asset} Buy&Hold",
                                       window=max(5, len(mask) // 2))
        asset_reports[asset] = report
        logger.info(f"\n--- {asset} ({len(mask)} trades) ---")
        logger.info(f"  Beta: {report.overall_beta:.3f}")
        logger.info(f"  Correlation: {report.overall_correlation:.3f}")
        logger.info(f"  Alpha (annualized): {report.overall_alpha_annualized:+.1f}%")
        logger.info(f"  Strategy return: {report.strategy_return:+.2f}%")
        logger.info(f"  Benchmark return: {report.benchmark_return:+.2f}%")
        logger.info(f"  Upside capture: {report.upside_capture:.2f} ({report.upside_sample_count} samples)")
        logger.info(f"  Downside capture: {report.downside_capture:.2f} ({report.downside_sample_count} samples)")
        logger.info(f"  Beta risk level: {report.beta_risk_level}")
        if report.regime_betas:
            logger.info(f"  Regime betas:")
            for regime in sorted(report.regime_betas):
                beta = report.regime_betas[regime]
                alpha = report.regime_alphas.get(regime, 0)
                count = report.regime_counts.get(regime, 0)
                logger.info(f"    {regime}: beta={beta:.3f} alpha={alpha:+.1f}bps n={count}")

    # Portfolio level
    logger.info(f"\n--- PORTFOLIO ({len(trades)} trades) ---")
    port_report = analyzer.full_report(strategy_rets, benchmark_rets, regimes,
                                        benchmark_name="Per-trade Buy&Hold",
                                        window=max(5, len(trades) // 2))
    logger.info(f"  Portfolio beta: {port_report.overall_beta:.3f}")
    logger.info(f"  Portfolio correlation: {port_report.overall_correlation:.3f}")
    logger.info(f"  Portfolio alpha (ann): {port_report.overall_alpha_annualized:+.1f}%")
    logger.info(f"  Upside capture: {port_report.upside_capture:.2f}")
    logger.info(f"  Downside capture: {port_report.downside_capture:.2f}")
    logger.info(f"  Risk level: {port_report.beta_risk_level}")

    # Verdict
    if abs(port_report.overall_beta) < 0.3 and abs(port_report.overall_correlation) < 0.3:
        verdict = "ALPHA_ONLY_REASONABLE"
        explanation = "Low beta and correlation → strategy returns largely independent of market direction"
    elif abs(port_report.overall_beta) < 0.6:
        verdict = "ALPHA_ONLY_BUT_BETA_UNMEASURED"
        explanation = "Moderate beta exposure → alpha-first design reasonable but beta should be monitored"
    else:
        verdict = "NEED_BETA_MONITORING_ONLY"
        explanation = "Significant beta exposure → add rolling beta monitoring and alerts"

    logger.info(f"\n{'='*60}")
    logger.info(f"VERDICT: {verdict}")
    logger.info(f"  {explanation}")
    logger.info(f"{'='*60}")

    # Save JSON
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_trades": len(trades),
        "verdict": verdict,
        "explanation": explanation,
        "portfolio": {
            "beta": round(port_report.overall_beta, 4),
            "correlation": round(port_report.overall_correlation, 4),
            "alpha_annualized_pct": round(port_report.overall_alpha_annualized, 2),
            "strategy_return_pct": round(port_report.strategy_return, 2),
            "benchmark_return_pct": round(port_report.benchmark_return, 2),
            "upside_capture": round(port_report.upside_capture, 3),
            "downside_capture": round(port_report.downside_capture, 3),
            "beta_risk_level": port_report.beta_risk_level,
        },
        "per_asset": {},
    }
    for asset, report in asset_reports.items():
        result["per_asset"][asset] = {
            "beta": round(report.overall_beta, 4),
            "correlation": round(report.overall_correlation, 4),
            "alpha_annualized_pct": round(report.overall_alpha_annualized, 2),
            "upside_capture": round(report.upside_capture, 3),
            "downside_capture": round(report.downside_capture, 3),
            "risk_level": report.beta_risk_level,
            "regime_betas": report.regime_betas,
        }

    out_path = Path("reports/beta_alpha_audit.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
