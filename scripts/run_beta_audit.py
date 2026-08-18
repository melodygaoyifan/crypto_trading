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
    """Load per-trade returns from exit_alpha_tracking.

    [P227b] A MISSING input file is a refusal, not an empty result. The old
    `return []` on absence made "no data source" indistinguishable from "no
    trades" (the P199/P183 conflation) — and it is exactly how this tool sat
    unrunnable through the Apr–Jun −25%: its last report predates the loss it
    existed to measure, because `data/exit_alpha_tracking.jsonl` never existed
    on this machine and nothing said so.
    """
    trades = []
    path = Path("data/exit_alpha_tracking.jsonl")
    if not path.exists():
        print(
            f"REFUSING TO REPORT: {path} does not exist on this machine.\n"
            f"This is 'no data source', not 'no trades' — a report from here "
            f"would look like a finding.\n"
            f"The file is written by the exit-alpha tracker on the LIVE "
            f"engine's data volume; run this in-container or scp the file "
            f"first. For sleeve-era beta, note the sleeve PnL series is "
            f"data/coinbase_sleeve_pnl.jsonl (different schema — this tool "
            f"does not read it yet).",
            file=sys.stderr,
        )
        sys.exit(2)
    with open(path, encoding="utf-8") as f:
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


def load_bar_level_returns():
    """Load 4H bar-level OHLCV returns from Kraken for stronger beta measurement.
    Returns dict: {asset: np.array of bar returns}. Uses existing CCXT client."""
    bar_returns = {}
    try:
        import requests, time as _time
        PAIRS = {"BTC": "XXBTZUSD", "ETH": "XETHZUSD", "SOL": "SOLUSD"}
        for asset, pair in PAIRS.items():
            since = int(_time.time()) - 720 * 14400  # ~120 days of 4H bars
            url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=240&since={since}"
            resp = requests.get(url, timeout=15)
            data = resp.json()
            if data.get("error"):
                continue
            ohlc_key = [k for k in data["result"] if k != "last"][0]
            bars = data["result"][ohlc_key]
            closes = np.array([float(b[4]) for b in bars])
            if len(closes) > 1:
                rets = np.diff(closes) / closes[:-1]
                bar_returns[asset] = rets[np.isfinite(rets)]
        logger.info(f"[BAR_LEVEL] Loaded: {', '.join(f'{a}={len(r)} bars' for a, r in bar_returns.items())}")
    except Exception as e:
        logger.warning(f"[BAR_LEVEL] OHLCV fetch failed: {e}")
    return bar_returns


def build_market_basket_returns(bar_returns):
    """Build equal-weight BTC/ETH/SOL basket returns from bar-level OHLCV."""
    assets = [a for a in ["BTC", "ETH", "SOL"] if a in bar_returns]
    if len(assets) < 2:
        return None
    min_len = min(len(bar_returns[a]) for a in assets)
    basket = np.zeros(min_len)
    for a in assets:
        basket += bar_returns[a][-min_len:] / len(assets)
    return basket


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

    # =====================================================================
    # [BP2+BP4] Bar-level beta measurement + market basket benchmark
    # Stronger statistical basis: 720 bars × 3 assets vs 18 sparse trades
    # =====================================================================
    bar_rets = load_bar_level_returns()
    basket_rets = build_market_basket_returns(bar_rets)
    bar_level_results = {}

    if bar_rets:
        logger.info(f"\n{'='*60}")
        logger.info(f"BAR-LEVEL BETA (4H OHLCV, ~120 days)")
        logger.info(f"{'='*60}")

        # Per-asset vs BTC buy-and-hold
        btc_rets = bar_rets.get("BTC")
        for asset in ["ETH", "SOL"]:
            asset_rets = bar_rets.get(asset)
            if asset_rets is None or btc_rets is None:
                continue
            min_n = min(len(asset_rets), len(btc_rets))
            if min_n < 42:
                continue
            report = analyzer.full_report(
                asset_rets[-min_n:], btc_rets[-min_n:],
                benchmark_name="BTC Buy&Hold",
                window=126,
            )
            bar_level_results[f"{asset}_vs_BTC"] = {
                "beta": round(report.overall_beta, 4),
                "correlation": round(report.overall_correlation, 4),
                "downside_beta": round(report.downside_beta, 4),
                "upside_capture": round(report.upside_capture, 3),
                "downside_capture": round(report.downside_capture, 3),
                "bars": min_n,
                "risk_level": report.beta_risk_level,
            }
            logger.info(
                f"  {asset} vs BTC: beta={report.overall_beta:.3f} corr={report.overall_correlation:.3f} "
                f"downside_beta={report.downside_beta:.3f} risk={report.beta_risk_level} ({min_n} bars)"
            )

        # Market basket benchmark
        if basket_rets is not None:
            for asset in ["BTC", "ETH", "SOL"]:
                asset_rets = bar_rets.get(asset)
                if asset_rets is None:
                    continue
                min_n = min(len(asset_rets), len(basket_rets))
                if min_n < 42:
                    continue
                report = analyzer.full_report(
                    asset_rets[-min_n:], basket_rets[-min_n:],
                    benchmark_name="EQ-Weight BTC/ETH/SOL Basket",
                    window=126,
                )
                bar_level_results[f"{asset}_vs_Basket"] = {
                    "beta": round(report.overall_beta, 4),
                    "correlation": round(report.overall_correlation, 4),
                    "downside_beta": round(report.downside_beta, 4),
                    "bars": min_n,
                    "risk_level": report.beta_risk_level,
                }
                logger.info(
                    f"  {asset} vs Basket: beta={report.overall_beta:.3f} corr={report.overall_correlation:.3f} "
                    f"downside_beta={report.downside_beta:.3f} ({min_n} bars)"
                )

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
            "downside_beta": round(port_report.downside_beta, 4),
            "alpha_annualized_pct": round(port_report.overall_alpha_annualized, 2),
            "strategy_return_pct": round(port_report.strategy_return, 2),
            "benchmark_return_pct": round(port_report.benchmark_return, 2),
            "upside_capture": round(port_report.upside_capture, 3),
            "downside_capture": round(port_report.downside_capture, 3),
            "beta_risk_level": port_report.beta_risk_level,
        },
        "bar_level_beta": bar_level_results if bar_rets else {},
        "per_asset": {},
    }
    for asset, report in asset_reports.items():
        result["per_asset"][asset] = {
            "beta": round(report.overall_beta, 4),
            "correlation": round(report.overall_correlation, 4),
            "downside_beta": round(report.downside_beta, 4),
            "alpha_annualized_pct": round(report.overall_alpha_annualized, 2),
            "upside_capture": round(report.upside_capture, 3),
            "downside_capture": round(report.downside_capture, 3),
            "risk_level": report.beta_risk_level,
            "regime_betas": report.regime_betas,
        }

    out_path = Path("reports/beta_alpha_audit.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
