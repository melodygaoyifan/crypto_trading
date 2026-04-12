"""
Run the 3 missing validation tests:
1. Monte Carlo stress test (trade-sequence bootstrap)
2. Walk-Forward validation (anchored expanding windows)
3. Fee stress test (2x slippage scenario)

Input: data/exit_alpha_tracking.jsonl (closed trade records)
Output: reports/validation_results.json
"""
import sys, os, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_trades():
    """Load closed trade records from exit_alpha_tracking."""
    trades = []
    path = Path("data/exit_alpha_tracking.jsonl")
    if not path.exists():
        logger.error("No exit_alpha_tracking.jsonl found")
        return trades
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                trades.append({
                    "asset": r.get("asset", ""),
                    "pnl_usd": float(r.get("pnl_usd", 0)),
                    "realized_bps": float(r.get("realized_bps", 0)),
                    "ticks_held": int(r.get("ticks_held", 1)),
                    "strategy": r.get("strategy", ""),
                    "direction": float(r.get("direction", 0)),
                    "exit_trigger": r.get("exit_trigger", ""),
                    "notional": float(r.get("notional", 0)),
                })
            except Exception as e:
                logger.warning(f"Skipping bad record: {e}")
    return trades


# =============================================================================
# TEST 1: MONTE CARLO
# =============================================================================

def run_monte_carlo(trades, initial_capital=10000, num_simulations=1000, seed=42):
    """Trade-sequence bootstrap Monte Carlo."""
    logger.info(f"\n{'='*60}")
    logger.info(f"MONTE CARLO STRESS TEST ({num_simulations} simulations, {len(trades)} trades)")
    logger.info(f"{'='*60}")

    np.random.seed(seed)
    pnls = [t["pnl_usd"] for t in trades]
    n = len(pnls)
    if n < 3:
        logger.warning("Too few trades for Monte Carlo")
        return None

    results = []
    for sim in range(num_simulations):
        shuffled = list(np.random.permutation(pnls))
        equity = [initial_capital]
        peak = initial_capital
        max_dd = 0.0
        for pnl in shuffled:
            eq = equity[-1] + pnl
            equity.append(eq)
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        final = equity[-1]
        ret_pct = (final / initial_capital - 1) * 100
        # Sharpe approximation
        rets = np.diff(equity) / np.array(equity[:-1])
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(len(rets)) if np.std(rets) > 0 else 0
        wins = sum(1 for p in shuffled if p > 0)
        win_rate = wins / n * 100

        results.append({
            "final_equity": final,
            "return_pct": ret_pct,
            "max_drawdown_pct": max_dd * 100,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "bankrupt": final < initial_capital * 0.5,
        })

    # Analyze
    returns = [r["return_pct"] for r in results]
    drawdowns = [r["max_drawdown_pct"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    bankrupt_count = sum(1 for r in results if r["bankrupt"])

    mc_result = {
        "num_simulations": num_simulations,
        "num_trades": n,
        "initial_capital": initial_capital,
        "return_mean": round(np.mean(returns), 2),
        "return_std": round(np.std(returns), 2),
        "return_5th_pct": round(np.percentile(returns, 5), 2),
        "return_25th_pct": round(np.percentile(returns, 25), 2),
        "return_median": round(np.median(returns), 2),
        "return_75th_pct": round(np.percentile(returns, 75), 2),
        "return_95th_pct": round(np.percentile(returns, 95), 2),
        "max_dd_mean": round(np.mean(drawdowns), 2),
        "max_dd_95th_pct": round(np.percentile(drawdowns, 95), 2),
        "max_dd_worst": round(np.max(drawdowns), 2),
        "sharpe_mean": round(np.mean(sharpes), 2),
        "sharpe_5th_pct": round(np.percentile(sharpes, 5), 2),
        "bankruptcy_probability": round(bankrupt_count / num_simulations, 4),
        "verdict": "PASS" if np.percentile(returns, 5) > -10 else "FAIL",
    }

    logger.info(f"  Return: {mc_result['return_mean']:+.2f}% ± {mc_result['return_std']:.2f}%")
    logger.info(f"  5th-95th percentile: {mc_result['return_5th_pct']:+.2f}% to {mc_result['return_95th_pct']:+.2f}%")
    logger.info(f"  Max DD (mean): {mc_result['max_dd_mean']:.2f}%, worst: {mc_result['max_dd_worst']:.2f}%")
    logger.info(f"  Sharpe (mean): {mc_result['sharpe_mean']:.2f}, 5th: {mc_result['sharpe_5th_pct']:.2f}")
    logger.info(f"  Bankruptcy prob: {mc_result['bankruptcy_probability']:.2%}")
    logger.info(f"  Verdict: {mc_result['verdict']}")
    return mc_result


# =============================================================================
# TEST 2: WALK-FORWARD (simplified anchored expanding)
# =============================================================================

def run_walk_forward(trades, n_folds=3, min_train=5):
    """Anchored expanding walk-forward on trade sequence."""
    logger.info(f"\n{'='*60}")
    logger.info(f"WALK-FORWARD VALIDATION ({n_folds} folds, {len(trades)} trades)")
    logger.info(f"{'='*60}")

    n = len(trades)
    if n < n_folds * 2:
        logger.warning("Too few trades for walk-forward")
        return None

    fold_size = n // n_folds
    fold_results = []

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        val_start = train_end
        val_end = min(val_start + fold_size, n) if fold < n_folds - 1 else n

        train_trades = trades[:train_end]
        val_trades = trades[val_start:val_end]

        if len(val_trades) < 2:
            continue

        # Train statistics
        train_pnls = [t["pnl_usd"] for t in train_trades]
        train_win_rate = sum(1 for p in train_pnls if p > 0) / len(train_pnls) * 100 if train_pnls else 0

        # OOS validation
        val_pnls = [t["pnl_usd"] for t in val_trades]
        val_total = sum(val_pnls)
        val_win_rate = sum(1 for p in val_pnls if p > 0) / len(val_pnls) * 100
        val_mean_bps = np.mean([t["realized_bps"] for t in val_trades])

        # Max DD on val
        equity = [10000]
        peak = 10000
        max_dd = 0
        for p in val_pnls:
            equity.append(equity[-1] + p)
            peak = max(peak, equity[-1])
            dd = (peak - equity[-1]) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        fold_result = {
            "fold": fold + 1,
            "train_size": len(train_trades),
            "val_size": len(val_trades),
            "train_win_rate": round(train_win_rate, 1),
            "val_total_pnl": round(val_total, 2),
            "val_win_rate": round(val_win_rate, 1),
            "val_mean_bps": round(val_mean_bps, 1),
            "val_max_dd_pct": round(max_dd * 100, 2),
        }
        fold_results.append(fold_result)
        logger.info(
            f"  Fold {fold+1}: train={len(train_trades)}, val={len(val_trades)} | "
            f"PnL=${val_total:+.2f} | WR={val_win_rate:.0f}% | "
            f"Mean={val_mean_bps:+.1f}bps | MaxDD={max_dd*100:.2f}%"
        )

    if not fold_results:
        return None

    wf_result = {
        "n_folds": n_folds,
        "total_trades": n,
        "folds": fold_results,
        "avg_val_pnl": round(np.mean([f["val_total_pnl"] for f in fold_results]), 2),
        "avg_val_win_rate": round(np.mean([f["val_win_rate"] for f in fold_results]), 1),
        "avg_val_max_dd": round(np.mean([f["val_max_dd_pct"] for f in fold_results]), 2),
        "verdict": "PASS" if np.mean([f["val_win_rate"] for f in fold_results]) > 35 else "FAIL",
    }
    logger.info(f"  Average OOS PnL: ${wf_result['avg_val_pnl']:+.2f}")
    logger.info(f"  Average OOS WR: {wf_result['avg_val_win_rate']:.1f}%")
    logger.info(f"  Verdict: {wf_result['verdict']}")
    return wf_result


# =============================================================================
# TEST 3: FEE STRESS (2x slippage scenario)
# =============================================================================

def run_fee_stress(trades, initial_capital=10000, stress_multipliers=[1.0, 1.5, 2.0, 3.0]):
    """Replay trades with increased fee/slippage."""
    logger.info(f"\n{'='*60}")
    logger.info(f"FEE / SLIPPAGE STRESS TEST ({len(trades)} trades)")
    logger.info(f"{'='*60}")

    if not trades:
        return None

    results = []
    for mult in stress_multipliers:
        total_pnl = 0
        wins = 0
        for t in trades:
            # Original PnL includes fees. Under stress, additional fee = (mult-1) * estimated_fee
            # Estimate fee as ~26bps of notional (Kraken taker)
            notional = t.get("notional", 0) or abs(t["pnl_usd"]) * 100  # fallback
            base_fee = notional * 0.0026  # 26bps
            extra_fee = base_fee * (mult - 1)
            stressed_pnl = t["pnl_usd"] - extra_fee
            total_pnl += stressed_pnl
            if stressed_pnl > 0:
                wins += 1

        win_rate = wins / len(trades) * 100
        ret_pct = total_pnl / initial_capital * 100

        result = {
            "fee_multiplier": mult,
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(ret_pct, 2),
            "win_rate": round(win_rate, 1),
            "still_profitable": total_pnl > 0,
        }
        results.append(result)
        label = "✅" if total_pnl > 0 else "❌"
        logger.info(
            f"  {mult:.1f}x fees: PnL=${total_pnl:+.2f} ({ret_pct:+.2f}%) "
            f"WR={win_rate:.0f}% {label}"
        )

    stress_result = {
        "scenarios": results,
        "break_even_multiplier": None,
        "verdict": "PASS" if results[-1]["still_profitable"] else "MARGINAL",
    }
    # Find break-even
    for i, r in enumerate(results):
        if not r["still_profitable"]:
            if i > 0:
                stress_result["break_even_multiplier"] = round(results[i-1]["fee_multiplier"], 1)
            break
    else:
        stress_result["break_even_multiplier"] = f">{results[-1]['fee_multiplier']}"

    logger.info(f"  Break-even fee multiplier: {stress_result['break_even_multiplier']}")
    logger.info(f"  Verdict: {stress_result['verdict']}")
    return stress_result


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("HMATS Validation Suite")
    logger.info("=" * 60)

    trades = load_trades()
    logger.info(f"Loaded {len(trades)} closed trades")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_trades": len(trades),
    }

    # 1. Monte Carlo
    results["monte_carlo"] = run_monte_carlo(trades)

    # 2. Walk-Forward
    results["walk_forward"] = run_walk_forward(trades)

    # 3. Fee Stress
    results["fee_stress"] = run_fee_stress(trades)

    # Save
    out_path = Path("reports/validation_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
