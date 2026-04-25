#!/usr/bin/env python3
"""
training/exit_drl/validate_against_baseline.py

Offline validator: replay a held-out time window through Exit-SAC vs a rule-based
baseline (mirrors `execution/exit_alpha.py` triggers in pure Python, no live deps)
and report per-action lift in:
  - Avg PnL per closed trade (bps)
  - Sharpe (rolling)
  - Hit rate (% winning trades)
  - Avg bars held
  - Action histogram

Used as the gate input for Stage-3 promotion (SHADOW → EXIT_ONLY). The promotion
threshold per HMATS_EXIT_DRL_TRAINING.md Stage 3 Layer 2 is **+10% Sharpe** vs
rule-based; this script computes the number, doesn't enforce it.

Usage:
  python -X utf8 training/exit_drl/validate_against_baseline.py \\
      --asset BTC --models-dir models/exit_drl_v2 --val-frac 0.2

Output: prints per-asset comparison + writes
  data/exit_drl_validation/{asset}_validation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure the repo root is importable when this script is launched directly
# (`python training/exit_drl/validate_against_baseline.py …`).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    print("[FATAL] torch required for validation.", file=sys.stderr)
    sys.exit(1)

# ────────────────────────────────────────────────────────────────
# Local imports — must match training pipeline
# ────────────────────────────────────────────────────────────────
from training.exit_drl.generate_expert_trajectories import (
    ACTION_HOLD, ACTION_PARTIAL_EXIT, ACTION_RELEASE_RUNNER, ACTION_EXIT_ALL,
    LOOKAHEAD_BARS, MIN_BARS_HELD, MAX_BARS_HELD, ENTRY_STRIDE,
    PARTIAL_PROFIT_MIN_BPS, POST_PEAK_DRAWDOWN_PCT, STOP_ATR_MULT,
    build_state,
)
from training.exit_drl.train_exit_sac import DiscreteActor, STATE_DIM, ACTION_DIM


# ────────────────────────────────────────────────────────────────
# Rule-based baseline — pure-Python mirror of exit_alpha.py triggers
# ────────────────────────────────────────────────────────────────
@dataclass
class BaselineConfig:
    """Mirrors `execution/exit_alpha.ScaleOutConfig` defaults that matter for
    a single-asset offline replay (no DRL trigger, no CRACK trigger — those
    require live signal dicts we don't have in the parquet)."""
    min_profit_bps: float = 100.0      # exit_alpha default
    min_bars_held: int = 3             # exit_alpha default
    momentum_stall_bars: int = 4       # exit_alpha default
    profit_drawdown_pct: float = 0.30  # exit_alpha default (peak give-back)
    runner_min_bars: int = 12          # if held > this, treat as runner
    stop_atr_mult: float = STOP_ATR_MULT


def baseline_action(
    entry_idx: int,
    current_idx: int,
    direction: int,
    prices: np.ndarray,
    atr_pct: float,
    cfg: BaselineConfig,
    momentum_buffer: List[float],
) -> int:
    """
    Single-step rule-based exit decision (mirrors exit_alpha.py minus DRL/CRACK
    triggers, which depend on live agent signals not in the offline parquet).

    Triggers (first match wins):
      A. Stop-out:     unrealized_pnl < -1.5*ATR                  → EXIT_ALL
      B. Hard horizon: bars_held >= MAX_BARS_HELD                  → EXIT_ALL
      C. Momentum stall after profit:
            profit >= 100bps AND bars_held >= 3 AND momentum stall → PARTIAL_EXIT
      D. Profit drawdown from peak:
            past_peak >= 100bps AND now < 0.7*past_peak            → RELEASE_RUNNER
      E. Otherwise                                                  → HOLD
    """
    bars_held = current_idx - entry_idx
    if bars_held >= MAX_BARS_HELD:
        return ACTION_EXIT_ALL
    if bars_held < MIN_BARS_HELD:
        return ACTION_HOLD

    entry_price = prices[entry_idx]
    if entry_price <= 0:
        return ACTION_HOLD
    current_price = prices[current_idx]
    unrealized_pnl = direction * (current_price - entry_price) / entry_price
    profit_bps = unrealized_pnl * 10000.0

    # A. Stop-out
    if unrealized_pnl < -cfg.stop_atr_mult * max(atr_pct, 1e-4):
        return ACTION_EXIT_ALL

    if bars_held < cfg.min_bars_held:
        return ACTION_HOLD

    # Past peak (from entry to current)
    past_pnls = direction * (prices[entry_idx:current_idx + 1] - entry_price) / entry_price
    past_peak = float(past_pnls.max()) if len(past_pnls) else 0.0

    # D. Profit drawdown from peak (release runner)
    if past_peak * 10000.0 >= cfg.min_profit_bps:
        if unrealized_pnl < past_peak * (1.0 - cfg.profit_drawdown_pct):
            return ACTION_RELEASE_RUNNER

    # C. Momentum stall after profit
    if profit_bps >= cfg.min_profit_bps and len(momentum_buffer) >= cfg.momentum_stall_bars:
        recent = momentum_buffer[-cfg.momentum_stall_bars:]
        # "Stall" = max abs delta in window < 25bps (tight band)
        if (max(recent) - min(recent)) * 10000.0 < 25.0:
            return ACTION_PARTIAL_EXIT

    return ACTION_HOLD


# ────────────────────────────────────────────────────────────────
# Trade simulator — applies an action sequence and returns realized PnL
# ────────────────────────────────────────────────────────────────
@dataclass
class TradeOutcome:
    entry_idx: int
    exit_idx: int
    direction: int
    pnl_bps: float
    bars_held: int
    final_action: int
    # Per-step action counts (HOLD / PARTIAL_EXIT / RELEASE_RUNNER / EXIT_ALL).
    # Counting only the *terminal* action across all trades is misleading
    # because PARTIAL_EXIT continues the trade — the terminal histogram
    # always converges to mostly EXIT_ALL even when the actor fires many
    # PARTIAL_EXIT / HOLD calls during the trajectory.
    action_counts: List[int] = field(default_factory=lambda: [0, 0, 0, 0])


def simulate_trade(
    actor_fn,                       # callable: (state, idx, entry_idx, direction, df) → action
    entry_idx: int,
    direction: int,
    df: pd.DataFrame,
    close: np.ndarray,
    atr_pct_at_entry: float,
) -> Optional[TradeOutcome]:
    """
    Simulate a single trade from `entry_idx` until the actor closes it.
    Returns None if the asset has insufficient lookahead.

    Position model (matches what oracle saw during training):
    - PARTIAL_EXIT  → keeps tracking same trajectory; we record the partial
                      contribution (25%) AT TIME OF ACTION using current PnL
                      and continue running the remaining 75% until terminal.
    - RELEASE_RUNNER → records 50% lock-in at current PnL, terminate (we treat
                       runner release as a terminal-with-runoff-already-priced).
    - EXIT_ALL      → records full PnL at current price, terminate.
    """
    n = len(df)
    if entry_idx + LOOKAHEAD_BARS >= n:
        return None

    momentum_buffer: List[float] = []
    realized_bps = 0.0   # accumulated realized return so far (bps)
    remaining_size = 1.0
    last_idx = entry_idx
    last_action = ACTION_HOLD
    step_action_counts = [0, 0, 0, 0]  # HOLD, PARTIAL_EXIT, RELEASE_RUNNER, EXIT_ALL

    for bar_offset in range(MIN_BARS_HELD, MAX_BARS_HELD + 1):
        current_idx = entry_idx + bar_offset
        if current_idx >= n:
            current_idx = n - 1
        last_idx = current_idx

        entry_price = close[entry_idx]
        current_price = close[current_idx]
        unrealized_bps = direction * (current_price - entry_price) / entry_price * 10000.0

        action = actor_fn(
            current_idx=current_idx, entry_idx=entry_idx, direction=direction,
            df=df, close=close, atr_pct=atr_pct_at_entry,
            momentum_buffer=momentum_buffer,
        )
        last_action = action
        step_action_counts[action] += 1

        if action == ACTION_HOLD:
            momentum_buffer.append(unrealized_bps / 10000.0)
            if len(momentum_buffer) > 12:
                momentum_buffer = momentum_buffer[-12:]
            continue

        if action == ACTION_PARTIAL_EXIT:
            realized_bps += 0.25 * unrealized_bps
            remaining_size -= 0.25
            momentum_buffer.append(unrealized_bps / 10000.0)
            # Continue trajectory; matches exit_alpha behavior allowing at most
            # 2 partials before treating the next non-HOLD as terminal.
            continue

        if action == ACTION_RELEASE_RUNNER:
            realized_bps += 0.5 * remaining_size * unrealized_bps
            remaining_size = 0.0
            return TradeOutcome(entry_idx, current_idx, direction,
                                realized_bps, bar_offset, action,
                                action_counts=step_action_counts)

        if action == ACTION_EXIT_ALL:
            realized_bps += remaining_size * unrealized_bps
            remaining_size = 0.0
            return TradeOutcome(entry_idx, current_idx, direction,
                                realized_bps, bar_offset, action,
                                action_counts=step_action_counts)

    # Force-close at horizon
    final_price = close[last_idx]
    final_bps = direction * (final_price - close[entry_idx]) / close[entry_idx] * 10000.0
    realized_bps += remaining_size * final_bps
    step_action_counts[ACTION_EXIT_ALL] += 1
    return TradeOutcome(entry_idx, last_idx, direction, realized_bps,
                        last_idx - entry_idx, ACTION_EXIT_ALL,
                        action_counts=step_action_counts)


# ────────────────────────────────────────────────────────────────
# Actor wrappers
# ────────────────────────────────────────────────────────────────
def make_baseline_actor(cfg: BaselineConfig):
    def actor(current_idx, entry_idx, direction, df, close, atr_pct, momentum_buffer):
        return baseline_action(
            entry_idx, current_idx, direction, close, atr_pct, cfg, momentum_buffer,
        )
    return actor


def make_drl_actor(checkpoint_path: Path, device: str, extra_roots=None):
    from infra.safe_torch_load import safe_torch_load
    ckpt = safe_torch_load(
        checkpoint_path, map_location=device, weights_only=False,
        extra_allowed_roots=extra_roots,
    )
    actor = DiscreteActor(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    val_align = ckpt.get("val_alignment", float("nan"))

    @torch.no_grad()
    def fn(current_idx, entry_idx, direction, df, close, atr_pct, momentum_buffer):
        state = build_state(current_idx, entry_idx, direction, df, close)
        state_t = torch.tensor(state, device=device).unsqueeze(0)
        probs, _ = actor(state_t)
        return int(probs.argmax(dim=-1).item())

    return fn, val_align


# ────────────────────────────────────────────────────────────────
# Aggregate metrics
# ────────────────────────────────────────────────────────────────
def aggregate(outcomes: List[TradeOutcome]) -> Dict:
    if not outcomes:
        return {"n_trades": 0}
    pnls = np.array([o.pnl_bps for o in outcomes], dtype=np.float64)
    held = np.array([o.bars_held for o in outcomes], dtype=np.int64)
    final_actions = np.array([o.final_action for o in outcomes], dtype=np.int64)

    mean_bps = float(pnls.mean())
    std_bps = float(pnls.std())
    sharpe = mean_bps / std_bps if std_bps > 1e-9 else 0.0
    win_rate = float((pnls > 0).mean())
    avg_held = float(held.mean())

    # Final action histogram = the action that closed the trade. This is what
    # the spec called "final_action_dist" — useful but misleading on its own
    # because PARTIAL_EXIT continues the trade so it never appears here.
    final_dist = np.bincount(final_actions, minlength=ACTION_DIM).tolist()

    # Per-step action histogram = sum of action_counts across all steps in all
    # trades. Reflects what the actor was *deciding* every bar — more honest
    # picture of the policy. Should show HOLD as the dominant action (~70%)
    # and PARTIAL_EXIT firing during long winning trades.
    step_total = np.zeros(ACTION_DIM, dtype=np.int64)
    for o in outcomes:
        for i, c in enumerate(o.action_counts):
            step_total[i] += c
    step_total_n = max(1, int(step_total.sum()))
    step_pct = {
        name: round(100.0 * step_total[i] / step_total_n, 2)
        for i, name in enumerate(['HOLD', 'PARTIAL_EXIT', 'RELEASE_RUNNER', 'EXIT_ALL'])
    }

    return {
        "n_trades": int(len(outcomes)),
        "mean_pnl_bps": round(mean_bps, 2),
        "std_pnl_bps": round(std_bps, 2),
        "sharpe": round(sharpe, 3),
        "win_rate": round(win_rate, 3),
        "avg_bars_held": round(avg_held, 2),
        "median_pnl_bps": round(float(np.median(pnls)), 2),
        "p10_pnl_bps": round(float(np.percentile(pnls, 10)), 2),
        "p90_pnl_bps": round(float(np.percentile(pnls, 90)), 2),
        "final_action_dist": dict(zip(
            ['HOLD', 'PARTIAL_EXIT', 'RELEASE_RUNNER', 'EXIT_ALL'], final_dist,
        )),
        "step_action_pct": step_pct,
        "total_steps": int(step_total.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, choices=["BTC", "ETH", "SOL"])
    parser.add_argument("--data-path", default="training/training_data/drl_training")
    parser.add_argument("--models-dir", default="models/exit_drl_v2")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="fraction of bars to use as held-out (last N pct of timeline)")
    parser.add_argument("--output-path", default="data/exit_drl_validation")
    args = parser.parse_args()

    df = pd.read_parquet(Path(args.data_path) / f"{args.asset}_4H_full.parquet")
    n = len(df)
    val_start = int(n * (1.0 - args.val_frac))
    print(f"[{args.asset}] full bars={n}, val window starts at bar {val_start} "
          f"({args.val_frac*100:.0f}% held-out)")

    close = df["close"].to_numpy(dtype=np.float64)
    atr_14 = df["atr_14"].to_numpy(dtype=np.float64)
    atr_pct_series = np.where(close > 0, atr_14 / close, 0.02)

    # Validation entries: every ENTRY_STRIDE bars in the held-out window
    entry_indices = [i for i in range(val_start, n - LOOKAHEAD_BARS, ENTRY_STRIDE)]
    print(f"  {len(entry_indices)} entry points × 2 directions = {len(entry_indices)*2} trades")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.models_dir) / args.asset / "exit_sac_best.pt"
    if not ckpt_path.exists():
        print(f"[FATAL] No checkpoint at {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    drl_actor_fn, val_align = make_drl_actor(
        ckpt_path, device, extra_roots=[str(Path("models").absolute())],
    )
    print(f"  Loaded checkpoint (val_alignment from training={val_align:.3f})")

    baseline_actor_fn = make_baseline_actor(BaselineConfig())

    # Run both actors over identical entry points
    drl_outcomes: List[TradeOutcome] = []
    baseline_outcomes: List[TradeOutcome] = []

    for entry_idx in entry_indices:
        atr_at_entry = float(atr_pct_series[entry_idx])
        if atr_at_entry <= 0:
            atr_at_entry = 0.02
        for direction in (+1, -1):
            d_out = simulate_trade(drl_actor_fn, entry_idx, direction, df, close, atr_at_entry)
            b_out = simulate_trade(baseline_actor_fn, entry_idx, direction, df, close, atr_at_entry)
            if d_out is not None:
                drl_outcomes.append(d_out)
            if b_out is not None:
                baseline_outcomes.append(b_out)

    drl_metrics = aggregate(drl_outcomes)
    base_metrics = aggregate(baseline_outcomes)

    # Per-action lift
    def lift(a, b, key):
        if b == 0 or b is None:
            return None
        return round((a - b) / abs(b) * 100.0, 2)

    summary = {
        "asset": args.asset,
        "val_window": {"start_bar": val_start, "end_bar": n - LOOKAHEAD_BARS,
                       "n_entry_points": len(entry_indices)},
        "drl": drl_metrics,
        "baseline": base_metrics,
        "lift_pct": {
            "mean_pnl_bps": lift(drl_metrics.get("mean_pnl_bps", 0),
                                 base_metrics.get("mean_pnl_bps", 0), "mean_pnl_bps"),
            "sharpe": lift(drl_metrics.get("sharpe", 0),
                           base_metrics.get("sharpe", 0), "sharpe"),
            "win_rate": lift(drl_metrics.get("win_rate", 0),
                             base_metrics.get("win_rate", 0), "win_rate"),
            "avg_bars_held": lift(drl_metrics.get("avg_bars_held", 0),
                                  base_metrics.get("avg_bars_held", 0), "avg_bars_held"),
        },
        "training_val_alignment": float(val_align),
    }

    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.asset}_validation.json"
    out_file.write_text(json.dumps(summary, indent=2))

    # Pretty print
    print(f"\n{'='*70}")
    print(f"[{args.asset}] held-out validation: {len(drl_outcomes)} trades each")
    print(f"{'='*70}")
    print(f"{'metric':<25}{'DRL':>12}{'baseline':>12}{'lift %':>12}")
    print("-" * 70)
    for k in ("n_trades", "mean_pnl_bps", "sharpe", "win_rate", "avg_bars_held"):
        d = drl_metrics.get(k, "—")
        b = base_metrics.get(k, "—")
        l = summary["lift_pct"].get(k, "—")
        l_str = f"{l:+.1f}%" if isinstance(l, (int, float)) else str(l)
        print(f"{k:<25}{str(d):>12}{str(b):>12}{l_str:>12}")
    print(f"  DRL final-action dist:    {drl_metrics.get('final_action_dist', '?')}")
    print(f"  DRL per-step action %:    {drl_metrics.get('step_action_pct', '?')}")
    print(f"  baseline final dist:      {base_metrics.get('final_action_dist', '?')}")
    print(f"  baseline per-step %:      {base_metrics.get('step_action_pct', '?')}")
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
