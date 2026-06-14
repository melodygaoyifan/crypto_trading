"""
Vol-targeted time-series trend-following (the root-cause-fix strategy).

WHY (session 2026-06-13/14, memory session-endtoend-fix-2026-06-13): the live
system is a directional-ensemble-timing engine ("predict each asset's 4H
direction") — a strategy CLASS with no edge (live Sharpe -2.62, PSR 21%, alpha
-62%/yr). Deep research found time-series trend-following is the ONLY directional
approach with surviving out-of-sample evidence. Backtest on 5.3y of BTC/ETH/SOL
4H bars (textbook params, NOT tuned, 15bps cost, no lookahead): per-asset Sharpe
0.26/0.33/0.73, equal-risk 3-asset PORTFOLIO Sharpe 0.53, PSR 89%, maxDD -13.6%.
That is modest but POSITIVE and regime-robust (it goes SHORT in downtrends — the
regime that crushed the current net-long engine).

CONSTRUCTION (standard managed-futures):
  - signal = mean over lookbacks of sign(momentum) in [-1, +1].
  - vol-targeting: position = signal * (target_vol / realized_annualized_vol),
    clamped to +-max_pos. Equalizes risk across assets and across time.
  - NO lookahead: every value at bar t uses only close[:t+1].

This module is a PURE signal generator — it decides nothing live. Wiring it as a
SHADOW strategy (log what it would do vs the live engine) then promoting on
FORWARD evidence is a separate, gated step. Backtest params here MATCH the
validated run exactly (C:/tmp/backtest_trend.py) so live == backtest.
"""

from __future__ import annotations

import math
from typing import List, Optional, Dict, Any, Sequence


class TrendFollowingStrategy:
    def __init__(
        self,
        lookbacks_days: Sequence[int] = (10, 20, 40),
        vol_window_days: int = 30,
        target_vol: float = 0.15,
        max_pos: float = 1.0,
        bars_per_day: int = 6,  # 4H bars
    ) -> None:
        self.bpd = int(bars_per_day)
        self.lookbacks = [int(d) * self.bpd for d in lookbacks_days]
        self.vol_window = int(vol_window_days) * self.bpd
        self.target_vol = float(target_vol)
        self.max_pos = float(max_pos)
        self.year_bars = 365 * self.bpd

    def min_history(self) -> int:
        return max(self.lookbacks) + self.vol_window + 1

    def compute(self, close_history: Sequence[float]) -> Optional[Dict[str, Any]]:
        """Return {signal, ann_vol, target_position} from a close-price history,
        or None if there isn't enough history (fail-open). target_position in
        [-max_pos, +max_pos]: positive=long, negative=short."""
        c = close_history
        n = len(c)
        if n < self.min_history():
            return None
        t = n - 1  # decide as of the latest closed bar
        # multi-lookback momentum sign
        sig = 0.0
        for L in self.lookbacks:
            base = c[t - L]
            if base <= 0:
                continue
            mom = (c[t] - base) / base
            sig += 1.0 if mom > 0 else -1.0
        sig /= len(self.lookbacks)
        # realized annualized vol from the trailing window of 4H returns
        rets = []
        for i in range(t - self.vol_window + 1, t + 1):
            if c[i - 1] > 0:
                rets.append((c[i] - c[i - 1]) / c[i - 1])
        if not rets:
            return None
        sd = math.sqrt(sum(x * x for x in rets) / len(rets))
        ann_vol = sd * math.sqrt(self.year_bars)
        scale = (self.target_vol / ann_vol) if ann_vol > 1e-9 else 0.0
        target = max(-self.max_pos, min(self.max_pos, sig * scale))
        return {"signal": sig, "ann_vol": ann_vol, "target_position": target}
