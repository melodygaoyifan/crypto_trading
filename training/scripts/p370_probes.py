"""[P370] Falsification probes — reintroduce each defect, require red."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.falsify import Probe, run_probes  # noqa: E402

T = "tests/test_p370_risk_controls_armed.py"
F = "execution/fast_risk_tick.py"
M = "main.py"
C = "configs/live_high_risk.json"

PROBES = [
    Probe(
        name="a retired price trigger fires anyway (the enabled flag is ignored)",
        path=F,
        old="        _price_move_triggered = (self.price_trigger_enabled\n                                 and price_move_pct > self.price_move_threshold)",
        new="        _price_move_triggered = price_move_pct > self.price_move_threshold",
        expect_red=[T],
    ),
    Probe(
        name="the disable range is wrong: 0.0 reads as a real threshold",
        path=F,
        old="        self.price_trigger_enabled = 0.0 < self.price_move_threshold < 1.0",
        new="        self.price_trigger_enabled = self.price_move_threshold < 1.0",
        expect_red=[T],
    ),
    Probe(
        name="the vol trigger ignores its per-instance multiplier (uses the class const)",
        path=F,
        old="                and current_vol > baseline_vol * self.vol_spike_mult):",
        new="                and current_vol > baseline_vol * self.VOLATILITY_SPIKE_MULT):",
        expect_red=[T],
    ),
    Probe(
        name="a retired vol trigger (mult<=0) fires anyway",
        path=F,
        old="        if (self.vol_trigger_enabled and baseline_vol > 0",
        new="        if (baseline_vol > 0",
        expect_red=[T],
    ),
    Probe(
        name="main.py stops threading the knobs into the ctor (config cannot take effect)",
        path=M,
        old="                    price_move_threshold=getattr(\n                        self.config, 'fast_risk_price_move_threshold', None),\n                    vol_spike_mult=getattr(\n                        self.config, 'fast_risk_vol_spike_mult', None))",
        new="                    )",
        expect_red=[T],
    ),
    Probe(
        name="the live profile silently reverts the halt to 15%",
        path=C,
        old='  "coinbase_max_sleeve_drawdown_pct": 0.25,',
        new='  "coinbase_max_sleeve_drawdown_pct": 0.15,',
        expect_red=[T, "tests/test_hygiene_p239.py"],
    ),
    Probe(
        name="the live profile silently re-arms the 3% price trigger",
        path=C,
        old='  "fast_risk_price_move_threshold": 0.0,',
        new='  "fast_risk_price_move_threshold": 0.03,',
        expect_red=[T],
    ),
    # ---- vol-parity fractions (same commit) ----
    Probe(
        name="the live profile silently reverts SOL to the flat 0.15 fraction",
        path=C,
        old='"coinbase_target_fraction_by_asset": {"BTC": 0.20, "ETH": 0.15, "SOL": 0.095}',
        new='"coinbase_target_fraction_by_asset": {"BTC": 0.20, "ETH": 0.15, "SOL": 0.15}',
        expect_red=["tests/test_p370_vol_parity_fractions.py"],
    ),
    Probe(
        name="parity is turned into a real loosening (ETH raised past 0.15)",
        path=C,
        old='"coinbase_target_fraction_by_asset": {"BTC": 0.20, "ETH": 0.15, "SOL": 0.095}',
        new='"coinbase_target_fraction_by_asset": {"BTC": 0.20, "ETH": 0.22, "SOL": 0.095}',
        expect_red=["tests/test_p370_vol_parity_fractions.py"],
    ),
]

if __name__ == "__main__":
    sys.exit(0 if run_probes(PROBES) else 1)
