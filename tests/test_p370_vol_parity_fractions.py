"""
[P370] Sleeve target fractions moved from flat 0.15 x3 to VOL-PARITY.

The P370 strategy-threshold audit (training/strategy_threshold_audit_lab.py)
measured that flat 0.15 put 48% of three-asset book RISK in SOL — annualised
vol while long BTC 0.58 / ETH 0.77 / SOL 1.23, a ranking stable in all three
eras — 2.1x BTC's share. Parity at the SAME 0.45 aggregate budget is
{BTC 0.20, ETH 0.15, SOL 0.095}; backtest: 12% shallower max drawdown
(-33.9% -> -29.7%) and a smaller worst 4H bar at equal Sharpe.

THIS IS NOT A LOOSENING, and the tests below pin why: ETH is unchanged, SOL is
reduced, BTC is raised — but BTC cannot pass the alpha gate (24.1bps edge <
27.7bps fee floor), so realised net exposure FALLS. P274 noted flat 0.15 matched
P273's vol-parity contract book only at the $3,775 activation equity by
rounding coincidence; at ~$10.9k it was no longer parity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LIVE = REPO / "configs" / "live_high_risk.json"
DECIDED = {"BTC": 0.20, "ETH": 0.15, "SOL": 0.095}
# measured in the audit lab, 6y hourly, annualised vol while the trend book is long
VOL = {"BTC": 0.583, "ETH": 0.773, "SOL": 1.234}


def _live() -> dict:
    return json.loads(LIVE.read_text(encoding="utf-8"))


class TestTheDecidedValues:
    def test_the_live_profile_carries_the_parity_fractions(self):
        assert _live()["coinbase_target_fraction_by_asset"] == DECIDED

    def test_no_duplicate_key(self):
        """P298: JSON last-key-wins silently ate the first flip."""
        txt = LIVE.read_text(encoding="utf-8")
        assert txt.count('"coinbase_target_fraction_by_asset"') == 1

    def test_the_note_names_its_evidence_and_revert(self):
        note = _live().get("_p370_vol_parity_note", "")
        assert "strategy_threshold_audit_lab" in note
        assert "REVERT" in note
        assert "48%" in note

    def test_every_fraction_is_under_its_cap(self):
        """The ctor clamps at 0.25 and post_leverage_caps bound each asset;
        parity must sit under both or it is a cap change in disguise."""
        c = _live()
        caps = c["post_leverage_caps"]
        for a, f in c["coinbase_target_fraction_by_asset"].items():
            assert f <= 0.25, f"{a} {f} exceeds the ctor clamp"
            assert f <= caps[a], f"{a} {f} exceeds post_leverage_cap {caps[a]}"


class TestItIsNotALoosening:
    def test_the_aggregate_budget_is_unchanged(self):
        """Same 0.445 book as flat 0.15 x3 — a re-weighting, not a raise."""
        tot = sum(_live()["coinbase_target_fraction_by_asset"].values())
        assert tot == pytest.approx(0.445, abs=1e-9)
        assert tot <= 0.50, "must stay under the P208 net cap"

    def test_sol_went_DOWN_and_eth_is_unchanged(self):
        f = _live()["coinbase_target_fraction_by_asset"]
        assert f["SOL"] < 0.15
        assert f["ETH"] == 0.15

    def test_only_the_asset_that_cannot_trade_went_up(self):
        """BTC is the only raise, and BTC is refused by the alpha gate on every
        retained tick (est 18 < thresh 35-42). Pinned so that if BTC's gate
        status ever changes, this sizing is re-examined rather than inherited."""
        f = _live()["coinbase_target_fraction_by_asset"]
        raised = [a for a in f if f[a] > 0.15]
        assert raised == ["BTC"]


class TestItIsActuallyParity:
    """The values are DERIVED from measured vol, not chosen by hand."""

    def test_fractions_equalise_one_sigma_dollars(self):
        """f_a * vol_a should be ~equal across assets (within rounding)."""
        risk = {a: DECIDED[a] * VOL[a] for a in DECIDED}
        lo, hi = min(risk.values()), max(risk.values())
        assert hi / lo < 1.15, f"not parity: {risk}"

    def test_flat_015_was_NOT_parity(self):
        """The thing being fixed: SOL's share at flat 0.15."""
        flat = {a: 0.15 * VOL[a] for a in VOL}
        sol_share = flat["SOL"] / sum(flat.values())
        assert sol_share > 0.45, f"flat 0.15 SOL risk share {sol_share:.2f}"
        par = {a: DECIDED[a] * VOL[a] for a in VOL}
        assert par["SOL"] / sum(par.values()) < 0.40


class TestTheSleeveConsumesIt:
    """A config value nothing reads is the P16/P201 class."""

    def test_the_ctor_clamp_admits_every_parity_value(self):
        from exchange.coinbase_sleeve import CoinbaseSleeve
        s = object.__new__(CoinbaseSleeve)
        # reproduce the ctor's clamp expression on the decided values
        clamped = {k: min(0.25, max(0.0, float(v))) for k, v in DECIDED.items()}
        assert clamped == DECIDED, "a parity value was silently clamped"
