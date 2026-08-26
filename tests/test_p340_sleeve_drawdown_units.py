"""[P340] The sleeve halt and the certified drawdowns are in DIFFERENT UNITS.

P325 recorded a gap: "halts tighter than the strategy's certified drawdown --
SOL trend-only historical maxDD -199 pts at 1.0 exposure -> ~ -26% realized at
the live 0.15 fraction, against a 15% sleeve halt". Acting on it would have
meant halving live sizing.

It does not survive measurement, because the two numbers are not comparable:

  * the certified figures are PEAK-TO-TROUGH drawdowns of an additive per-bar
    return sum at 1.0 exposure (P301 states the additive convention);
  * `CoinbaseSleeve` measures `dd = (basis - equity) / basis` where
    `basis = sleeve_start_equity + external_flow_usd` -- an INCEPTION anchor
    with NO ratchet (exchange/coinbase_sleeve.py, and it reproduces the live
    `[COINBASE-PNL] dd=2.3%`).

A peak-anchored drawdown asks "how much of the gains were given back". An
inception-anchored one asks "are you below the money you put in". For a
profitable strategy those diverge without limit: measured over 2020-2026 at
the live fractions, the combined book's peak-to-trough drawdown is -28.1%
while its worst inception-anchored drawdown is 2.5% and the 15% halt would
have fired ZERO times.

The residual risk is real but much smaller, and it is a COLD-START risk --
the halt is only forgiving once a profit buffer exists, and the live sleeve
has none (it sits 2.34% BELOW basis). Sampling every possible start date:

    fraction  median worst-DD yr1   p90     max    cold starts tripping 15%
      0.15           6.3%          15.9%   22.9%            12%
      0.10           4.1%          10.7%   15.7%             0.3%

These tests pin the UNIT distinction, because that is what the error was --
not an arithmetic slip but two quantities sharing the word "drawdown".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

pd = pytest.importorskip("pandas")

from training.scripts.sleeve_drawdown_probe import (  # noqa: E402
    equity_curve, max_drawdown, trend_only_position,
    HALF_SPREAD_BPS, PER_CONTRACT_FEE_BPS, LIVE_FRACTION,
)


def _rising_then_dipping():
    """A curve that gains a lot, then gives back a third of it.

    Peak-anchored: a large drawdown. Inception-anchored: still far above the
    basis, so no drawdown at all. The whole confusion in one series.
    """
    import numpy as np
    up = np.linspace(1.0, 3.0, 100)
    down = np.linspace(3.0, 2.0, 50)
    return pd.Series(np.concatenate([up, down]))


class TestTheTwoDrawdownsAreDifferentQuantities:

    def test_peak_anchored_sees_a_big_drawdown(self):
        c = _rising_then_dipping()
        assert max_drawdown(c) < -0.30

    def test_inception_anchored_sees_none(self):
        """This is the sleeve's actual formula: (basis - equity)/basis with
        basis fixed at inception. Equity is 2x the basis, so dd is negative
        (i.e. a gain) and no halt can fire."""
        c = _rising_then_dipping()
        dd = 1.0 - c            # basis normalised to 1.0
        assert dd.max() <= 0.0, "a book above its basis has no inception drawdown"

    def test_they_disagree_by_construction_not_by_arithmetic_error(self):
        c = _rising_then_dipping()
        peak_dd = abs(max_drawdown(c))
        incep_dd = max(0.0, float((1.0 - c).max()))
        assert peak_dd > 0.30 and incep_dd == 0.0, (
            "P325 compared these two numbers directly; they are not the same "
            "quantity and the comparison cannot be repaired by rescaling")


class TestTheSleeveFormulaIsInceptionAnchored:
    """If the sleeve ever switches to a trailing peak, P340's conclusion is
    void and the halt becomes far more binding — so pin the formula."""

    def test_dd_is_measured_against_start_plus_flows(self):
        from tests._source_scan import code_only
        src = code_only(REPO / "exchange" / "coinbase_sleeve.py",
                        strip_docstrings=True)
        i = src.index("_basis = ")
        blk = src[i:i + 400]
        assert "_sleeve_start_equity" in blk and "_flows" in blk
        assert "(_basis - eq) / _basis" in blk, (
            "the halt's denominator is the inception basis; a cummax/peak "
            "here would make the certified peak-to-trough figures binding")

    def test_no_ratchet_appears_in_the_basis(self):
        from tests._source_scan import code_only
        src = code_only(REPO / "exchange" / "coinbase_sleeve.py",
                        strip_docstrings=True)
        i = src.index("_basis = ")
        blk = src[i:i + 400]
        for ratchet in ("cummax", "max(self._peak", "_peak_equity"):
            assert ratchet not in blk, f"{ratchet} would change the units"


class TestTheProbeItself:

    def test_position_rule_is_the_certified_one(self):
        """long above the 200-bar SMA, else flat — and shifted, so the
        decision never uses the bar it acts in (P164)."""
        import numpy as np
        close = pd.Series(np.arange(1, 400, dtype=float))   # monotone up
        pos = trend_only_position(close)
        assert pos.iloc[:200].sum() == 0.0, "no position before the SMA warms"
        assert pos.iloc[-1] == 1.0, "a rising series must end long"

    def test_the_decision_cannot_see_its_own_bar(self):
        """Perturb the LAST bar violently; every earlier position must be
        bit-identical (the P164 construction test)."""
        import numpy as np
        close = pd.Series(np.linspace(100.0, 200.0, 400))
        base = trend_only_position(close)
        poisoned = close.copy()
        poisoned.iloc[-1] = 1e9
        after = trend_only_position(poisoned)
        assert base.iloc[:-1].equals(after.iloc[:-1])

    def test_costs_only_ever_reduce_the_curve(self):
        """Fees and carry must make the drawdown WORSE, never better — an
        under-costed backtest understates the drawdown a halt must tolerate."""
        import numpy as np
        close = pd.Series(np.linspace(100.0, 160.0, 500))
        pos = trend_only_position(close)
        free = equity_curve(close, pos, "ETH", 0.15, carry_bps_per_bar=0.0)
        paid = equity_curve(close, pos, "ETH", 0.15, carry_bps_per_bar=2.0)
        assert paid.iloc[-1] < free.iloc[-1]

    def test_live_fractions_match_the_live_profile(self):
        """The probe's answer is only about the live book if its fractions are
        the live ones. [P412] The probe is HOME-SCOPED (it simulates BTC/ETH/SOL
        drawdown on 2020-2026 data); XRP breadth (1ct/~$250, activated one-first
        P197) is outside its scope, so the pin checks the home subset matches
        and that XRP is the sole breadth addition."""
        import json
        cfg = json.loads((REPO / "configs" / "live_high_risk.json").read_text(
            encoding="utf-8-sig"))
        frac = cfg.get("coinbase_target_fraction_by_asset")
        assert {a: frac[a] for a in LIVE_FRACTION} == LIVE_FRACTION
        assert {a: v for a, v in frac.items() if a not in LIVE_FRACTION} == {
            "XRP": 0.01, "BNB": 0.005}

    def test_cost_constants_are_per_leg_and_nonzero(self):
        for a in ("BTC", "ETH", "SOL"):
            assert HALF_SPREAD_BPS[a] > 0 and PER_CONTRACT_FEE_BPS[a] > 0


class TestTheHaltValueIsDeliberatelyLeftAbsent:
    """[P340] I measured 0.15 as the right value and then did NOT write it in.

    P239 pins both sleeve knobs ABSENT from the live profile: "absent = ctor
    defaults = today's behavior. Setting them is an operator risk decision,
    not a side effect of this wiring." My pin would have been byte-neutral
    (0.15 IS the default), but the guard is about WHO DECIDES, not about the
    value — and weakening a guard to admit your own change is never the fix
    (P248). The measurement lives in P340; the config stays untouched until
    an operator says so.
    """

    def _cfg(self):
        import json
        return json.loads((REPO / "configs" / "live_high_risk.json").read_text(
            encoding="utf-8-sig"))

    def test_the_key_is_still_absent_and_p239_still_owns_it(self):
        # [P370] The decision this pin was guarding has now been MADE, by
        # explicit operator instruction on the P369 six-year backtest: at 15%
        # the halt trips SOL 48-77x in 6y and removes 60-85% of its return;
        # at 25% it is a tolerable premium on all three and equals the
        # existing risk.hard_drawdown_halt. So the pin moves from must-be-
        # ABSENT to the DECIDED value (P237/P270 pattern), exactly as the
        # sibling P239 pin did in the same commit. A silent revert to 0.15
        # and a silent loosening past 0.25 both fail here.
        assert self._cfg().get("coinbase_max_sleeve_drawdown_pct") == 0.25, (
            "P370 decided 0.25; any other value is a new operator decision "
            "needing its own P-entry")

    def test_the_effective_value_is_the_one_that_was_measured(self):
        """The absence is only safe while the default equals what P340
        measured as acceptable — if the ctor default ever moves, the recorded
        12%-cold-start figure no longer describes the live halt."""
        import inspect
        from exchange.coinbase_sleeve import CoinbaseSleeve
        d = inspect.signature(CoinbaseSleeve.__init__).parameters[
            "max_sleeve_drawdown_pct"].default
        assert d == 0.15, (
            "P340's cold-start measurement (12% of cold starts trip the halt) "
            "was computed at 0.15; a different default needs it re-derived")
