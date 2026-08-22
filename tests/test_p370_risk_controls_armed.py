"""
[P370] Three live risk controls re-set on the P369 six-year backtest, by
explicit operator instruction ("don't be over conservative, we can afford
to lose ... retire trigger A, retire or loosen B to 4x, raise the sleeve halt
15% -> 25%").

  A  FastRiskTick price-move EXIT_ONLY (3% drift from a resetting 4H anchor)
     -> RETIRED via fast_risk_price_move_threshold = 0.0.
     Backtest: cost 10-93%/yr of notional, no tail protection, era-unstable,
     94th-99th-percentile trigger, reads no position state. Live: 33 flattens
     of profitable longs in ~2 days, ~$195 churn vs -$118.64 PnL.
  B  FastRiskTick vol-spike REDUCE_50 (2x) -> LOOSENED to 4x.
     Backtest: 166-194 fires/yr at 2x, 29-40%/yr tax, ~zero tail effect.
  D  Sleeve drawdown halt 15% -> 25%.
     Backtest: at 15% it trips SOL 48-77x in 6y and removes 60-85% of its
     return; at 25% it is a tolerable premium on all three.

The 10% venue-resting protective stop (P197) is UNTOUCHED and is the control
that actually protects: anchored to ENTRY, survives process death.

DESIGN. The triggers are retired by a threshold nothing can reach, not by
deleting code: P367's shadow counters keep measuring both quantities, so the
evidence keeps accruing and a future re-arming is one config key. Each new
knob is Optional; None means the class constant and is byte-identical to the
pre-P370 behaviour, which the default-path tests below pin.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from execution.fast_risk_tick import FastRiskTick, FastRiskAction  # noqa: E402

LIVE = REPO / "configs" / "live_high_risk.json"


def _md(price: float, vol: float = 0.0, depth: float = 1_000_000.0) -> dict:
    return {"current_price": price, "volatility_30m": vol,
            "orderbook_depth_1pct_usd": depth, "data_valid": True}


def _armed(**kw) -> FastRiskTick:
    t = FastRiskTick(shadow_mode=False, **kw)
    t.set_4h_anchor("ETH", price=100.0, volatility=0.01, depth=1_000_000.0)
    return t


# --------------------------------------------------------------------------
# Default path: None -> class constants, byte-identical to before P370
# --------------------------------------------------------------------------
class TestTheDefaultIsUnchanged:
    def test_none_means_the_class_constants(self):
        t = FastRiskTick(shadow_mode=False)
        assert t.price_move_threshold == pytest.approx(FastRiskTick.PRICE_MOVE_THRESHOLD)
        assert t.vol_spike_mult == pytest.approx(FastRiskTick.VOLATILITY_SPIKE_MULT)
        assert t.price_trigger_enabled is True
        assert t.vol_trigger_enabled is True

    def test_default_still_fires_EXIT_ONLY_on_a_4pct_drift(self):
        """The pre-P370 behaviour, pinned so the default path cannot drift."""
        t = _armed()
        res = t.evaluate("ETH", _md(104.0))
        assert res.action == FastRiskAction.EXIT_ONLY

    def test_default_still_fires_REDUCE_50_on_a_3x_vol_spike(self):
        t = _armed()
        res = t.evaluate("ETH", _md(100.0, vol=0.03))   # 3x the 0.01 baseline
        assert res.action == FastRiskAction.REDUCE_50


# --------------------------------------------------------------------------
# A: RETIRED price trigger
# --------------------------------------------------------------------------
class TestTheRetiredPriceTrigger:
    @pytest.mark.parametrize("thr", [0.0, -1.0, 1.0, 5.0])
    def test_a_disable_value_means_it_can_never_fire(self, thr):
        t = _armed(price_move_threshold=thr)
        assert t.price_trigger_enabled is False
        # a move that would have fired at the old 3% — and a huge one
        for px in (104.0, 96.0, 150.0, 50.0):
            res = t.evaluate("ETH", _md(px))
            assert res.action != FastRiskAction.EXIT_ONLY, (
                f"retired price trigger fired EXIT_ONLY at px={px}")

    def test_retiring_A_does_not_retire_B(self):
        """Independent knobs: the vol REDUCE_50 must still work with A off."""
        t = _armed(price_move_threshold=0.0)
        res = t.evaluate("ETH", _md(100.0, vol=0.03))
        assert res.action == FastRiskAction.REDUCE_50

    def test_the_shadow_counters_still_accrue_when_retired(self):
        """Retire the ACTION, keep the MEASUREMENT (P367) — the evidence for
        any future re-arming must keep accumulating."""
        t = _armed(price_move_threshold=0.0)
        t.evaluate("ETH", _md(104.0))
        assert t._shadow_evals.get("ETH", 0) >= 1
        assert t._shadow_drift_fires.get("ETH", 0) >= 1

    def test_a_positive_in_range_value_still_fires(self):
        """A non-disable value is a real threshold, not a second off-switch."""
        t = _armed(price_move_threshold=0.07)
        assert t.price_trigger_enabled is True
        assert t.evaluate("ETH", _md(104.0)).action != FastRiskAction.EXIT_ONLY
        assert t.evaluate("ETH", _md(108.0)).action == FastRiskAction.EXIT_ONLY


# --------------------------------------------------------------------------
# B: vol trigger loosened to 4x
# --------------------------------------------------------------------------
class TestTheLoosenedVolTrigger:
    def test_at_4x_a_3x_spike_no_longer_reduces(self):
        t = _armed(vol_spike_mult=4.0)
        assert t.evaluate("ETH", _md(100.0, vol=0.03)).action == FastRiskAction.HOLD

    def test_at_4x_a_5x_spike_still_reduces(self):
        """Loosened, not retired: the control still exists above 4x."""
        t = _armed(vol_spike_mult=4.0)
        assert t.evaluate("ETH", _md(100.0, vol=0.05)).action == FastRiskAction.REDUCE_50

    @pytest.mark.parametrize("m", [0.0, -2.0])
    def test_zero_or_negative_retires_it(self, m):
        t = _armed(vol_spike_mult=m)
        assert t.vol_trigger_enabled is False
        assert t.evaluate("ETH", _md(100.0, vol=0.50)).action == FastRiskAction.HOLD


# --------------------------------------------------------------------------
# The config trio (P201): declared, parsed, consumed — and the live values
# --------------------------------------------------------------------------
class TestTheConfigTrio:
    def test_declared_on_ProductionConfig(self):
        src = (REPO / "main.py").read_text(encoding="utf-8-sig", errors="replace")
        assert re.search(r"^\s+fast_risk_price_move_threshold: Optional\[float\] = None", src, re.M)
        assert re.search(r"^\s+fast_risk_vol_spike_mult: Optional\[float\] = None", src, re.M)

    def test_parsed_in_from_file(self):
        src = (REPO / "main.py").read_text(encoding="utf-8-sig", errors="replace")
        assert 'data.get("fast_risk_price_move_threshold")' in src
        assert 'data.get("fast_risk_vol_spike_mult")' in src

    def test_consumed_at_the_FastRiskTick_ctor(self):
        src = (REPO / "main.py").read_text(encoding="utf-8-sig", errors="replace")
        i = src.find("self.fast_risk_tick = FastRiskTick(")
        blk = src[i:i + 900]
        assert "fast_risk_price_move_threshold" in blk
        assert "fast_risk_vol_spike_mult" in blk

    def test_the_live_profile_carries_the_DECIDED_values(self):
        """Pinned to the decided values (P237 pattern): a silent revert AND a
        silent further loosening both fail, because either is a live-money
        change that should be argued for rather than drifted into."""
        c = json.loads(LIVE.read_text(encoding="utf-8"))
        assert c["fast_risk_price_move_threshold"] == 0.03, "P380: velocity armed @3% (drift retired via velocity_trigger selection)"
        assert c["fast_risk_velocity_trigger"] is True, "P380: velocity trigger armed"
        assert c["fast_risk_vol_spike_mult"] == 4.0, "B must stay at 4x"
        assert c["coinbase_max_sleeve_drawdown_pct"] == 0.25, "halt must stay at 25%"

    def test_the_live_profile_has_no_duplicate_keys(self):
        """P298: JSON last-key-wins silently ate the first flip. A duplicate
        here would make the config read one value and the diff show another."""
        txt = LIVE.read_text(encoding="utf-8")
        for k in ("fast_risk_price_move_threshold", "fast_risk_vol_spike_mult",
                  "coinbase_max_sleeve_drawdown_pct"):
            assert txt.count(f'"{k}"') == 1, f"{k} appears {txt.count(chr(34)+k+chr(34))}x"

    def test_the_venue_stop_is_UNTOUCHED(self):
        """The 10% venue-resting stop is the control that actually protects;
        this change must not have moved it."""
        c = json.loads(LIVE.read_text(encoding="utf-8"))
        assert c["coinbase_protective_stop_pct"] == pytest.approx(0.10)

    def test_the_arming_note_names_its_evidence_and_revert(self):
        c = json.loads(LIVE.read_text(encoding="utf-8"))
        note = c.get("_p370_risk_control_note", "")
        assert "P369" in note and "REVERT" in note and "risk_control_audit_lab" in note


class TestTheLiveParserRoundTrip:
    """Drive the REAL from_file, not a re-statement of it (P234)."""

    def test_absent_keys_parse_to_None(self, tmp_path):
        from main import ProductionConfig
        base = json.loads(LIVE.read_text(encoding="utf-8"))
        for k in ("fast_risk_price_move_threshold", "fast_risk_vol_spike_mult"):
            base.pop(k, None)
        p = tmp_path / "c.json"; p.write_text(json.dumps(base), encoding="utf-8")
        cfg = ProductionConfig.from_file(p)   # from_file takes a Path, not str
        assert cfg.fast_risk_price_move_threshold is None
        assert cfg.fast_risk_vol_spike_mult is None

    def test_the_live_profile_parses_to_the_decided_values(self):
        from main import ProductionConfig
        cfg = ProductionConfig.from_file(LIVE)
        assert cfg.fast_risk_price_move_threshold == 0.03  # P380 velocity armed
        assert cfg.fast_risk_vol_spike_mult == 4.0
        assert cfg.coinbase_max_sleeve_drawdown_pct == 0.25
