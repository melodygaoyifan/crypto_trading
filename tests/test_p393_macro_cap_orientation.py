"""[P393] The macro→order channel, re-oriented ADVERSE-RESTRICTS.

The P393 audit measured that the ONLY path by which fetched macro data can
change a sleeve order is `macro_leverage_cap` → fusion LAYER 6 → the
`fusion_conviction` ratio → sleeve contracts, and that the consumer acts
ONLY on `cap < 1.0` (caps above 1.0 have no reader anywhere). Under the
short-bias-era [MACRO-FIX1] map (CRISIS 1.5 / RISK_ON 0.7) that meant:
zero de-risking in CRISIS, and a ~50-58% long-book cut in RISK_ON —
observed live at conviction 0.42–0.52 on RISK_ON days.

Now: ADVERSE regimes carry cap < 1.0 (CRISIS 0.4 / RISK_OFF 0.7 /
DOLLAR_BREAKOUT 0.8) and restrict the LONG side; RISK_ON/NEUTRAL restrict
nothing; shorts get relief (cap×1.5, capped at 1.0 — macro never sizes UP);
sustained ETF OUTFLOWS de-risk (cap ≤ 0.8) instead of "boosting"; and the
HY credit spread (the one factor the P392/P393 labs measured stable in both
eras on all three assets) joins the caution set that classifies the regime.
"""
from __future__ import annotations

import pytest

from data_mgmt.global_context_informer import (
    GlobalContextInformer,
    MacroIndicator,
    MacroRegime,
)
from signals.authority_fusion import (
    AuthorityFusionEngine,
    AgentSignal,
    FusionContext,
    set_drl_authority_level,
)
from core.canonical_enums import SystemMode
from market.phase_detector import RegimePhase


@pytest.fixture(autouse=True)
def _reset_drl():
    set_drl_authority_level("DISABLED")
    yield
    set_drl_authority_level("DISABLED")


def _ctx(**overrides):
    base = dict(mode=SystemMode.NORMAL, regime_phase=RegimePhase.EXPANSION,
                data_valid=True, drl_enabled=False, regime="STEADY_UPTREND")
    base.update(overrides)
    return FusionContext(**base)


def _mi(name, z):
    import time as _t
    return MacroIndicator(ticker=name, name=name, value=4.0, prev_value=3.9,
                          change_pct=1.0, zscore_30d=z, timestamp=_t.time())


def _gci(regime, *, btc_flow_streak=0, hy_z=0.0, us10y_z=0.0):
    g = GlobalContextInformer()
    g._state.macro_regime = regime
    g._state.btc_flow_streak = btc_flow_streak
    if hy_z:
        g._state.hy_oas = _mi("HY_OAS", hy_z)
    if us10y_z:
        g._state.us10y = _mi("US10Y", us10y_z)
    return g


# ---------------------------------------------------------------------------
# The map: adverse -> low, favorable -> 1.0, never above 1.0
# ---------------------------------------------------------------------------
class TestMapOrientation:
    @pytest.mark.parametrize("regime,cap", [
        (MacroRegime.CRISIS, 0.4),
        (MacroRegime.RISK_OFF, 0.7),
        (MacroRegime.DOLLAR_BREAKOUT, 0.8),
        (MacroRegime.NEUTRAL, 1.0),
        (MacroRegime.RISK_ON, 1.0),
    ])
    def test_adverse_restricts_favorable_does_not(self, regime, cap):
        sig = _gci(regime).get_macro_signal()
        assert sig["leverage_cap"] == pytest.approx(cap), (
            "the [MACRO-FIX1] short-bias map (CRISIS 1.5 / RISK_ON 0.7) is "
            "the inversion P393 removed — do not restore it without a P-entry")

    def test_cap_never_exceeds_one(self):
        # caps > 1.0 have NO reader anywhere (P393 audit) — a value above
        # 1.0 is a claim of a boost that cannot and must not exist
        for regime in MacroRegime:
            for streak in (0, -5, 5):
                sig = _gci(regime, btc_flow_streak=streak).get_macro_signal()
                assert sig["leverage_cap"] <= 1.0, (regime, streak)

    def test_sustained_etf_outflows_derisk_instead_of_boosting(self):
        base = _gci(MacroRegime.NEUTRAL).get_macro_signal()["leverage_cap"]
        out = _gci(MacroRegime.NEUTRAL, btc_flow_streak=-4).get_macro_signal()
        assert base == pytest.approx(1.0)
        assert out["leverage_cap"] <= 0.8, (
            "outflow streaks used to RAISE the cap toward 1.5 (short-bias "
            "era); adverse flow must de-risk a long book")
        # the streak magnitude is still exported for observability
        assert out["etf_short_boost"] > 0

    def test_crisis_beats_the_outflow_floor(self):
        sig = _gci(MacroRegime.CRISIS, btc_flow_streak=-4).get_macro_signal()
        assert sig["leverage_cap"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# HY credit spread joins the caution set
# ---------------------------------------------------------------------------
class TestHyCreditSpreadCaution:
    def test_hy_stress_is_a_caution_input(self):
        g = _gci(MacroRegime.NEUTRAL, hy_z=2.0)
        g._update_regime_signals()
        assert g._state.hy_stress_caution is True
        # one caution -> RISK_OFF classification
        assert g._state.macro_regime == MacroRegime.RISK_OFF

    def test_two_cautions_classify_crisis(self):
        g = _gci(MacroRegime.NEUTRAL, hy_z=2.0, us10y_z=2.0)
        g._update_regime_signals()
        assert g._state.macro_regime == MacroRegime.CRISIS

    def test_absent_series_is_calm_never_fabricated_stress(self):
        g = _gci(MacroRegime.NEUTRAL)          # no hy_oas at all
        g._update_regime_signals()
        assert g._state.hy_stress_caution is False

    def test_hy_oas_is_in_the_fred_map(self):
        from data_mgmt.feeds.fred_macro_series import FRED_SERIES_MAP
        assert FRED_SERIES_MAP.get("HY_OAS") == "BAMLH0A0HYM2"


# ---------------------------------------------------------------------------
# LAYER 6 consumer: longs take the map's restriction, shorts get relief,
# cap >= 1.0 touches nothing
# ---------------------------------------------------------------------------
class TestLayer6Consumer:
    def _fuse(self, engine, direction, cap):
        signals = {
            "quant": AgentSignal(direction=direction, confidence=0.9),
            # the VETO agent must be PRESENT and quiet, else fuse()
            # fail-closes before LAYER 6 ever runs
            "risk": AgentSignal(direction=0.0, confidence=0.0,
                                veto_active=False),
            "macro": AgentSignal(direction=0.0, confidence=0.5,
                                 leverage_cap=cap),
        }
        return engine.fuse(signals, _ctx())

    def test_adverse_cap_binds_longs_at_the_map_value(self):
        engine = AuthorityFusionEngine()
        r = self._fuse(engine, +0.9, 0.4)
        assert r.target_exposure <= 0.4 + 1e-9
        assert r.caps_applied.get("macro") == pytest.approx(0.4), (
            "the long branch must take the map's restriction AS-IS — the "
            "old x0.6 extra penalty belonged to the inverted orientation")
        # and it is inside the conviction ratio the sleeve consumes
        assert r.fusion_conviction < 1.0

    def test_shorts_get_relief_not_full_restriction(self):
        engine = AuthorityFusionEngine()
        r = self._fuse(engine, -0.9, 0.4)
        assert r.caps_applied.get("macro") == pytest.approx(0.6)  # 0.4*1.5

    def test_short_relief_never_exceeds_one(self):
        engine = AuthorityFusionEngine()
        r = self._fuse(engine, -0.9, 0.9)
        assert r.caps_applied.get("macro") == pytest.approx(1.0)  # capped

    def test_cap_at_one_touches_nothing(self):
        engine = AuthorityFusionEngine()
        r = self._fuse(engine, +0.9, 1.0)
        assert "macro" not in r.caps_applied
        r2 = self._fuse(AuthorityFusionEngine(), +0.9, 1.5)
        assert "macro" not in r2.caps_applied
