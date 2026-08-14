"""[P265] The halt/equity-denomination cluster.

Live incident 2026-08-13: a Coinbase 502 window produced
`RISK_STATE_CHANGE: REDUCED -> HALTED (dd=34.76%)` — the P261 phantom number to
the basis point ($10,865.13 combined anchor vs $7,088.69 Kraken-only). Four
gaps let a partial venue outage read as a real drawdown:

  1. `_update_drawdown_snapshot` had NO guard for an unconstructed sleeve, and
     its built-but-unpriced guard required a PRIOR drawdown — a fresh process
     with the venue down measured the Kraken-only book against the restored
     COMBINED peak.
  2. `CoinbaseSleeve.update_risk` computed dd from eq=0.0 against the restored
     baseline: instant "100% drawdown", sticky, persisted. (P263 made the
     first update_risk run at startup construction — exactly when a boot-time
     outage yields 0.)
  3. `sleeve_equity_usd`'s fallback RETURNED the FCM-only futures-summary
     subset (~$439 vs the ~$4,000 portfolio total, P153) with no degraded
     marker — a wrong-DENOMINATION number fed to the halt, the existence fuse
     and the P0 fold.
  4. `_portfolio_uuid` cached an empty string forever on one malformed
     response, permanently disabling the primary equity path.

Plus: the LIVE drawdown halt set `_running = False` without `continue`, so the
halt tick still ran the full decide loop AND the sleeve driver; and FORCE_FLAT
reported "already flat" when the venue was merely unreadable, and skipped an
unconstructed sleeve in silence.
"""

import asyncio
import re
import types
from pathlib import Path

import pytest

from exchange.coinbase_sleeve import CoinbaseSleeve

REPO = Path(__file__).resolve().parent.parent
MAIN_SRC = (REPO / "main.py").read_text(encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bare_sleeve(**attrs):
    """CoinbaseSleeve without __init__ (the P195 test pattern) — set only the
    fields the method under test actually reads."""
    s = object.__new__(CoinbaseSleeve)
    s._halted = False
    s._halt_reason = ""
    s._sleeve_start_equity = None
    s._max_sleeve_drawdown_pct = 0.15
    s._last_dd_pct = 0.0
    s._last_equity_usd = 0.0
    s._last_positions = {}
    s._cb_portfolio_uuid = None
    s._persist_state = lambda: None  # type: ignore[assignment]
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


class _RaisingClient:
    """Portfolio endpoint down, futures endpoint up — the exact partial-outage
    shape of the incident."""

    def __init__(self, subset_total="439.0", upnl="12.0"):
        self._subset_total = subset_total
        self._upnl = upnl

    def get_portfolios(self):
        return {"portfolios": [{"type": "DEFAULT", "uuid": "abc-123"}]}

    def get_portfolio_breakdown(self, portfolio_uuid=None):
        raise ConnectionError("502 Bad Gateway")

    def get_futures_balance_summary(self):
        return {"balance_summary": {
            "total_usd_balance": {"value": self._subset_total},
            "unrealized_pnl": {"value": self._upnl},
        }}


# ---------------------------------------------------------------------------
# 1. update_risk: unknown equity must not read as $0
# ---------------------------------------------------------------------------

class TestUpdateRiskUnknownEquity:
    def test_the_incident_cold_boot_eq_zero_does_not_halt(self):
        # Restored baseline is real, venue down -> eq 0.0. Old code: dd=100%,
        # sticky persisted halt. New code: skip, degraded, no halt.
        s = _bare_sleeve(_sleeve_start_equity=3997.75)
        s.sleeve_equity_usd = lambda: 0.0  # type: ignore[assignment]
        r = s.update_risk()
        assert s._halted is False, (
            "eq=0.0 against a restored baseline tripped the halt — the exact "
            "cold-boot false-halt this fix removes (unknown != $0)")
        assert r.get("degraded") is True
        assert r["drawdown_pct"] == 0.0  # last real dd, not a fabricated 1.0

    def test_negative_equity_read_also_skips(self):
        s = _bare_sleeve(_sleeve_start_equity=4000.0)
        s.sleeve_equity_usd = lambda: -1.0  # type: ignore[assignment]
        s.update_risk()
        assert s._halted is False

    def test_a_real_drawdown_still_halts(self):
        # The guard must not disarm the control it protects.
        s = _bare_sleeve(_sleeve_start_equity=4000.0)
        s.sleeve_equity_usd = lambda: 3000.0  # type: ignore[assignment]
        r = s.update_risk()
        assert s._halted is True, "a genuine 25% drawdown no longer halts — the guard over-reached"
        assert r["drawdown_pct"] == pytest.approx(0.25)

    def test_baseline_still_anchors_on_first_real_read(self):
        s = _bare_sleeve()
        s.sleeve_equity_usd = lambda: 3800.0  # type: ignore[assignment]
        s.update_risk()
        assert s._sleeve_start_equity == 3800.0

    def test_unknown_equity_does_not_anchor_the_baseline(self):
        s = _bare_sleeve()
        s.sleeve_equity_usd = lambda: 0.0  # type: ignore[assignment]
        s.update_risk()
        assert s._sleeve_start_equity is None

    def test_last_dd_served_while_degraded_reflects_prior_real_reading(self):
        s = _bare_sleeve(_sleeve_start_equity=4000.0)
        s.sleeve_equity_usd = lambda: 3900.0  # type: ignore[assignment]
        s.update_risk()
        s.sleeve_equity_usd = lambda: 0.0  # type: ignore[assignment]
        r = s.update_risk()
        assert r["drawdown_pct"] == pytest.approx(0.025)
        assert r.get("degraded") is True

    def test_a_tripped_halt_stays_tripped_through_a_degraded_pass(self):
        s = _bare_sleeve(_halted=True, _halt_reason="sleeve drawdown 16% >= 15%",
                         _sleeve_start_equity=4000.0)
        s.sleeve_equity_usd = lambda: 0.0  # type: ignore[assignment]
        r = s.update_risk()
        assert r["halted"] is True


# ---------------------------------------------------------------------------
# 2. sleeve_equity_usd: the FCM subset is diagnostic-only
# ---------------------------------------------------------------------------

class TestEquityFallbackDenomination:
    def _sleeve_with_client(self, client, last_known=0.0):
        s = _bare_sleeve(_last_equity_usd=last_known)
        s.is_ready = lambda: True  # type: ignore[assignment]
        s._adapter = types.SimpleNamespace(_client=client)
        return s

    def test_subset_is_never_returned_when_last_known_exists(self):
        s = self._sleeve_with_client(_RaisingClient(), last_known=3997.75)
        eq = s.sleeve_equity_usd()
        assert eq == pytest.approx(3997.75), (
            f"got {eq} — the FCM-only subset (~$451) was substituted for the "
            "portfolio equity; that wrong-denomination number is what tripped "
            "the phantom halt/fuse arithmetic")

    def test_subset_is_never_returned_on_a_never_priced_process(self):
        s = self._sleeve_with_client(_RaisingClient(), last_known=0.0)
        eq = s.sleeve_equity_usd()
        assert eq == 0.0, (
            f"got {eq} — a never-priced process must report UNKNOWN (0.0), "
            "never the FCM subset; consumers treat <=0 as unknown and hold")

    def test_last_known_is_not_overwritten_by_the_subset(self):
        s = self._sleeve_with_client(_RaisingClient(), last_known=3997.75)
        s.sleeve_equity_usd()
        assert s._last_equity_usd == pytest.approx(3997.75)

    def test_primary_path_still_wins_when_it_works(self):
        class _GoodClient(_RaisingClient):
            def get_portfolio_breakdown(self, portfolio_uuid=None):
                return {"breakdown": {"portfolio_balances": {
                    "total_balance": {"value": "4001.5"}}}}
        s = self._sleeve_with_client(_GoodClient())
        assert s.sleeve_equity_usd() == pytest.approx(4001.5)
        assert s._last_equity_usd == pytest.approx(4001.5)


class TestPortfolioUuidCaching:
    def test_malformed_response_does_not_poison_the_cache(self):
        class _NoUuidClient:
            def get_portfolios(self):
                return {"portfolios": [{"type": "DEFAULT"}]}  # no uuid key
        s = _bare_sleeve()
        s._adapter = types.SimpleNamespace(_client=_NoUuidClient())
        assert not s._portfolio_uuid()
        assert not s._cb_portfolio_uuid, (
            "an empty-string uuid was cached — that permanently disabled the "
            "primary equity path with no retry (the P265 aggravator)")

    def test_recovers_on_the_next_good_response(self):
        calls = {"n": 0}

        class _FlakyClient:
            def get_portfolios(self):
                calls["n"] += 1
                if calls["n"] == 1:
                    return {"portfolios": [{"type": "DEFAULT"}]}
                return {"portfolios": [{"type": "DEFAULT", "uuid": "real-uuid"}]}
        s = _bare_sleeve()
        s._adapter = types.SimpleNamespace(_client=_FlakyClient())
        assert not s._portfolio_uuid()
        assert s._portfolio_uuid() == "real-uuid"
        assert s._cb_portfolio_uuid == "real-uuid"


# ---------------------------------------------------------------------------
# 3. _update_drawdown_snapshot: the incident's exact numbers
# ---------------------------------------------------------------------------

def _dd_snapshot_self(**over):
    """Bound-fake self for HMATSProductionRunner._update_drawdown_snapshot
    (the P263 pattern): only the attributes the function reads."""
    fake = types.SimpleNamespace(
        # LIVE + routing enabled = "sleeve expected" — the incident's shape.
        config=types.SimpleNamespace(
            initial_capital=10_000.0,
            coinbase_routing_enabled=True,
            mode=types.SimpleNamespace(value="live"),
        ),
        account_sync=types.SimpleNamespace(get_equity=lambda: 7_088.69),
        _coinbase_sleeve=None,
        _market_pipeline=None,
    )
    for k, v in over.items():
        setattr(fake, k, v)
    return fake


def _run_dd_snapshot(fake):
    import main as main_mod
    return main_mod.HMATSProductionRunner._update_drawdown_snapshot(fake)


class TestDrawdownSnapshotColdBoot:
    INCIDENT_PEAK = 10_865.13
    INCIDENT_HELD = 10_864.14  # p0_last_combined_equity from the live volume

    def test_unbuilt_sleeve_fresh_process_uses_the_held_combined(self):
        fake = _dd_snapshot_self(
            _peak_equity=self.INCIDENT_PEAK,
            _p0_last_combined_equity=self.INCIDENT_HELD,
        )
        eq, dd = _run_dd_snapshot(fake)
        assert dd < 0.05, (
            f"dd={dd:.2%} — the Kraken-only book was measured against the "
            "restored combined peak: the exact 34.76% phantom of the "
            "2026-08-13 incident (P261 arithmetic on the third equity feed)")
        assert eq == pytest.approx(self.INCIDENT_HELD)

    def test_built_but_unpriced_sleeve_fresh_process_uses_the_held_combined(self):
        sleeve = types.SimpleNamespace(
            sleeve_equity_usd=lambda: 0.0, _last_equity_usd=0.0)
        fake = _dd_snapshot_self(
            _coinbase_sleeve=sleeve,
            _peak_equity=self.INCIDENT_PEAK,
            _p0_last_combined_equity=self.INCIDENT_HELD,
        )
        eq, dd = _run_dd_snapshot(fake)
        assert dd < 0.05
        assert eq == pytest.approx(self.INCIDENT_HELD)

    def test_prior_drawdown_still_wins_over_the_combined_fallback(self):
        fake = _dd_snapshot_self(
            _peak_equity=self.INCIDENT_PEAK,
            _p0_last_combined_equity=self.INCIDENT_HELD,
            _current_drawdown_pct=0.031,
        )
        _eq, dd = _run_dd_snapshot(fake)
        assert dd == pytest.approx(0.031)

    def test_no_combined_anchor_is_the_only_kraken_only_path(self):
        # Genuinely single-venue history: no anchor exists, Kraken-only is the
        # whole book by definition. Documents the surviving (correct) path.
        fake = _dd_snapshot_self(_peak_equity=7_100.0)
        eq, dd = _run_dd_snapshot(fake)
        assert eq == pytest.approx(7_088.69)
        assert 0.0 <= dd < 0.01

    def test_healthy_sleeve_still_folds_normally(self):
        sleeve = types.SimpleNamespace(
            sleeve_equity_usd=lambda: 3_775.45, _last_equity_usd=3_775.45)
        fake = _dd_snapshot_self(
            _coinbase_sleeve=sleeve,
            _peak_equity=self.INCIDENT_PEAK,
            _p0_last_combined_equity=self.INCIDENT_HELD,
        )
        eq, dd = _run_dd_snapshot(fake)
        assert eq == pytest.approx(7_088.69 + 3_775.45)
        assert dd == pytest.approx(
            (self.INCIDENT_PEAK - eq) / self.INCIDENT_PEAK)

    def test_a_sleeve_less_run_never_enters_the_hold_branch(self):
        # Paper/tests/single-venue: no sleeve and routing off. The first
        # version of this fix routed no-sleeve into the hold branch, which
        # froze dd at its first value forever (caught by the P163 suite).
        fake = _dd_snapshot_self(
            _peak_equity=10_000.0,
            _current_drawdown_pct=0.0,  # a prior dd exists — must NOT be held
            _p0_last_combined_equity=self.INCIDENT_HELD,
        )
        fake.config.coinbase_routing_enabled = False
        fake.account_sync = types.SimpleNamespace(get_equity=lambda: 8_000.0)
        _eq, dd = _run_dd_snapshot(fake)
        assert dd == pytest.approx(0.20), (
            f"dd={dd:.2%} — a sleeve-less run held/substituted instead of "
            "computing the real Kraken-only drawdown")

    def test_a_real_combined_drawdown_still_registers(self):
        # The fallback must not mask a genuine loss once equity is readable.
        sleeve = types.SimpleNamespace(
            sleeve_equity_usd=lambda: 2_000.0, _last_equity_usd=2_000.0)
        fake = _dd_snapshot_self(
            _coinbase_sleeve=sleeve,
            _peak_equity=self.INCIDENT_PEAK,
            _p0_last_combined_equity=self.INCIDENT_HELD,
        )
        _eq, dd = _run_dd_snapshot(fake)
        assert dd > 0.15


# ---------------------------------------------------------------------------
# 4. source pins (structure that cannot be tested behaviorally without a live
#    loop; each verified red-on-revert during development)
# ---------------------------------------------------------------------------

class TestSourcePins:
    def test_the_drawdown_halt_stops_the_tick_immediately(self):
        # Between the halt message and the NAV except-handler there must be a
        # bare `continue` after `_running = False` — without it the halt tick
        # ran the full decide loop and the sleeve driver.
        start = MAIN_SRC.index("STOPPING LIVE TRADING")
        end = MAIN_SRC.index("except Exception as _nav_err", start)
        block = MAIN_SRC[start:end]
        m = re.search(r"self\._running\s*=\s*False(.*)", block, re.S)
        assert m is not None
        assert re.search(r"^\s*continue\s*$", m.group(1), re.M), (
            "the LIVE drawdown halt no longer `continue`s — the halt tick "
            "will run one more full trading iteration (P265)")

    def test_force_flat_already_flat_is_gated_on_reconcile_ok(self):
        start = MAIN_SRC.index("[EMERGENCY_FLAT] Coinbase sleeve holds")
        seg = MAIN_SRC[start:start + 4000]
        flat_idx = seg.index("sleeve already flat")
        gate_idx = seg.index('_reconcile_ok')
        assert gate_idx < flat_idx, (
            '"already flat" is reachable without a _reconcile_ok check — a '
            "failed venue read reports as flat on the emergency path (P265)")
        assert "UNREADABLE" in seg

    def test_force_flat_attempts_sleeve_construction_and_names_the_gap(self):
        start = MAIN_SRC.index("[EMERGENCY_FLAT] Coinbase sleeve NOT CONSTRUCTED")
        seg = MAIN_SRC[max(0, start - 2000):start]
        assert "_ensure_coinbase_sleeve" in seg, (
            "the unbuilt-sleeve branch no longer tries to construct the "
            "sleeve before declaring it untouchable")
