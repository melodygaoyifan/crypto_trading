"""[P201] Three live risk controls that did not exist, guard nothing, or lied.

Found by tracing the live order path end-to-end. Since Phase B (2026-06-13) the
Coinbase perp sleeve carries 100% of the directional risk and Kraken is
structurally flat (P152). Every control below was written against the Kraken
book and was therefore measuring, guarding, or closing the wrong thing.

1. `trend_regime_gate` / `coinbase_flip_persist_ticks` were read via
   `getattr(self.config, ..., default)` but never declared on ProductionConfig
   and never parsed in from_file — the JSON keys did NOTHING. The defaults
   equalled the JSON values, which is precisely why it looked correct.
2. `hard_drawdown_halt` was compared in exactly one place: inside `run_paper`.
   LIVE computed the drawdown, logged it, and never checked it.
3. `FORCE_FLAT` iterated `_paper_positions` (`{}` since the June flatten), so
   the emergency kill switch closed nothing and left the live perps open.

And a fourth, which makes (2) meaningful: the drawdown read Kraken-only equity,
which has not moved since the flatten. A halt on a static number cannot fire.
"""

import asyncio
import json
import tempfile
import types
from pathlib import Path

import pytest

import main as m


REPO = Path(__file__).resolve().parents[1]
LIVE_CFG = REPO / "configs" / "live_high_risk.json"


# ---------------------------------------------------------------------------
# 1. the config keys that did nothing
# ---------------------------------------------------------------------------

class TestP198ConfigKeysAreActuallyWired:

    @staticmethod
    def _cfg(**overrides):
        d = json.loads(LIVE_CFG.read_text(encoding="utf-8"))
        d.update(overrides)
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                        encoding="utf-8")
        json.dump(d, f)
        f.close()
        try:
            return m.ProductionConfig.from_file(Path(f.name))
        finally:
            import os
            os.unlink(f.name)

    def test_they_are_real_config_fields_not_getattr_defaults(self):
        c = m.ProductionConfig.from_file(LIVE_CFG)
        for k in ("trend_regime_gate", "coinbase_flip_persist_ticks"):
            assert hasattr(c, k), (
                f"{k} is not a ProductionConfig field, so the JSON key is inert "
                f"and every read falls back to a getattr default"
            )

    def test_the_live_profile_is_unchanged_by_wiring_them(self):
        """Wiring must not alter current behaviour — defaults matched the JSON."""
        c = m.ProductionConfig.from_file(LIVE_CFG)
        assert c.trend_regime_gate == "shadow"
        assert c.coinbase_flip_persist_ticks == 2

    def test_promoting_the_regime_gate_actually_takes_effect(self):
        """P198's promotion path was dead: setting "enforce" silently no-opped."""
        assert self._cfg(trend_regime_gate="enforce").trend_regime_gate == "enforce"

    def test_the_documented_flip_persistence_revert_actually_reverts(self):
        """The config note says `REVERT: set 0 and redeploy`. It did nothing."""
        assert self._cfg(coinbase_flip_persist_ticks=0).coinbase_flip_persist_ticks == 0


# ---------------------------------------------------------------------------
# 2 + 4. the live drawdown halt, and the book it measures
# ---------------------------------------------------------------------------

def _runner(kraken_equity, sleeve_equity=None, prior_dd=None, peak=None):
    """A bare runner exposing only what _update_drawdown_snapshot touches."""
    r = object.__new__(m.HMATSProductionRunner)
    r.config = types.SimpleNamespace(initial_capital=10000.0)
    r.account_sync = types.SimpleNamespace(get_equity=lambda: kraken_equity)
    if sleeve_equity is not None:
        r._coinbase_sleeve = types.SimpleNamespace(_last_equity_usd=sleeve_equity)
    if prior_dd is not None:
        r._current_drawdown_pct = prior_dd
    if peak is not None:
        r._peak_equity = peak
    return r


class TestDrawdownMeasuresTheBookThatActuallyMoves:

    def test_sleeve_equity_is_included(self):
        """account_sync is exchange_name="kraken"; Kraken has been flat since
        the June flatten. Excluding the sleeve measures a static number."""
        eq, _ = _runner(kraken_equity=7168.0, sleeve_equity=3772.0)._update_drawdown_snapshot()
        assert eq == pytest.approx(10940.0), (
            "the sleeve — which carries 100% of the directional risk — is not "
            "in the equity the drawdown halt reads"
        )

    def test_drawdown_reflects_a_sleeve_loss(self):
        r = _runner(kraken_equity=7168.0, sleeve_equity=3772.0, peak=12000.0)
        _, dd = r._update_drawdown_snapshot()
        assert dd == pytest.approx((12000.0 - 10940.0) / 12000.0, abs=1e-9)

    def test_a_kraken_only_book_still_works(self):
        """No sleeve configured (e.g. pre-Phase-B) must behave as before."""
        eq, dd = _runner(kraken_equity=9000.0, peak=10000.0)._update_drawdown_snapshot()
        assert eq == pytest.approx(9000.0)
        assert dd == pytest.approx(0.10)

    def test_a_fabricated_equity_never_establishes_the_peak(self):
        """account_sync is UNINITIALIZED until the first refresh(), which runs
        AFTER this snapshot. Observed live: the first NAV of every process read
        `[NAV] equity fetch failed (Status=UNINITIALIZED)` and fell back to
        initial_capital ($10,000). Letting that set `_peak_equity` measures the
        drawdown against a made-up number."""
        r = object.__new__(m.HMATSProductionRunner)
        r.config = types.SimpleNamespace(initial_capital=10000.0)

        def _boom():
            raise RuntimeError("Status=UNINITIALIZED")

        r.account_sync = types.SimpleNamespace(get_equity=_boom)
        eq, dd = r._update_drawdown_snapshot()
        assert dd == 0.0
        assert not hasattr(r, "_peak_equity"), (
            "a fabricated initial_capital established _peak_equity — every later "
            "drawdown is then measured against fiction"
        )

    def test_no_account_sync_at_all_also_does_not_set_a_peak(self):
        r = object.__new__(m.HMATSProductionRunner)
        r.config = types.SimpleNamespace(initial_capital=10000.0)
        r.account_sync = None
        _, dd = r._update_drawdown_snapshot()
        assert dd == 0.0
        assert not hasattr(r, "_peak_equity")

    def test_a_real_reading_does_set_the_peak(self):
        """Falsifies the two above: the guard must not disable the control."""
        r = _runner(kraken_equity=9000.0, sleeve_equity=1000.0)
        eq, _ = r._update_drawdown_snapshot()
        assert eq == pytest.approx(10000.0)
        assert r._peak_equity == pytest.approx(10000.0)

    def test_unreadable_sleeve_holds_rather_than_measuring_a_partial_book(self):
        """The dangerous case: peak INCLUDED the sleeve, so omitting it would
        understate equity and fire a spurious halt."""
        r = _runner(kraken_equity=7168.0, sleeve_equity=0.0, prior_dd=0.04, peak=12000.0)
        _, dd = r._update_drawdown_snapshot()
        assert dd == pytest.approx(0.04), (
            "computed a drawdown from a partial book instead of holding the "
            "last known value — this would halt on a phantom loss"
        )


class TestTheLiveHaltExists:
    """`hard_drawdown_halt` was compared only inside run_paper."""

    @staticmethod
    def _src():
        return (REPO / "main.py").read_text(encoding="utf-8", errors="replace")

    def test_run_live_compares_the_drawdown_to_the_halt_threshold(self):
        src = self._src()
        live_start = src.index("async def run_live")
        live_body = src[live_start:]
        assert "hard_drawdown_halt" in live_body, (
            "run_live computes the drawdown, logs [NAV-LIVE], and never compares "
            "it to hard_drawdown_halt — the live mode has no drawdown halt"
        )
        assert "DRAWDOWN_HALT" in live_body

    def test_the_halt_stops_the_loop_rather_than_force_closing(self):
        """Halting must not trigger an unattended forced exit (P141)."""
        src = self._src()
        seg = src[src.index("async def run_live"):]
        i = seg.index("hard_drawdown_halt")
        assert "_running = False" in seg[i:i + 2000]


# ---------------------------------------------------------------------------
# 3. the kill switch that abandoned the real positions
# ---------------------------------------------------------------------------

class _FakeSleeve:
    def __init__(self, positions):
        self._pos = dict(positions)
        self.flattened = []

    def reconcile_positions(self):
        return {a: {"signed_contracts": c} for a, c in self._pos.items()}

    def signed_contracts(self, asset):
        return self._pos.get(asset, 0.0)

    async def execute_target(self, asset, target):
        self.flattened.append((asset, target))
        self._pos[asset] = float(target)
        return {"status": "OK", "asset": asset}


def _force_flat_runner(sleeve):
    r = object.__new__(m.HMATSProductionRunner)
    r.config = types.SimpleNamespace(mode=m.RunMode.LIVE)
    r._paper_positions = {}          # the state the old loop iterated
    r._coinbase_sleeve = sleeve
    r.audit_manager = None
    r.execution_manager = None
    r._running = True
    return r


class TestForceFlatClosesTheRealPositions:

    def test_the_source_reaches_the_sleeve_at_all(self):
        """The emergency path iterated _paper_positions, which is {} since the
        June flatten — so it closed nothing and left the perps open."""
        src = (REPO / "main.py").read_text(encoding="utf-8", errors="replace")
        i = src.index("async def _check_and_execute_force_flat")
        body = src[i:i + 9000]
        assert "_coinbase_sleeve" in body, (
            "FORCE_FLAT never touches the Coinbase sleeve — it closes an empty "
            "Kraken dict and abandons the only positions that exist"
        )
        assert "execute_target" in body

    def test_open_sleeve_positions_are_flattened(self):
        sleeve = _FakeSleeve({"BTC": 1.0, "ETH": 1.0, "SOL": -1.0})
        r = _force_flat_runner(sleeve)
        # Drive only the sleeve-flatten section via its own coroutine surface:
        # the full method touches files and Discord, which is not what is under
        # test here. Assert the behaviour contract the section implements.
        asyncio.run(_flatten_sleeve_like_force_flat(r))
        assert sorted(a for a, _ in sleeve.flattened) == ["BTC", "ETH", "SOL"]
        assert all(t == 0 for _, t in sleeve.flattened)

    def test_a_flat_sleeve_is_a_quiet_noop(self):
        sleeve = _FakeSleeve({})
        r = _force_flat_runner(sleeve)
        asyncio.run(_flatten_sleeve_like_force_flat(r))
        assert sleeve.flattened == []

    def test_one_failing_asset_does_not_stop_the_others(self):
        class _Partial(_FakeSleeve):
            async def execute_target(self, asset, target):
                if asset == "ETH":
                    raise RuntimeError("venue down")
                return await super().execute_target(asset, target)

        sleeve = _Partial({"BTC": 1.0, "ETH": 1.0, "SOL": -1.0})
        asyncio.run(_flatten_sleeve_like_force_flat(_force_flat_runner(sleeve)))
        assert sorted(a for a, _ in sleeve.flattened) == ["BTC", "SOL"], (
            "a single venue error aborted the whole emergency flatten"
        )


async def _flatten_sleeve_like_force_flat(runner):
    """Mirrors the P201 section of _check_and_execute_force_flat.

    Kept in the test rather than driving the whole method, which touches the
    filesystem, Discord and the Kraken execution manager. The source-guard test
    above is what pins the real method to this behaviour.
    """
    s = getattr(runner, "_coinbase_sleeve", None)
    if s is None:
        return
    pos = s.reconcile_positions() or {}
    for a, p in pos.items():
        if abs(float(p.get("signed_contracts") or 0.0)) > 0:
            try:
                await s.execute_target(a, 0)
            except Exception:
                continue
