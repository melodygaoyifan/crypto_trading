"""[P150] Coinbase sleeve persistence — the 15% drawdown halt baseline and the
sticky halt must survive a container restart, and the forward-PnL log must be
appended each tick. Without this the "loss is capped" guarantee re-anchors to
post-restart equity (same bug class as P148 DRL buffer / P140 _peak_equity)."""

import json
import os

import pytest

from exchange.coinbase_sleeve import CoinbaseSleeve


class _StubAdapter:
    """Not connected — sleeve.is_ready() is False, so reconcile/buying_power use
    the last-known values we set directly. Persistence is independent of the venue."""

    def is_connected(self):
        return False


def _sleeve(tmp_path, monkeypatch, start_eq=None, **kw):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    s = CoinbaseSleeve(_StubAdapter(), assets=("BTC", "ETH", "SOL"), **kw)
    if start_eq is not None:
        s._last_buying_power_usd = start_eq  # equity = bp + upnl(0)
    return s


def test_baseline_persists_across_restart(tmp_path, monkeypatch):
    a = _sleeve(tmp_path, monkeypatch, start_eq=4000.0)
    r = a.update_risk()  # sets + persists baseline = 4000
    assert a._sleeve_start_equity == pytest.approx(4000.0)
    assert r["drawdown_pct"] == pytest.approx(0.0)
    assert os.path.exists(a._state_path())

    # restart: equity has since dropped to 3600 (-10% from inception)
    b = _sleeve(tmp_path, monkeypatch, start_eq=3600.0)
    assert b._sleeve_start_equity == pytest.approx(4000.0)  # restored, NOT re-anchored
    r2 = b.update_risk()
    assert r2["drawdown_pct"] == pytest.approx(0.10, abs=1e-6)  # measured from 4000, not 3600
    assert r2["halted"] is False  # 10% < 15% cap


def test_halt_triggers_from_restored_baseline(tmp_path, monkeypatch):
    a = _sleeve(tmp_path, monkeypatch, start_eq=4000.0)
    a.update_risk()  # baseline 4000

    # restart at 3300 = -17.5% from inception -> must halt (would NOT if re-anchored)
    b = _sleeve(tmp_path, monkeypatch, start_eq=3300.0)
    r = b.update_risk()
    assert r["halted"] is True
    assert r["drawdown_pct"] >= 0.15


def test_sticky_halt_survives_restart(tmp_path, monkeypatch):
    a = _sleeve(tmp_path, monkeypatch, start_eq=4000.0)
    a.update_risk()
    a._last_buying_power_usd = 3300.0
    a.update_risk()  # trips + persists halt
    assert a._halted is True

    # restart: even if equity recovered, the halt is sticky (manual recovery only)
    b = _sleeve(tmp_path, monkeypatch, start_eq=4100.0)
    assert b._halted is True
    allowed, reason = b.can_trade("BTC", -1)
    assert allowed is False
    assert "halted" in reason


def test_reset_halt_clears_persisted(tmp_path, monkeypatch):
    a = _sleeve(tmp_path, monkeypatch, start_eq=4000.0)
    a.update_risk()
    a._last_buying_power_usd = 3000.0
    a.update_risk()
    assert a._halted is True
    a.reset_halt()
    # a fresh instance must see the cleared halt
    b = _sleeve(tmp_path, monkeypatch, start_eq=4000.0)
    assert b._halted is False


def test_pnl_log_appends_each_tick(tmp_path, monkeypatch):
    a = _sleeve(tmp_path, monkeypatch, start_eq=4000.0)
    a.update_risk()  # baseline
    a.log_pnl_point()
    a._last_buying_power_usd = 4080.0
    a.log_pnl_point()
    p = a._pnl_path()
    assert os.path.exists(p)
    with open(p, encoding="utf-8") as fh:
        lines = [json.loads(x) for x in fh if x.strip()]
    assert len(lines) == 2
    assert lines[0]["start_equity_usd"] == pytest.approx(4000.0)
    assert lines[1]["pnl_usd"] == pytest.approx(80.0)
    assert lines[1]["pnl_pct"] == pytest.approx(0.02, abs=1e-6)


def test_no_state_file_starts_fresh(tmp_path, monkeypatch):
    a = _sleeve(tmp_path, monkeypatch, start_eq=4000.0)
    assert a._sleeve_start_equity is None  # nothing to restore
    assert a._halted is False


def test_corrupt_state_file_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    p = os.path.join(str(tmp_path), "coinbase_sleeve_state.json")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("{not valid json")
    # must not raise — degrades to fresh baseline
    s = CoinbaseSleeve(_StubAdapter(), assets=("BTC",))
    assert s._sleeve_start_equity is None
