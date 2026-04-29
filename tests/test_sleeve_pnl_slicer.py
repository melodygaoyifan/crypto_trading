"""Phase 6.0 prep — sleeve PnL slicer + realized-vol updater unit tests."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from analytics.sleeve_attribution.compute_sleeve_pnl import (
    aggregate_daily_sleeve_pnl,
    build_report,
    compute_vol_and_sharpe,
    load_fills,
    map_agent_to_sleeve,
    update_allocator_from_report,
)


# ---------- Sleeve mapping --------------------------------------------------

@pytest.mark.parametrize("agent,expected", [
    ("quant", "directional_short"),
    ("drl", "directional_short"),
    ("DRL", "directional_short"),                # case-insensitive
    ("kraken_quant", "directional_short"),
    ("sentiment", "directional_short"),
    ("ofi", "microstructure"),
    ("vpin_spike", "microstructure"),
    ("kyle_lambda", "microstructure"),
    ("phase4_anything", "microstructure"),
    ("cascade_anticipation", "cascade"),
    ("stop_hunt_defense", "cascade"),
    ("phase8_x", "cascade"),
    ("totally_unknown_agent", "unknown"),
    ("", "unknown"),
    (None, "unknown"),
])
def test_map_agent_to_sleeve(agent, expected):
    assert map_agent_to_sleeve(agent) == expected


# ---------- load_fills ------------------------------------------------------

def _write_fill_line(f, ts: datetime, asset: str, primary_agent: str,
                    realized_pnl: float = 0.0, fee: float = 1.0,
                    notional: float = 1000.0, side: str = "BUY"):
    rec = {
        "timestamp": ts.isoformat(),
        "entry_type": "FILL",
        "asset": asset,
        "data": {
            "primary_agent": primary_agent,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "trade_fee_usd": fee,
            "notional": notional,
            "side": side,
            "size": 0.01,
            "price": 60000.0,
        },
    }
    f.write(json.dumps(rec) + "\n")


def test_load_fills_missing_dir(tmp_path: Path):
    out = load_fills(tmp_path / "does_not_exist")
    assert out == []


def test_load_fills_filters_by_since(tmp_path: Path):
    p = tmp_path / "ledger_20260429.jsonl"
    old = datetime.now(timezone.utc) - timedelta(days=40)
    new = datetime.now(timezone.utc)
    with p.open("w", encoding="utf-8") as f:
        _write_fill_line(f, old, "BTC", "quant", realized_pnl=10.0)
        _write_fill_line(f, new, "BTC", "drl", realized_pnl=20.0)

    since = datetime.now(timezone.utc) - timedelta(days=14)
    rows = load_fills(tmp_path, since=since)
    assert len(rows) == 1
    assert rows[0]["primary_agent"] == "drl"
    assert rows[0]["realized_pnl"] == 20.0


def test_load_fills_skips_malformed(tmp_path: Path):
    p = tmp_path / "ledger_20260429.jsonl"
    p.write_text(
        '{ this is not json }\n'
        + '{"entry_type": "FILL", "no_timestamp": true}\n'
        + '{"timestamp": "not-a-date", "entry_type": "FILL", "data": {}}\n',
        encoding="utf-8",
    )
    rows = load_fills(tmp_path)
    assert rows == []


def test_load_fills_ignores_non_fill_records(tmp_path: Path):
    p = tmp_path / "ledger_20260429.jsonl"
    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entry_type": "INTENT",
        "asset": "BTC",
        "data": {"primary_agent": "drl"},
    }
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    assert load_fills(tmp_path) == []


def test_load_fills_handles_nan_pnl(tmp_path: Path):
    p = tmp_path / "ledger_20260429.jsonl"
    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entry_type": "FILL",
        "asset": "BTC",
        "data": {"primary_agent": "drl", "realized_pnl": "bad", "fee": 0.5},
    }
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    rows = load_fills(tmp_path)
    assert len(rows) == 1
    assert rows[0]["realized_pnl"] == 0.0
    assert rows[0]["fee"] == 0.5


# ---------- aggregate_daily_sleeve_pnl -------------------------------------

def test_aggregate_daily_sleeve_pnl_sums_correctly():
    ts = datetime(2026, 4, 28, 12, 0, 0, tzinfo=timezone.utc)
    fills = [
        {"ts": ts, "asset": "BTC", "sleeve": "directional_short",
         "primary_agent": "drl", "realized_pnl": 100.0, "fee": 5.0,
         "trade_fee_usd": 5.0, "notional": 1000, "side": "BUY"},
        {"ts": ts + timedelta(hours=1), "asset": "BTC", "sleeve": "directional_short",
         "primary_agent": "drl", "realized_pnl": 50.0, "fee": 2.0,
         "trade_fee_usd": 2.0, "notional": 500, "side": "SELL"},
        {"ts": ts, "asset": "BTC", "sleeve": "microstructure",
         "primary_agent": "ofi", "realized_pnl": 0.0, "fee": 1.0,
         "trade_fee_usd": 1.0, "notional": 200, "side": "BUY"},
    ]
    out = aggregate_daily_sleeve_pnl(fills)
    assert "directional_short" in out and "microstructure" in out
    # directional_short net = (100 - 5) + (50 - 2) = 143
    assert out["directional_short"]["2026-04-28"] == 143.0
    # microstructure = 0 - 1 = -1
    assert out["microstructure"]["2026-04-28"] == -1.0


def test_aggregate_daily_sleeve_pnl_splits_days():
    fills = [
        {"ts": datetime(2026, 4, 28, 23, 0, tzinfo=timezone.utc),
         "asset": "BTC", "sleeve": "directional_short", "primary_agent": "drl",
         "realized_pnl": 50.0, "fee": 0.0, "trade_fee_usd": 0.0,
         "notional": 1000, "side": "BUY"},
        {"ts": datetime(2026, 4, 29, 1, 0, tzinfo=timezone.utc),
         "asset": "BTC", "sleeve": "directional_short", "primary_agent": "drl",
         "realized_pnl": 75.0, "fee": 0.0, "trade_fee_usd": 0.0,
         "notional": 1000, "side": "SELL"},
    ]
    out = aggregate_daily_sleeve_pnl(fills)
    assert out["directional_short"]["2026-04-28"] == 50.0
    assert out["directional_short"]["2026-04-29"] == 75.0


# ---------- compute_vol_and_sharpe -----------------------------------------

def test_compute_vol_zero_for_empty():
    vol, sh, n = compute_vol_and_sharpe({})
    assert vol == 0.0 and sh == 0.0 and n == 0


def test_compute_vol_zero_for_single_day():
    vol, sh, n = compute_vol_and_sharpe({"2026-04-29": 100.0})
    assert vol == 0.0 and sh == 0.0 and n == 1


def test_compute_vol_positive_on_realistic_input():
    # 7 days of returns averaging +1% with some volatility
    daily = {
        "2026-04-21":  50.0,
        "2026-04-22": -30.0,
        "2026-04-23":  80.0,
        "2026-04-24": -20.0,
        "2026-04-25":  40.0,
        "2026-04-26":  60.0,
        "2026-04-27": -10.0,
    }
    vol, sh, n = compute_vol_and_sharpe(daily, initial_capital=10_000.0)
    assert n == 7
    assert 0.0 < vol < 5.0          # plausible annualized vol range
    # Net positive PnL → positive Sharpe
    assert sh > 0


def test_compute_vol_zero_when_constant_zero():
    daily = {f"2026-04-{i:02d}": 0.0 for i in range(20, 27)}
    vol, sh, n = compute_vol_and_sharpe(daily)
    assert vol == 0.0 and sh == 0.0 and n == 7


# ---------- build_report end-to-end -----------------------------------------

def test_build_report_end_to_end(tmp_path: Path):
    p = tmp_path / "ledger_20260429.jsonl"
    now = datetime.now(timezone.utc)
    with p.open("w", encoding="utf-8") as f:
        # Spread fills across 5 days with mixed sleeves
        for i in range(5):
            ts = now - timedelta(days=i)
            _write_fill_line(f, ts, "BTC", "drl",
                             realized_pnl=10.0 + i, fee=0.5, notional=1000)
            _write_fill_line(f, ts, "BTC", "quant",
                             realized_pnl=5.0, fee=0.2, notional=500)
            _write_fill_line(f, ts, "BTC", "ofi",
                             realized_pnl=0.0, fee=0.1, notional=200)

    report = build_report(p.parent, window_days=14, initial_capital=10_000.0)
    assert report["n_total_fills"] == 15
    assert report["window_days"] == 14
    sleeves = {s["name"]: s for s in report["sleeves"]}
    assert "directional_short" in sleeves
    assert "microstructure" in sleeves
    # directional_short has 5 days × 2 fills/day = 10 fills
    assert sleeves["directional_short"]["n_fills"] == 10
    assert sleeves["microstructure"]["n_fills"] == 5


# ---------- update_allocator_from_report (Phase 10 hook) -------------------

def test_update_allocator_from_report_feeds_realized_vols():
    from risk.sleeve_allocator_v5_1 import build_phase7_sleeve_allocator
    a = build_phase7_sleeve_allocator()
    report = {
        "sleeves": [
            {"name": "directional_short", "annualized_vol": 0.32},
            {"name": "microstructure",   "annualized_vol": 0.18},
            {"name": "cascade",          "annualized_vol": 0.0},  # skipped
            {"name": "unknown",          "annualized_vol": 0.10},  # not registered
        ]
    }
    n = update_allocator_from_report(a, report)
    # 2 updated (directional + microstructure); cascade skipped (vol=0);
    # unknown skipped because not registered (logged debug, not error)
    assert n == 2
    assert len(a._sleeves["directional_short"].realized_vol_history) == 1
    assert a._sleeves["directional_short"].realized_vol_history[0] == 0.32
    assert len(a._sleeves["microstructure"].realized_vol_history) == 1


def test_update_allocator_from_report_handles_allocator_exception():
    """Iron Law 4: an allocator method raising must not abort the loop."""
    class _BadAllocator:
        def update_realized_vol(self, name, vol):
            raise RuntimeError("nope")
    report = {"sleeves": [
        {"name": "directional_short", "annualized_vol": 0.30},
        {"name": "microstructure",   "annualized_vol": 0.20},
    ]}
    n = update_allocator_from_report(_BadAllocator(), report)
    assert n == 0  # both raised, neither counted as updated


# ---------- Iron Law check: no live sizer touch ---------------------------

def test_slicer_does_not_import_unified_position_sizer():
    src = Path("analytics/sleeve_attribution/compute_sleeve_pnl.py").read_text(encoding="utf-8")
    assert "from risk.unified_position_sizer" not in src
    assert "import unified_position_sizer" not in src
