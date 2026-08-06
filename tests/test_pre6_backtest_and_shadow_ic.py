"""Phase Pre-6 — backtest engine + shadow IC compute unit tests."""

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from analytics.shadow_ic.compute_shadow_ic import (
    Verdict,
    _spearman as ic_spearman,
    determine_verdict,
    load_shadow_ledgers,
)
from training.backtest_framework.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    FeeSchedule,
)


# ---------- FeeSchedule -----------------------------------------------------

def test_fee_schedule_v14_default():
    fs = FeeSchedule()
    assert fs.maker_bps == 0.0
    assert fs.taker_bps == 3.0
    # round_trip_bps = 2 * (0.987*0 + 0.013*3) = 2 * 0.039 = 0.078
    assert abs(fs.round_trip_bps() - 0.078) < 1e-9


def test_fee_schedule_all_taker():
    fs = FeeSchedule(maker_pct=0.0)
    # 2 * 3 = 6 bps round-trip when 100% taker
    assert abs(fs.round_trip_bps() - 6.0) < 1e-9


# ---------- Spearman --------------------------------------------------------

def test_spearman_perfect_positive():
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 30, 40, 50]
    assert abs(BacktestEngine._spearman(x, y) - 1.0) < 1e-9


def test_spearman_perfect_negative():
    x = [1, 2, 3, 4, 5]
    y = [50, 40, 30, 20, 10]
    assert abs(BacktestEngine._spearman(x, y) + 1.0) < 1e-9


def test_spearman_zero_for_constant():
    assert BacktestEngine._spearman([1, 1, 1, 1], [2, 3, 4, 5]) == 0.0
    assert ic_spearman([1, 1, 1, 1], [2, 3, 4, 5]) == 0.0


def test_spearman_short_input_returns_zero():
    assert BacktestEngine._spearman([1, 2], [3, 4]) == 0.0


# ---------- BacktestEngine on synthetic strategy ---------------------------

class _ConstantLongStrategy:
    def evaluate(self, asset, market_data):
        from strategies.microstructure_v5_1 import ShadowSignal
        return ShadowSignal(
            strategy_name="const_long",
            asset=asset,
            direction=1.0,
            confidence=0.5,
            reason="test",
            diagnostics={},
        )


class _NeutralStrategy:
    def evaluate(self, asset, market_data):
        from strategies.microstructure_v5_1 import ShadowSignal
        return ShadowSignal("neutral", asset, 0.0, 0.0, "no_op", {})


def _synthetic_parquet(tmp_path: Path, n_rows: int = 100) -> Path:
    """Build a tiny synthetic 4H parquet for backtest tests."""
    import pandas as pd
    ts0 = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    price = 100.0
    for i in range(n_rows):
        ts = ts0 + pd.Timedelta(hours=4 * i)
        # Trending up 0.5% per bar
        price *= 1.005
        rows.append({
            "timestamp": ts.tz_convert(None),  # parquet stores naive
            "open": price * 0.999,
            "high": price * 1.001,
            "low": price * 0.998,
            "close": price,
            "volume": 1.0e6,
            "ret_4h": 0.005,
            "regime": "STEADY_UPTREND",
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "SYNTH_4H_full.parquet"
    df.to_parquet(path)
    return path


def test_backtest_engine_runs_on_synthetic_uptrend(tmp_path: Path):
    parquet_path = _synthetic_parquet(tmp_path, n_rows=120)
    cfg = BacktestConfig(asset="SYNTH", parquet_path=parquet_path,
                         confidence_threshold=0.30)
    engine = BacktestEngine(cfg)
    result = engine.run(_ConstantLongStrategy())

    assert isinstance(result, BacktestResult)
    assert result.n_bars == 120
    # confidence=0.5 > threshold=0.3, direction=1.0 -> every bar generates signal
    # except the last `max_horizon` bars
    assert result.n_signals > 0
    assert result.n_trades > 0
    # Uptrend + always long -> positive PnL
    assert result.gross_pnl > 0
    # IC at 4b should be high (always-long on always-up-trend)
    assert result.ic_per_horizon[4] >= 0.0  # may be 0 due to constant signal
    # Win rate should be ~100% on monotone uptrend
    assert result.win_rate > 0.9


def test_backtest_engine_neutral_strategy_no_trades(tmp_path: Path):
    parquet_path = _synthetic_parquet(tmp_path, n_rows=60)
    cfg = BacktestConfig(asset="SYNTH", parquet_path=parquet_path)
    engine = BacktestEngine(cfg)
    result = engine.run(_NeutralStrategy())
    assert result.n_signals == 0
    assert result.n_trades == 0
    assert result.gross_pnl == 0.0


def test_backtest_engine_missing_parquet_raises(tmp_path: Path):
    cfg = BacktestConfig(asset="NOPE", parquet_path=tmp_path / "missing.parquet")
    engine = BacktestEngine(cfg)
    with pytest.raises(FileNotFoundError):
        engine.run(_ConstantLongStrategy())


def test_backtest_engine_handles_strategy_exception(tmp_path: Path):
    """Iron Law 4: strategy raising mid-run does not crash the engine."""
    class _BadStrategy:
        def __init__(self):
            self.calls = 0
        def evaluate(self, asset, md):
            self.calls += 1
            if self.calls > 5:
                raise RuntimeError("kaboom")
            from strategies.microstructure_v5_1 import ShadowSignal
            return ShadowSignal("bad", asset, 1.0, 0.5, "ok", {})

    parquet_path = _synthetic_parquet(tmp_path, n_rows=60)
    cfg = BacktestConfig(asset="SYNTH", parquet_path=parquet_path)
    engine = BacktestEngine(cfg)
    result = engine.run(_BadStrategy())
    # First 5 calls produced signals; rest raised and were skipped
    assert result.n_signals == 5
    assert result.n_trades >= 1


# ---------- determine_verdict ----------------------------------------------

def test_verdict_insufficient_samples_when_all_horizons_low_n():
    v = determine_verdict({4: 0.10, 12: 0.10, 24: 0.10},
                          {4: 5, 12: 5, 24: 5},
                          sharpe=2.0, window_days=14)
    assert v is Verdict.INSUFFICIENT_SAMPLES


def test_verdict_kill_when_short_window_and_weak_ic():
    v = determine_verdict({4: 0.01, 12: 0.02, 24: 0.03},
                          {4: 100, 12: 100, 24: 100},
                          sharpe=0.0, window_days=14)
    assert v is Verdict.KILL


def test_verdict_hold_when_short_window_and_some_ic():
    v = determine_verdict({4: 0.06, 12: 0.04, 24: 0.03},
                          {4: 100, 12: 100, 24: 100},
                          sharpe=0.0, window_days=14)
    assert v is Verdict.HOLD


def test_verdict_promote_when_long_window_clean():
    # [P166] These inputs were IC 0.06/0.07/0.08 at n=100 with no volatility
    # supplied. That combination is 0.6 SE from zero and worth ~5bps against a
    # 12bps cost bar — it is not a clean promote, it is noise that the old gate
    # could not tell apart from an edge. A genuine PROMOTE now needs an IC that
    # is both statistically real and economically positive.
    v = determine_verdict({4: 0.12, 12: 0.13, 24: 0.14},
                          {4: 400, 12: 400, 24: 400},
                          sharpe=0.8, window_days=30,
                          fwd_vol_bps_per_h={4: 400.0, 12: 400.0, 24: 400.0})
    assert v is Verdict.PROMOTE


def test_verdict_kill_when_long_window_weak():
    v = determine_verdict({4: 0.01, 12: 0.02, 24: 0.03},
                          {4: 100, 12: 100, 24: 100},
                          sharpe=1.0, window_days=30)
    assert v is Verdict.KILL


def test_verdict_hold_when_long_window_mixed():
    v = determine_verdict({4: 0.06, 12: 0.04, 24: 0.07},
                          {4: 100, 12: 100, 24: 100},
                          sharpe=0.8, window_days=30)
    assert v is Verdict.HOLD


def test_verdict_no_promote_without_sharpe():
    v = determine_verdict({4: 0.06, 12: 0.07, 24: 0.08},
                          {4: 100, 12: 100, 24: 100},
                          sharpe=0.3,    # below 0.5 cutoff
                          window_days=30)
    assert v is Verdict.HOLD


# ---------- load_shadow_ledgers (graceful) ---------------------------------

def test_load_shadow_ledgers_missing_dir(tmp_path: Path):
    out = load_shadow_ledgers(tmp_path / "does_not_exist")
    assert out == []


def test_load_shadow_ledgers_filters_by_since(tmp_path: Path):
    p = tmp_path / "microstructure_20260429.jsonl"
    old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    new_ts = datetime.now(timezone.utc).isoformat()
    p.write_text(
        json.dumps({"ts": old_ts, "asset": "BTC", "strategy": "ofi",
                    "direction": 1.0, "confidence": 0.5, "reason": "old",
                    "diagnostics": {}}) + "\n" +
        json.dumps({"ts": new_ts, "asset": "BTC", "strategy": "ofi",
                    "direction": -1.0, "confidence": 0.6, "reason": "new",
                    "diagnostics": {}}) + "\n",
        encoding="utf-8",
    )
    since = datetime.now(timezone.utc) - timedelta(days=14)
    out = load_shadow_ledgers(tmp_path, prefixes=("microstructure",), since=since)
    assert len(out) == 1
    assert out[0]["reason"] == "new"


def test_load_shadow_ledgers_skips_malformed(tmp_path: Path):
    p = tmp_path / "microstructure_20260429.jsonl"
    p.write_text(
        '{ this is not json }\n' +
        json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                    "asset": "BTC", "strategy": "ofi",
                    "direction": 1.0, "confidence": 0.5, "reason": "ok",
                    "diagnostics": {}}) + "\n",
        encoding="utf-8",
    )
    out = load_shadow_ledgers(tmp_path, prefixes=("microstructure",))
    assert len(out) == 1
    assert out[0]["reason"] == "ok"
