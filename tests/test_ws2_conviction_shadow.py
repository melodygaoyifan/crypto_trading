"""[WS2] Conviction-sizing forward shadow: the agree-tier rule, fail-soft
recording, observation-only, and the PnL reader's forward math.

The shadow sizes UP when trend AND the live skew agree, DOWN when skew
disagrees. It must never claim a position beyond a ledger row (Iron Law 7),
never fabricate a skew opinion when the read fails (P2), and its reader must
compute the forward PnL increment correctly."""
import importlib.util
from pathlib import Path

import pytest

from defense.conviction_sizing_shadow import (
    ConvictionSizingShadow, conviction_mult, CAP, DERISK)

REPO = Path(__file__).resolve().parent.parent


def _shadow(skew, tmp_path):
    s = ConvictionSizingShadow.__new__(ConvictionSizingShadow)
    s._skew = skew
    s._dir = tmp_path / "conviction_shadow"
    s._dir.mkdir(parents=True, exist_ok=True)
    s._warned = set()
    return s


# ---- the agree-tier rule ----
def test_conviction_mult_truth_table():
    assert conviction_mult(False, +1.0) == 0.0          # no trend -> flat
    assert conviction_mult(True, +1.0) == CAP           # agree -> size up
    assert conviction_mult(True, -1.0) == DERISK        # disagree -> de-risk
    assert conviction_mult(True, 0.0) == 1.0            # skew neutral -> base
    # a flat trend is flat regardless of skew (the de-risk can't create a short)
    assert conviction_mult(False, -1.0) == 0.0


def test_never_shorts_or_exceeds_cap():
    for tl in (True, False):
        for sd in (-1.0, 0.0, 1.0):
            m = conviction_mult(tl, sd)
            assert 0.0 <= m <= CAP, "sizing must be long/flat, bounded by cap"


# ---- fail-soft recording (no network) ----
def test_a_fetch_failure_skips_the_tick(tmp_path):
    s = _shadow(skew=None, tmp_path=tmp_path)
    s._fetch_closes_4h = lambda a: None    # simulate OHLC failure
    assert s.record_tick("BTC") is None    # skipped, not crashed


def test_a_stale_skew_reads_as_neutral_not_a_fabricated_opinion(tmp_path):
    # skew read raises -> treated neutral (base 1x), never a fabricated dir (P2)
    class _Boom:
        def seat_direction(self, a): raise RuntimeError("laevitas down")
    s = _shadow(skew=_Boom(), tmp_path=tmp_path)
    s._fetch_closes_4h = lambda a: [100.0] * 210   # flat series, past warmup
    rec = s.record_tick("BTC")
    assert rec is not None
    assert rec["skew_fresh"] is False
    # flat price -> trend not long -> conv 0; the point is it did not crash
    assert rec["conv_pos"] == 0.0


def test_agreeing_signal_sizes_up_in_an_uptrend(tmp_path):
    class _Skew:
        def seat_direction(self, a): return (1.0, True)   # skew says long
    s = _shadow(skew=_Skew(), tmp_path=tmp_path)
    # rising series so close > SMA200 -> trend long
    s._fetch_closes_4h = lambda a: [float(i) for i in range(1, 260)]
    rec = s.record_tick("ETH")
    assert rec["trend_long"] is True and rec["skew_fresh"] is True
    assert rec["conv_pos"] == CAP          # trend AND skew agree -> size up
    assert rec["base_pos"] == 1.0


def test_disagreeing_skew_de_risks_in_an_uptrend(tmp_path):
    class _Skew:
        def seat_direction(self, a): return (-1.0, True)  # skew says euphoria
    s = _shadow(skew=_Skew(), tmp_path=tmp_path)
    s._fetch_closes_4h = lambda a: [float(i) for i in range(1, 260)]
    rec = s.record_tick("ETH")
    assert rec["conv_pos"] == DERISK       # de-risk, the "sell before top" leg


def test_observation_only_no_position_concept(tmp_path):
    s = _shadow(skew=None, tmp_path=tmp_path)
    assert not hasattr(s, "target_exposure")
    assert not hasattr(s, "direction")
    # the record is a ledger row, not an order/intent
    s._fetch_closes_4h = lambda a: [float(i) for i in range(1, 260)]
    rec = s.record_tick("BTC")
    assert set(rec) >= {"base_pos", "conv_pos", "close", "skew_dir"}
    assert "order" not in rec and "intent" not in rec


# ---- the PnL reader's forward math ----
def _load_reader():
    spec = importlib.util.spec_from_file_location(
        "conv_review", REPO / "scripts" / "conviction_sizing_review.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_reader_book_pnl_scales_with_size_on_a_rising_series():
    r = _load_reader()
    close = [100.0, 110.0, 121.0]        # +10% each step
    base_net, _ = r._book(close, [1.0, 1.0, 1.0], cost_bps=0.0)
    conv_net, _ = r._book(close, [2.0, 2.0, 2.0], cost_bps=0.0)
    assert conv_net > base_net
    # at zero fee a 2x book earns exactly 2x the 1x book
    assert abs(conv_net - 2 * base_net) < 1e-9


def test_reader_maxdd_is_negative_on_a_drawdown():
    r = _load_reader()
    close = [100.0, 90.0, 95.0]          # a drop then partial recovery
    _, dd = r._book(close, [1.0, 1.0, 1.0], cost_bps=0.0)
    assert dd < 0.0


def test_reader_refuses_thin_ledger(tmp_path, capsys):
    r = _load_reader()
    d = tmp_path / "conviction_shadow"; d.mkdir(parents=True)
    import json
    (d / "convsize_BTC.jsonl").write_text(
        "\n".join(json.dumps({"close": 100.0, "base_pos": 1.0, "conv_pos": 2.0,
                              "reason": "ok"}) for _ in range(5)),
        encoding="utf-8")
    rc = r.review(str(tmp_path))
    assert rc == 2   # thin -> no verdict (P199/P348)
