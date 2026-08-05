"""[P156] FastRiskTick must not act on a stale 4H anchor.

Every trigger in `_evaluate` compares a live reading against a reference
captured by `set_4h_anchor()`. That call is the LAST statement of the 4H
decision path (main.py:10166), so every early return before it — notably the
P0 ABORT at main.py:7998 — silently leaves the baselines frozen while this
evaluator keeps running every 30s against them.

A depth baseline anchored during a healthy orderbook then makes ordinary depth
look like an 80%+ collapse indefinitely, firing REDUCE_50 forever and
ratcheting exposure toward zero. That is the same failure shape as P155's
`_last_quant_directions` high-water mark, and a candidate explanation for the
observed `[FastRiskTick][LIVE] BTC: REDUCE_50 - depth_drop=82%(3x)`.

Fail-SAFE direction matters: this evaluator can only REDUCE or EXIT, so
refusing to act is always the conservative side of the error.
"""

import logging

import pytest

from execution.fast_risk_tick import FastRiskAction, FastRiskTick


HEALTHY_DEPTH = 2_000_000.0
COLLAPSED_DEPTH = 360_000.0  # 82% below HEALTHY_DEPTH, above MIN_VALID_DEPTH_USD


def _tick(shadow=True):
    t = FastRiskTick(shadow_mode=shadow)
    t.set_4h_anchor("BTC", price=100_000.0, volatility=0.01, depth=HEALTHY_DEPTH)
    return t


def _md(price=100_000.0, depth=COLLAPSED_DEPTH):
    return {
        "current_price": price,
        "orderbook_depth_1pct_usd": depth,
        "volatility_30m": 0.01,
        "data_valid": True,
        "orderbook_stale": False,
    }


def _age(t, asset, seconds):
    """Rewind the anchor timestamp to simulate elapsed time."""
    t._anchor_set_at[asset] -= seconds


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------

def test_fresh_anchor_still_detects_a_real_depth_collapse():
    """The guard must not disarm the trigger it protects."""
    t = _tick()
    for _ in range(FastRiskTick.DEPTH_DROP_CONFIRM_STREAK):
        r = t.evaluate("BTC", _md())
    assert r.action == FastRiskAction.REDUCE_50
    assert "depth_drop" in r.reason


def test_stale_anchor_suppresses_the_trigger():
    t = _tick()
    _age(t, "BTC", FastRiskTick.ANCHOR_MAX_AGE_SEC + 60)
    for _ in range(FastRiskTick.DEPTH_DROP_CONFIRM_STREAK + 2):
        r = t.evaluate("BTC", _md())
    assert r.action == FastRiskAction.HOLD
    assert r.reason == "anchor_stale"


def test_stale_anchor_also_suppresses_the_price_move_exit():
    """EXIT_ONLY bypasses the cooldown, so it must not bypass this too."""
    t = _tick()
    _age(t, "BTC", FastRiskTick.ANCHOR_MAX_AGE_SEC + 60)
    r = t.evaluate("BTC", _md(price=130_000.0))  # +30%, far past the 3% floor
    assert r.action == FastRiskAction.HOLD
    assert r.reason == "anchor_stale"


def test_anchor_just_inside_the_limit_still_acts():
    t = _tick()
    _age(t, "BTC", FastRiskTick.ANCHOR_MAX_AGE_SEC - 600)
    for _ in range(FastRiskTick.DEPTH_DROP_CONFIRM_STREAK):
        r = t.evaluate("BTC", _md())
    assert r.action == FastRiskAction.REDUCE_50


def test_one_missed_4h_tick_is_tolerated():
    """A normally-late tick must not trip the guard — only a repeated miss."""
    t = _tick()
    _age(t, "BTC", 4 * 3600 + 300)
    for _ in range(FastRiskTick.DEPTH_DROP_CONFIRM_STREAK):
        r = t.evaluate("BTC", _md())
    assert r.action == FastRiskAction.REDUCE_50


def test_refreshing_the_anchor_rearms_the_evaluator():
    t = _tick()
    _age(t, "BTC", FastRiskTick.ANCHOR_MAX_AGE_SEC + 60)
    assert t.evaluate("BTC", _md()).reason == "anchor_stale"

    t.set_4h_anchor("BTC", price=100_000.0, volatility=0.01, depth=HEALTHY_DEPTH)
    for _ in range(FastRiskTick.DEPTH_DROP_CONFIRM_STREAK):
        r = t.evaluate("BTC", _md())
    assert r.action == FastRiskAction.REDUCE_50


def test_stale_anchor_resets_the_confirm_streak():
    """Otherwise a streak accumulated while stale would fire the instant the
    anchor refreshed, on evidence gathered against the wrong reference."""
    t = _tick()
    _age(t, "BTC", FastRiskTick.ANCHOR_MAX_AGE_SEC + 60)
    for _ in range(5):
        t.evaluate("BTC", _md())
    assert t._depth_drop_streak.get("BTC", 0) == 0


def test_staleness_is_logged_but_rate_limited(caplog):
    t = _tick()
    _age(t, "BTC", FastRiskTick.ANCHOR_MAX_AGE_SEC + 60)
    with caplog.at_level(logging.WARNING):
        for _ in range(6):
            t.evaluate("BTC", _md())
    hits = [r for r in caplog.records if "anchor is" in r.message]
    assert len(hits) == 1, "must surface, must not spam every 30s"
    assert "set_4h_anchor" in hits[0].message


def test_warmup_still_takes_precedence_over_staleness():
    """An asset that was never anchored reports warmup, not staleness."""
    t = FastRiskTick(shadow_mode=True)
    assert t.evaluate("ETH", _md()).reason in ("warmup_no_anchor", "no_anchor")


def test_anchor_timestamp_is_recorded_for_each_asset_independently():
    t = _tick()
    t.set_4h_anchor("ETH", price=3_000.0, volatility=0.01, depth=HEALTHY_DEPTH)
    _age(t, "BTC", FastRiskTick.ANCHOR_MAX_AGE_SEC + 60)
    assert t.evaluate("BTC", _md()).reason == "anchor_stale"
    for _ in range(FastRiskTick.DEPTH_DROP_CONFIRM_STREAK):
        r = t.evaluate("ETH", _md(price=3_000.0))
    assert r.action == FastRiskAction.REDUCE_50
