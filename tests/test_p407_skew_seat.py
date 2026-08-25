"""[P407] Skew direction seat: pure decision, feed contrarian mapping, fail-safe."""
import os, json, time, datetime as dt, importlib.util, pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

def _load_main_symbol(name):
    # import skew_seat_decision without importing the whole heavy runner tree
    import main
    return getattr(main, name)

# ---- pure decision fn ----
def test_skew_seat_decision_truth_table():
    dec = _load_main_symbol("skew_seat_decision")
    # not fresh -> None (fail-safe, P2)
    assert dec("BTC", 1.0, False, ["BTC", "ETH"]) is None
    # fresh directional for a decide-asset -> takes seat
    assert dec("BTC", 1.0, True, ["BTC", "ETH"]) == ("skew_contra", 1.0)
    assert dec("ETH", -1.0, True, ["BTC", "ETH"]) == ("skew_contra", -1.0)
    # asset not in decide set -> None
    assert dec("SOL", 1.0, True, ["BTC", "ETH"]) is None
    # fresh but flat -> None (no seat on a zero direction)
    assert dec("BTC", 0.0, True, ["BTC", "ETH"]) is None
    # empty decide set -> None
    assert dec("BTC", 1.0, True, []) is None

# ---- feed contrarian mapping + fail-safe ----
def _sig(monkeypatch, rows):
    from defense.skew_flow_signal import SkewFlowSignal
    s = SkewFlowSignal(data_dir=str(ROOT / "tests" / "_tmp_p407"))
    s._key = "test"  # non-empty so _fetch_trailing is attempted
    monkeypatch.setattr(s, "_fetch_trailing", lambda a: rows)
    return s

def _recent_rows(vals):
    now = dt.datetime.now(dt.timezone.utc)
    out = []
    for i, v in enumerate(vals):
        d = (now - dt.timedelta(days=len(vals) - 1 - i)).isoformat()
        out.append({"tenor": 30, "skew_25d": v, "date": d})
    return out

def test_feed_contrarian_long_on_fear(monkeypatch):
    # flat trailing then a sharp DROP (extra fear) -> z very negative -> LONG
    vals = [(-1.0) ** i for i in range(30)] + [-8.0]  # baseline std~1, then sharp drop
    s = _sig(monkeypatch, _recent_rows(vals))
    d, fresh = s.seat_direction("BTC")
    assert fresh is True and d == 1.0

def test_feed_contrarian_short_on_greed(monkeypatch):
    vals = [(-1.0) ** i for i in range(30)] + [8.0]  # baseline std~1, then spike UP
    s = _sig(monkeypatch, _recent_rows(vals))
    d, fresh = s.seat_direction("BTC")
    assert fresh is True and d == -1.0

def test_feed_deadband_holds(monkeypatch):
    vals = [(-1.0) ** i for i in range(30)] + [0.1]  # tiny move, inside deadband -> hold
    s = _sig(monkeypatch, _recent_rows(vals))
    d, fresh = s.seat_direction("BTC")
    assert fresh is True and d == 0.0

def test_feed_warmup_not_fresh(monkeypatch):
    s = _sig(monkeypatch, _recent_rows([0.0, -8.0]))  # <15 obs
    assert s.seat_direction("BTC") == (0.0, False)

def test_feed_stale_not_fresh(monkeypatch):
    now = dt.datetime.now(dt.timezone.utc)
    rows = [{"tenor": 30, "skew_25d": float(-i),
             "date": (now - dt.timedelta(days=40 + (30 - i))).isoformat()}
            for i in range(31)]  # latest ~40d old -> stale
    s = _sig(monkeypatch, rows)
    d, fresh = s.seat_direction("BTC")
    assert fresh is False

def test_feed_no_key_not_fresh():
    from defense.skew_flow_signal import SkewFlowSignal
    s = SkewFlowSignal(data_dir=str(ROOT / "tests" / "_tmp_p407"))
    s._key = ""  # no key
    assert s.seat_direction("BTC") == (0.0, False)

# ---- config parse ----
def test_config_parses_skew_keys(tmp_path):
    from main import ProductionConfig
    cfg = {"skew_seat_mode": "enforce", "skew_seat_assets": ["BTC", "ETH"]}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    c = ProductionConfig.from_file(p)
    assert c.skew_seat_mode == "enforce"
    assert c.skew_seat_assets == ["BTC", "ETH"]

def test_config_default_off():
    from main import ProductionConfig
    c = ProductionConfig()
    assert c.skew_seat_mode == "off"


def test_skew_edge_is_measured_era_median_not_hand_picked():
    """[P407e] the seat asserts the MEASURED era-median via the P320 framework,
    and that constant must equal the median of its own per-era table (P326: no
    silent drift), NOT a number hand-picked to clear the gate (the P231
    anti-pattern the framework exists to remove)."""
    from core.seat_alpha import (
        calibrated_seat_alpha, skew_contra_alpha_bps,
        SKEW_CONTRA_ALPHA_BY_ERA, SKEW_CONTRA_ALPHA_BPS_PER_ROUND_TRIP, _median,
    )
    # the asserted constant is the era-median of the table it summarises
    for a in ("BTC", "ETH"):
        eras = list(SKEW_CONTRA_ALPHA_BY_ERA[a].values())
        assert abs(SKEW_CONTRA_ALPHA_BPS_PER_ROUND_TRIP[a] - _median(eras)) < 0.6, a
    # dispatch resolves skew to the measured value with era-median provenance
    btc, prov = calibrated_seat_alpha("BTC", "skew_contra", 30.0)
    assert btc > 400 and "era_median" in prov, (btc, prov)  # clears gate BY MEASUREMENT
    eth, _ = calibrated_seat_alpha("ETH", "skew_contra", 30.0)
    assert eth > 600, eth
    # unknown asset asserts nothing (cannot trade on a non-measurement)
    assert skew_contra_alpha_bps("DOGE") == (0.0, "no_calibration_for:DOGE")


def test_uncalibrated_seats_keep_generic_fallback():
    """[P407e] whale/etf/mlp are deliberately uncalibrated (whale = noise, P324)
    and must keep the generic 30 fallback -- calibrating skew must not re-price
    a seat whose edge was never measured."""
    from core.seat_alpha import calibrated_seat_alpha
    for seat in ("whale", "etf_flow", "mlpshadow"):
        v, prov = calibrated_seat_alpha("BTC", seat, 30.0)
        assert v == 30.0 and "uncalibrated" in prov, (seat, v, prov)
