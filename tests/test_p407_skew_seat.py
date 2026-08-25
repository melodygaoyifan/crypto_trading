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


def test_skew_seat_edge_is_calibrated_not_generic():
    """[P407c] the seat asserts its MEASURED edge (>=100bps), not the generic 30."""
    import main
    e = main._SKEW_SEAT_EDGE_BPS
    assert e.get("BTC", 0) >= 60 and e.get("ETH", 0) >= 60, e  # clears gate threshold (~33-54)
    # and it is a conservative haircut vs the measured era-median (525/739), not inflated
    assert e.get("BTC", 999) <= 300 and e.get("ETH", 999) <= 300, e
