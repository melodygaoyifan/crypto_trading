"""[P409] The held BTC ridge forward shadow — the trained forecaster made to
fit the flat fee via the hold-longer lever (deadband on the trailing-z).

These pin the properties that make it a HONEST shadow: deterministic forward
pass, the deadband-HOLD (the "hold longer" mechanism), warmup honesty, the
P301 persistence that keeps the z warm across restart, and P224 (a held flat
contributes zero confidence, never a saturated 1.0)."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from defense.ridge_shadow import RidgeShadow, SHADOW_STRATEGY_NAMES


def _model():
    # a tiny self-contained model payload (no dependence on the exported BTC.json)
    return {
        "asset": "BTC", "feature_names": ["a", "b"],
        "scaler_mean": [0.0, 0.0], "scaler_scale": [1.0, 1.0],
        "coef": [1.0, 0.0], "intercept": 0.0,
        "deadband": 1.0, "z_window": 20, "z_min": 5,
    }


def _rs(tmp):
    rs = RidgeShadow.__new__(RidgeShadow)   # no config dir dependency
    from pathlib import Path
    rs._dir = Path(tmp) / "strategy_shadow"; rs._dir.mkdir(parents=True, exist_ok=True)
    rs._state_path = Path(tmp) / "ridgeshadow_state.json"
    rs._models = {"BTC": _model()}
    rs._state = {}; rs._transient = {}; rs._warned = {}; rs._last_records = {}
    return rs


def _rows(tmp):
    p = os.path.join(tmp, "strategy_shadow", "ridgeshadow_BTC.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def test_forward_is_deterministic_linear():
    rs = _rs(tempfile.mkdtemp())
    m = rs._models["BTC"]
    # coef=[1,0], intercept=0, unit scaler -> pred == first feature
    assert rs.forward(m, [3.0, 99.0]) == pytest.approx(3.0)
    assert rs.forward(m, [3.0, 99.0]) == rs.forward(m, [3.0, 99.0])  # no seed


def test_warmup_rows_are_transient_until_z_min():
    tmp = tempfile.mkdtemp(); rs = _rs(tmp)
    for i in range(4):   # z_min = 5
        rs.tick({"BTC": {"a": float(i), "b": 0.0}}, {"BTC": {"a": True, "b": True}})
    rows = _rows(tmp)
    assert len(rows) == 4 and all(r["warmup_transient"] for r in rows)
    assert all(r["z"] is None for r in rows)


def test_deadband_hold_is_the_hold_longer_lever():
    """z>band -> +1, z<-band -> -1, INSIDE the band -> HOLD the prior position.
    The hold (not a flip to flat) is exactly what amortizes the flat fee."""
    tmp = tempfile.mkdtemp(); rs = _rs(tmp)
    # fill the buffer with a flat baseline so the z of an outlier is large
    for _ in range(6):
        rs.tick({"BTC": {"a": 0.0, "b": 0.0}}, {"BTC": {"a": True, "b": True}})
    rs.tick({"BTC": {"a": 100.0, "b": 0.0}}, {"BTC": {"a": True, "b": True}})  # huge +z
    assert rs._state["BTC"]["cur"] == 1.0
    # a mild reading INSIDE the band must HOLD +1, not drop to flat
    rs.tick({"BTC": {"a": 0.5, "b": 0.0}}, {"BTC": {"a": True, "b": True}})
    assert rs._state["BTC"]["cur"] == 1.0, "deadband must HOLD, not flip to flat"


def test_coverage_gap_records_flat_not_error():
    tmp = tempfile.mkdtemp(); rs = _rs(tmp)
    rs.tick({"BTC": {"a": 1.0}}, {"BTC": {"a": True}})   # 'b' missing
    r = _rows(tmp)[-1]
    assert r["direction"] == 0.0 and r["coverage_note"].startswith("missing:")


def test_confidence_is_abs_direction_never_saturated_flat():
    tmp = tempfile.mkdtemp(); rs = _rs(tmp)
    for i in range(6):
        rs.tick({"BTC": {"a": float(i), "b": 0.0}}, {"BTC": {"a": True, "b": True}})
    for r in _rows(tmp):
        assert r["confidence"] == abs(r["direction"])   # P224


def test_persistence_keeps_the_z_warm_across_restart():
    """P301: the prediction buffer survives a restart, so the trailing z is
    correct immediately instead of re-warming ~17 days."""
    tmp = tempfile.mkdtemp(); rs = _rs(tmp)
    for i in range(8):   # past z_min
        rs.tick({"BTC": {"a": float(i), "b": 0.0}}, {"BTC": {"a": True, "b": True}})
    assert not rs._transient["BTC"]
    rs2 = _rs(tmp)
    rs2._models = {"BTC": _model()}
    rs2._restore_state()
    assert len(rs2._state["BTC"]["buffer"]) == 8
    assert rs2._transient["BTC"] is False   # not re-warming
    rs2.tick({"BTC": {"a": 9.0, "b": 0.0}}, {"BTC": {"a": True, "b": True}})
    assert _rows(tmp)[-1]["warmup_transient"] is False


def test_last_direction_is_none_during_warmup_and_coverage_gap():
    tmp = tempfile.mkdtemp(); rs = _rs(tmp)
    rs.tick({"BTC": {"a": 1.0, "b": 0.0}}, {"BTC": {"a": True, "b": True}})   # warmup
    assert rs.last_direction("BTC") is None   # P2: warming flat is NOT an opinion
    rs.tick({"BTC": {"a": 1.0}}, {"BTC": {"a": True}})                        # cov gap
    assert rs.last_direction("BTC") is None


def test_strategy_name_single_source():
    assert SHADOW_STRATEGY_NAMES == frozenset({"ridgeshadow"})


def test_exported_btc_payload_has_required_keys_if_present():
    """If the export was run, its payload must carry what the harness needs."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "configs" / "ridgeshadow" / "BTC.json"
    if not p.exists():
        pytest.skip("BTC.json not exported (operator-local data)")
    m = json.loads(p.read_text(encoding="utf-8"))
    for k in ("feature_names", "scaler_mean", "scaler_scale", "coef",
              "intercept", "deadband", "z_window", "z_min"):
        assert k in m, f"export missing {k}"
    assert len(m["feature_names"]) == len(m["coef"]) == len(m["scaler_mean"])
    assert m["deadband"] == 1.0 and m["asset"] == "BTC"
