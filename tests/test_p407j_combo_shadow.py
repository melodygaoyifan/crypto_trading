"""[P407j] skew+ETF combo shadow: agree-gate logic + observation-only ledger."""
import json
from defense.skew_etf_combo_shadow import combo_directions, SkewEtfComboShadow, SHADOW_STRATEGY_NAMES


def test_agree_gate_logic():
    # both fresh + agree long -> agree fires long
    d = combo_directions(1.0, True, 1.0, True)
    assert d == {"skewetf_skew": 1.0, "skewetf_etf": 1.0, "skewetf_agree": 1.0}
    # disagree -> agree flat, solos unchanged
    d = combo_directions(1.0, True, -1.0, True)
    assert d["skewetf_skew"] == 1.0 and d["skewetf_etf"] == -1.0 and d["skewetf_agree"] == 0.0
    # skew not fresh -> skew leg flat, agree flat
    d = combo_directions(1.0, False, 1.0, True)
    assert d["skewetf_skew"] == 0.0 and d["skewetf_agree"] == 0.0
    # both flat -> agree flat (no saturated claim)
    assert combo_directions(0.0, True, 0.0, True)["skewetf_agree"] == 0.0


def test_writer_emits_three_strategies_confidence_is_abs(tmp_path):
    sh = SkewEtfComboShadow(data_dir=str(tmp_path))
    sh.record_tick("BTC", 1.0, True, -1.0, True)  # disagree
    rows = [json.loads(l) for l in (tmp_path / "strategy_shadow" / "skewetf_BTC.jsonl").read_text().splitlines()]
    assert {r["strategy"] for r in rows} == set(SHADOW_STRATEGY_NAMES)
    for r in rows:  # confidence == |direction| (flat -> 0, never saturated)
        assert r["confidence"] == abs(r["direction"])
    agree = next(r for r in rows if r["strategy"] == "skewetf_agree")
    assert agree["direction"] == 0.0 and agree["confidence"] == 0.0
