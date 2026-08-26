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
    rows = [json.loads(l) for l in (tmp_path / "strategy_shadow" / "skewetf_BTC.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["strategy"] for r in rows} == set(SHADOW_STRATEGY_NAMES)
    for r in rows:  # confidence == |direction| (flat -> 0, never saturated)
        assert r["confidence"] == abs(r["direction"])
    agree = next(r for r in rows if r["strategy"] == "skewetf_agree")
    assert agree["direction"] == 0.0 and agree["confidence"] == 0.0


def test_combiner_is_equal_weight_never_learned():
    """[research 2026-08-26] The forecast-combination puzzle + DeMiguel 1/N:
    equal-weight beats an estimated/learned combiner OOS unless you have
    ~3000 months of data (we have ~24). A learned combiner over a handful of
    signals is the documented failure mode. This pins the combiner as pure
    equal-weight / agree-gate so a future learned weighting fails loudly.
    """
    # (a) BEHAVIOUR: an agreeing signal passes through UNSCALED — no coefficient
    # is applied. If a learned weight w!=1 were introduced, agree would be w*dir.
    for d in (1.0, -1.0):
        out = combo_directions(d, True, d, True)
        assert out["skewetf_agree"] == d, "agree leg must be the raw direction, not a weighted one"
        assert out["skewetf_skew"] == d and out["skewetf_etf"] == d
    # a strong-vs-weak disagreement can never be 'averaged' into a nonzero bet
    assert combo_directions(1.0, True, -1.0, True)["skewetf_agree"] == 0.0

    # (b) SOURCE: the module must not import an ML/fitting stack or carry learned
    # coefficients — the mechanisms by which a learned combiner would appear.
    import defense.skew_etf_combo_shadow as m
    src = __import__("inspect").getsource(m)
    for banned in ("sklearn", "from scipy.optimize", ".fit(", "coef_", "np.dot",
                   "LinearRegression", "Ridge(", "np.linalg.lstsq", "learned_weight"):
        assert banned not in src, f"combiner must stay equal-weight/agree-gate, not learned: found {banned!r}"
