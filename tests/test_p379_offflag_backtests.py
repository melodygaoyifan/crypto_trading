"""[P379] Pin the out-of-sample verdicts so nobody arms an overfit/dead off-flag.
The operator asked to backtest the off flags to turn them on; the answer is no —
every backtestable one fails out-of-sample. These pin the two run this turn.
"""
import json
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
ADJ = REPO / "training" / "reports" / "regimebook_adj_validation_p379.json"
MICRO = REPO / "training" / "reports" / "edge_probe_hf_p375.json"  # 60m flow (companion)


@pytest.mark.skipif(not ADJ.exists(), reason="report is operator-local")
def test_regimebook_adj_is_overfit_out_of_sample():
    d = json.loads(ADJ.read_text(encoding="utf-8"))
    # in-design it won all 3; out-of-sample it must NOT beat raw net on >=2/3
    # (arming it would trade an overfit signal). A future PASS is a real
    # finding to act on deliberately (P141), not a relaxed bar.
    assert d["net_beats"] < 2, d
    assert d["verdict"].startswith("OVERFIT")
    # the churn-reduction half is real and generalizes (turnover cut) — that is
    # WHY it looked good in-design; it is not enough to overcome worse timing.
    assert d["turnover_cut"] == 3


def test_bar_requires_net_and_turnover_both():
    # pin the verdict rule: arm only if net beats >=2/3 AND turnover cut on all 3
    def verdict(net_beats, turn_cut):
        return "GENERALIZES" if (net_beats >= 2 and turn_cut == 3) else "OVERFIT"
    assert verdict(0, 3) == "OVERFIT"   # the real result
    assert verdict(3, 2) == "OVERFIT"   # turnover must hold on all 3
    assert verdict(2, 3) == "GENERALIZES"
