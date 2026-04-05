from types import SimpleNamespace

from main import HMATSProductionRunner


class _PromotionGateStub:
    def __init__(self, level: str):
        self._level = level

    def get_authority_level(self):
        return self._level


def _make_runner(level: str = "EXIT_ONLY"):
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner._promotion_gate = _PromotionGateStub(level)
    runner._drl_authority_level = level
    runner._drl_models_ready = 3
    runner._drl_ensembles = {
        "BTC": SimpleNamespace(tqc_available=True),
        "ETH": SimpleNamespace(tqc_available=True),
    }
    runner._drl_inference_mode = "SHADOW"
    runner._drl_bootstrap_applied = True
    return runner


def test_normalize_drl_paper_bootstrap_caps_active_to_exit_only():
    runner = _make_runner()

    assert runner._normalize_drl_paper_bootstrap_authority("ACTIVE") == "EXIT_ONLY"
    assert runner._normalize_drl_paper_bootstrap_authority("EXIT_ONLY") == "EXIT_ONLY"
    assert runner._normalize_drl_paper_bootstrap_authority("shadow") == "SHADOW"
    assert runner._normalize_drl_paper_bootstrap_authority("bogus") == ""


def test_drl_runtime_snapshot_keeps_exit_only_trade_impact():
    runner = _make_runner("EXIT_ONLY")

    snapshot = runner._get_drl_runtime_snapshot()

    assert snapshot["promotion_level"] == "EXIT_ONLY"
    assert snapshot["gate_level"] == "EXIT_ONLY"
    assert snapshot["trade_impact"] == "EXIT_ONLY"
    assert snapshot["execution_authority"] == "ON"
    assert snapshot["bootstrap_applied"] is True
