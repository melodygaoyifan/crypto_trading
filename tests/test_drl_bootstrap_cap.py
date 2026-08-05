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


def test_normalize_drl_paper_bootstrap_passes_known_levels_and_rejects_junk():
    """[P165] Was `..._caps_active_to_exit_only`, asserting ACTIVE→EXIT_ONLY.

    `main.py:18589` was deliberately changed ("[UTIL-5] ACTIVE now allowed —
    auto-demotion safety (5 consec losses / 15% DD) still enforced"), so the cap
    the test named no longer exists. Rewritten to the contract the function
    actually has: it is a whitelist normalizer, and the property still worth
    holding is that an unrecognized value returns "" rather than being passed
    through to `_promotion_gate.promote()`.
    """
    runner = _make_runner()

    assert runner._normalize_drl_paper_bootstrap_authority("ACTIVE") == "ACTIVE"
    assert runner._normalize_drl_paper_bootstrap_authority("EXIT_ONLY") == "EXIT_ONLY"
    assert runner._normalize_drl_paper_bootstrap_authority("shadow") == "SHADOW"
    assert runner._normalize_drl_paper_bootstrap_authority("DISABLED") == "DISABLED"
    # Junk must NOT normalize to anything promotable — main.py:5039 gates the
    # bootstrap on this being truthy.
    for junk in ("bogus", "", None, "ACTIVE_", 3):
        assert runner._normalize_drl_paper_bootstrap_authority(junk) == "", (
            f"{junk!r} must not normalize to a promotable authority level"
        )


def test_drl_runtime_snapshot_keeps_exit_only_trade_impact():
    runner = _make_runner("EXIT_ONLY")

    snapshot = runner._get_drl_runtime_snapshot()

    assert snapshot["promotion_level"] == "EXIT_ONLY"
    assert snapshot["gate_level"] == "EXIT_ONLY"
    assert snapshot["trade_impact"] == "EXIT_ONLY"
    assert snapshot["execution_authority"] == "ON"
    assert snapshot["bootstrap_applied"] is True
