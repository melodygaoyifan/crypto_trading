from main import HMATSProductionRunner


def test_structure_mode_override_keeps_long_opportunity_only():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    cfg = {
        "apply_in_system_modes": ["OPPORTUNITY"],
        "apply_in_system_modes_long": ["OPPORTUNITY"],
        "apply_in_system_modes_short": ["NORMAL", "OPPORTUNITY"],
    }

    assert runner._is_structure_mode_allowed(cfg, "OPPORTUNITY", is_long=True) is True
    assert runner._is_structure_mode_allowed(cfg, "NORMAL", is_long=True) is False


def test_structure_mode_override_opens_short_normal_only():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    cfg = {
        "apply_in_system_modes": ["OPPORTUNITY"],
        "apply_in_system_modes_long": ["OPPORTUNITY"],
        "apply_in_system_modes_short": ["NORMAL", "OPPORTUNITY"],
    }

    assert runner._is_structure_mode_allowed(cfg, "NORMAL", is_long=False) is True
    assert runner._is_structure_mode_allowed(cfg, "OPPORTUNITY", is_long=False) is True


def test_structure_mode_override_short_falls_back_to_base_when_side_missing():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    cfg = {
        "apply_in_system_modes": ["OPPORTUNITY"],
    }

    assert runner._is_structure_mode_allowed(cfg, "OPPORTUNITY", is_long=False) is True
    assert runner._is_structure_mode_allowed(cfg, "NORMAL", is_long=False) is False
