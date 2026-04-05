from defense.production_reliability import (
    RiskVetoClassifier,
    SoftVetoCondition,
    VetoType,
)


def test_normal_mode_correlation_soft_veto_caps_instead_of_blocking():
    classifier = RiskVetoClassifier()

    result = classifier.classify(
        mode="NORMAL",
        correlation=0.90,
    )

    assert result.veto_type == VetoType.SOFT
    assert result.allows_trade is True
    assert result.exposure_cap == 0.70
    assert result.soft_conditions == [SoftVetoCondition.CORRELATION_ELEVATED]


def test_normal_mode_non_correlation_soft_veto_still_blocks():
    classifier = RiskVetoClassifier()

    result = classifier.classify(
        mode="NORMAL",
        correlation=0.90,
        dvol_zscore=3.5,
    )

    assert result.veto_type == VetoType.SOFT
    assert result.allows_trade is False
    assert SoftVetoCondition.CORRELATION_ELEVATED in result.soft_conditions


def test_normal_mode_correlation_and_weekend_soft_veto_caps_instead_of_blocking():
    classifier = RiskVetoClassifier()

    result = classifier.classify(
        mode="NORMAL",
        correlation=0.90,
        is_weekend=True,
    )

    assert result.veto_type == VetoType.SOFT
    assert result.allows_trade is True
    assert result.exposure_cap == 0.40
    assert set(result.soft_conditions) == {
        SoftVetoCondition.CORRELATION_ELEVATED,
        SoftVetoCondition.WEEKEND_LIQUIDITY,
    }
