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


def test_normal_mode_non_correlation_soft_veto_tightens_the_cap():
    """A second SOFT condition tightens the cap; it does not block.

    [P165] This test was `..._still_blocks` and asserted `allows_trade is False`
    — the behaviour `defense/production_reliability.py:608-618` deliberately
    removed ("In NORMAL mode, SOFT caps exposure (was: blocks entirely) …
    Hard-blocking kills trade frequency and biases toward no-action"). SOFT no
    longer blocks in NORMAL at all, so the old assertion contradicts the design.

    What still matters is that adding ELEVATED_DVOL (cap 0.50) on top of
    CORRELATION_ELEVATED (cap 0.70) resolves to the STRICTER of the two — a
    second risk condition must never loosen the result.
    """
    classifier = RiskVetoClassifier()

    corr_only = classifier.classify(mode="NORMAL", correlation=0.90)
    result = classifier.classify(
        mode="NORMAL",
        correlation=0.90,
        dvol_zscore=3.5,
    )

    assert result.veto_type == VetoType.SOFT
    assert result.allows_trade is True
    assert SoftVetoCondition.CORRELATION_ELEVATED in result.soft_conditions
    assert SoftVetoCondition.ELEVATED_DVOL in result.soft_conditions
    assert result.exposure_cap == 0.50
    assert result.exposure_cap < corr_only.exposure_cap, (
        "adding a second SOFT condition must not loosen the exposure cap"
    )


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
