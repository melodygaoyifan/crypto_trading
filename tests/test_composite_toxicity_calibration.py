from execution.composite_toxicity import (
    CompositeToxicityFilter,
    get_composite_toxicity_filter,
    reset_composite_toxicity_filter,
)


def test_composite_toxicity_uses_asset_side_specific_thresholds():
    filt = CompositeToxicityFilter()

    result_long = filt.compute(
        adverse_selection=0.62,
        vpin=0.62,
        flow_imbalance=0.62,
        price_reversal=0.62,
        asset="BTC",
        side="long",
    )
    result_short = filt.compute(
        adverse_selection=0.62,
        vpin=0.62,
        flow_imbalance=0.62,
        price_reversal=0.62,
        asset="BTC",
        side="short",
    )

    assert result_long.profile_key == "BTC:LONG"
    assert result_short.profile_key == "BTC:SHORT"
    assert result_long.warn_threshold == 0.60
    assert result_short.warn_threshold == 0.66
    assert result_long.should_warn is True
    assert result_short.should_warn is False


def test_getter_returns_clean_filter_by_default():
    reset_composite_toxicity_filter()
    filt1 = get_composite_toxicity_filter()
    filt1.configure_profile(
        asset="BTC",
        side="short",
        weights={
            "adverse_selection": 0.40,
            "vpin": 0.20,
            "flow_imbalance": 0.15,
            "price_reversal": 0.25,
        },
        warn_threshold=0.11,
        veto_threshold=0.22,
    )

    filt2 = get_composite_toxicity_filter()
    result = filt2.compute(
        adverse_selection=0.62,
        vpin=0.62,
        flow_imbalance=0.62,
        price_reversal=0.62,
        asset="BTC",
        side="short",
    )

    assert filt1 is not filt2
    assert result.warn_threshold == 0.66
    assert result.veto_threshold == 0.88
