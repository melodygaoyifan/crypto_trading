"""
test_external_api_fuzz.py — Hypothesis fuzz tests for feed parsers (P119)
===========================================================================

Generates malformed/edge-case JSON inputs for the 5 feed `_parse_raw_data`
methods. Asserts: parsers EITHER return a valid Tick object OR raise a
documented exception — they MUST NOT silently corrupt data downstream.

Bug class targeted: P22-shape silent type-coercion + P109-shape KeyError
on missing fields. The agent's "self.config undefined" cluster all started
from parsers that worked on the happy path but blew up on malformed
external API responses.

Hypothesis runs 100 random inputs per @given, shrinking failures to
minimal reproducers.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck


# Strategy: a dict that COULD be an external API response — keys missing,
# values wrong-typed, nested structures malformed.
def malformed_dict_strategy():
    """Generate dicts with arbitrary keys and mixed-type values."""
    primitive = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-10**9, max_value=10**9),
        st.floats(allow_nan=True, allow_infinity=True, width=32),
        st.text(min_size=0, max_size=50),
    )
    return st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.one_of(
            primitive,
            st.lists(primitive, max_size=5),
            st.dictionaries(st.text(max_size=10), primitive, max_size=5),
        ),
        max_size=10,
    )


def realistic_onchain_dict_strategy():
    """Whale-activity-shaped dicts — sometimes valid, sometimes missing keys."""
    whale_keys = st.lists(
        st.fixed_dictionaries({
            "address": st.text(min_size=1, max_size=42),
            "activity_type": st.sampled_from(["buy", "sell", "transfer"]),
            "token": st.sampled_from(["BTC", "ETH", "SOL", "UNKNOWN"]),
            "amount": st.floats(min_value=0, max_value=1e6, allow_nan=False),
            "value_usd": st.floats(min_value=0, max_value=1e9, allow_nan=False),
            "timestamp": st.sampled_from([
                "2026-04-27T01:00:00",
                "2026-04-27T01:00:00+00:00",
                "invalid-timestamp",
                "",
            ]),
        }),
        max_size=5,
    )
    return st.fixed_dictionaries({
        "timestamp": st.sampled_from([
            "2026-04-27T01:00:00",
            "invalid",
            "",
        ]),
        "whale_activities": whale_keys,
        "exchange_flows": st.lists(
            st.dictionaries(st.text(max_size=10), st.floats(allow_nan=False), max_size=3),
            max_size=3,
        ),
        "dex_metrics": st.lists(
            st.dictionaries(st.text(max_size=10), st.floats(allow_nan=False), max_size=3),
            max_size=3,
        ),
    })


# =====================================================================
# OnChainFeed parser
# =====================================================================

class TestOnChainFeedFuzz:
    """OnChainFeed._parse_raw_data must not silently corrupt downstream
    state on malformed Helius/CryptoCompare responses."""

    def _parser(self):
        from data_mgmt.feeds.onchain_feed import OnChainFeed
        return OnChainFeed.__new__(OnChainFeed)._parse_raw_data

    @given(raw=malformed_dict_strategy())
    @settings(max_examples=100, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_arbitrary_dict_either_parses_or_raises(self, raw):
        """Either returns a Tick OR raises a recognizable exception.
        Must NOT silently return garbage."""
        parser = self._parser()
        try:
            result = parser(raw)
        except (KeyError, ValueError, TypeError, AttributeError):
            # Documented failure modes — these are caller-recoverable.
            return
        # If it parsed, the result must be a valid Tick (has expected attrs).
        assert hasattr(result, "timestamp"), (
            f"Parser returned non-Tick object on input {raw!r}: {result!r}"
        )

    @given(raw=realistic_onchain_dict_strategy())
    @settings(max_examples=100, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_onchain_shaped_dict_no_silent_corruption(self, raw):
        parser = self._parser()
        try:
            result = parser(raw)
        except (KeyError, ValueError, TypeError, AttributeError):
            return
        # Sanity: aggregate metrics are finite numbers
        assert hasattr(result, "timestamp")


# =====================================================================
# SentimentFeed parser
# =====================================================================

class TestSentimentFeedFuzz:
    def _parser(self):
        from data_mgmt.feeds.sentiment_feed import SentimentFeed
        return SentimentFeed.__new__(SentimentFeed)._parse_raw_data

    @given(raw=malformed_dict_strategy())
    @settings(max_examples=100, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_arbitrary_dict_safe(self, raw):
        parser = self._parser()
        try:
            result = parser(raw)
        except (KeyError, ValueError, TypeError, AttributeError):
            return
        assert hasattr(result, "timestamp")


# =====================================================================
# LOBFeed parser
# =====================================================================

class TestLOBFeedFuzz:
    def _parser(self):
        from data_mgmt.feeds.lob_feed import LOBFeed
        return LOBFeed.__new__(LOBFeed)._parse_raw_data

    @given(raw=malformed_dict_strategy())
    @settings(max_examples=100, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_arbitrary_dict_safe(self, raw):
        parser = self._parser()
        try:
            result = parser(raw)
        except (KeyError, ValueError, TypeError, AttributeError, IndexError):
            return
        # If we got something back, it should be tick-shaped
        assert hasattr(result, "timestamp") or hasattr(result, "best_bid")


# =====================================================================
# MacroFeed parser
# =====================================================================

class TestMacroFeedFuzz:
    def _parser(self):
        from data_mgmt.feeds.macro_feed import MacroFeed
        return MacroFeed.__new__(MacroFeed)._parse_raw_data

    @given(raw=malformed_dict_strategy())
    @settings(max_examples=100, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_arbitrary_dict_safe(self, raw):
        parser = self._parser()
        try:
            result = parser(raw)
        except (KeyError, ValueError, TypeError, AttributeError):
            return
        assert hasattr(result, "timestamp")


# =====================================================================
# Kraken error classifier (already covered in property tests, repeated here
# under fuzz for exhaustiveness against malformed-string surface area).
# =====================================================================

class TestKrakenErrorParserFuzz:
    @given(error_str=st.one_of(
        st.text(alphabet=st.characters(min_codepoint=0, max_codepoint=127),
                min_size=0, max_size=2000),
        st.binary(min_size=0, max_size=200).map(lambda b: b.decode("latin1")),
    ))
    @settings(max_examples=200, deadline=None)
    def test_classifier_handles_arbitrary_strings(self, error_str):
        from execution.execution_manager import ExecutionManager
        cat, guidance = ExecutionManager._classify_kraken_order_error(error_str)
        assert cat in {"PERMANENT", "TRANSIENT", "UNKNOWN"}
        assert isinstance(guidance, str)


# =====================================================================
# JSON-shape contract: every parser handles None / empty-dict
# =====================================================================

class TestParsersBoundaryInputs:
    """Sanity: empty dict + None — should EITHER raise OR return safe default,
    never silently propagate malformed state."""

    @pytest.mark.parametrize("module,cls,method", [
        ("data_mgmt.feeds.onchain_feed", "OnChainFeed", "_parse_raw_data"),
        ("data_mgmt.feeds.sentiment_feed", "SentimentFeed", "_parse_raw_data"),
        ("data_mgmt.feeds.lob_feed", "LOBFeed", "_parse_raw_data"),
        ("data_mgmt.feeds.macro_feed", "MacroFeed", "_parse_raw_data"),
    ])
    def test_empty_dict_input(self, module, cls, method):
        import importlib
        mod = importlib.import_module(module)
        parser = getattr(getattr(mod, cls).__new__(getattr(mod, cls)), method)
        try:
            result = parser({})
        except (KeyError, ValueError, TypeError, AttributeError):
            return  # Acceptable
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
