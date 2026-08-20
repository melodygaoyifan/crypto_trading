"""P271 — breadth shadow books + the September execution kit, pinned.

The three residuals from the profitability autopsy, addressed:
  1. the dead live driver -> the tripwire countdown is instrumented and the
     decision tree is PRE-COMMITTED (criteria fixed before results exist);
  2. the pending replacements -> scripts/september_check.py makes the weekly
     read one command, including OHLCV for assets the scorer would otherwise
     report as ohlcv_missing (the P199/P264 registered-but-unscorable trap);
  3. the modest-earnings residual -> trend-only forward ledgers for the five
     P262-certified never-fitted assets (XRP/ADA/LTC/DOGE/BNB), starting
     their P166 clocks now so a widening decision has forward evidence.
"""

from pathlib import Path

import pytest

from tests._source_scan import read_source
from defense.regime_book_shadow import (
    BREADTH_ASSETS, BOOKS_VERSION, KRAKEN_PAIRS, ADJ_PARAMS, VOLSKIP_THR,
    RegimeBookShadow, book_target)

REPO = Path(__file__).resolve().parent.parent


class TestBreadthBooks:
    def test_the_five_p262_assets_exactly(self):
        assert set(BREADTH_ASSETS) == {"XRP", "ADA", "LTC", "DOGE", "BNB"}, (
            "the breadth roster is the P262 unread-era probe's certified "
            "set — adding an asset requires its own certification evidence")

    @pytest.mark.parametrize("asset", ["XRP", "ADA", "LTC", "DOGE", "BNB"])
    def test_trend_only_book_form(self, asset):
        # the CERTIFIED mechanism verbatim: bull -> long, everything else
        # flat. No funding legs — P262 marks funding legs as the
        # uncertified slice even on BTC.
        assert book_target(asset, "bull", None) == (1.0, "trend_hold")
        assert book_target(asset, "bear", None) == (0.0, "trend_flat")
        assert book_target(asset, "peace", None) == (0.0, "trend_flat")
        assert book_target(asset, "warmup", None)[0] == 0.0
        # funding z must not change anything (no funding legs)
        assert book_target(asset, "bear", 3.0) == (0.0, "trend_flat"), (
            "a breadth asset grew a funding leg — that is a NEW uncertified "
            "strategy, not the certified trend/hold mechanism")

    @pytest.mark.parametrize("asset", ["XRP", "ADA", "LTC", "DOGE", "BNB"])
    def test_versioned_and_fetchable(self, asset):
        assert BOOKS_VERSION[asset] == "v1_breadth_trend_only"
        assert asset in KRAKEN_PAIRS, (
            "no Kraken pair mapped — the harness would silently skip the "
            "asset every tick (fetch returns None)")

    def test_doge_uses_krakens_xdg_pair(self):
        # Kraken names Dogecoin XDG; "DOGEUSD" resolves to nothing.
        assert KRAKEN_PAIRS["DOGE"] == "XDGUSD"

    def test_default_tick_roster_includes_breadth(self):
        import inspect
        sig = inspect.signature(RegimeBookShadow.tick)
        default = sig.parameters["assets"].default
        assert set(BREADTH_ASSETS) <= set(default), (
            "the tick default roster lost the breadth assets — their "
            "forward ledgers would silently stop accruing (P199 class)")
        assert {"BTC", "ETH", "SOL"} <= set(default)

    def test_breadth_assets_get_no_adj_or_volskip_legs(self):
        # the overlay legs are per-asset MEASURED mechanisms; a default
        # would forward-test parameters no lab ever selected
        assert not (set(BREADTH_ASSETS) & set(ADJ_PARAMS))
        assert not (set(BREADTH_ASSETS) & set(VOLSKIP_THR))

    def test_adj_leg_is_membership_gated_in_source(self):
        src = read_source(REPO / "defense" / "regime_book_shadow.py")
        assert "if asset in ADJ_PARAMS:" in src, (
            "the adj leg lost its membership gate — the old .get() default "
            "(1,1,0) near-passthrough would duplicate every breadth row "
            "under a second strategy name")
        assert 'ADJ_PARAMS.get(asset, (1, 1, 0))' not in src


class TestSeptemberKit:
    def test_check_script_covers_every_candidate_prefix(self):
        src = read_source(REPO / "scripts" / "september_check.py")
        for prefix in ("regimebook", "derivflow", "ma_filter", "etfflow"):
            assert prefix in src, f"september_check lost candidate {prefix}"
        assert "tripwire" in src

    def test_check_script_builds_breadth_ohlcv(self):
        # without price series the scorer reports ohlcv_missing for the
        # breadth ledgers — registered-but-unscorable, the exact P264 trap
        src = read_source(REPO / "scripts" / "september_check.py")
        for asset in BREADTH_ASSETS:
            assert asset in src, (
                f"september_check does not build OHLCV for {asset} — its "
                "ledger would be unscorable on the read date (P264)")
        assert "rows[:-1]" in src, (
            "the in-progress Kraken candle must be dropped (P253c)")

    def test_decision_tree_precommits_every_read(self):
        doc = read_source(REPO / "docs" / "SEPTEMBER_DECISION_TREE.md")
        for anchor in ("tripwire", "ma_filter", "regimebook", "derivflow",
                       "volskip", "etfflow", "P166"):
            assert anchor in doc, f"decision tree lost the {anchor} read"
        assert "Failure-of-everything" in doc, (
            "the all-fail branch must be pre-committed or it gets "
            "improvised under pressure")


class TestBothReadsRun:
    """[P332] september_check must run the POOLED read, not only per-asset.

    P293g measured that an unpooled 30-day window cannot certify at 16h (needs
    IC >= 0.302 against an economic bar of ~0.13), so a per-asset-only exam is
    a clock that cannot fire — on the reads this entire roster is waiting for.
    P299 built --pool-assets and nothing called it (P170).
    """

    def _src(self):
        import io
        from pathlib import Path
        return io.open(Path(__file__).resolve().parents[1] / "scripts" /
                       "september_check.py", encoding="utf-8").read()

    def test_the_pooled_read_is_actually_invoked(self):
        assert '"--pool-assets"' in self._src()

    def test_the_per_asset_read_survives_as_diagnosis(self):
        """Swapping one for the other would lose the per-asset breakdown that
        P307c needed to explain two labs that appeared to disagree."""
        src = self._src()
        assert src.count("analytics.shadow_ic.compute_shadow_ic") >= 2

    def test_a_refusal_in_either_read_is_a_refusal(self):
        """A missing pooled read must not hide behind a healthy per-asset one
        (P199, one level up)."""
        assert "return rc_asset or rc_pooled" in self._src()

    def test_the_decision_tree_states_which_read_governs(self):
        import io
        from pathlib import Path
        doc = io.open(Path(__file__).resolve().parents[1] / "docs" /
                      "SEPTEMBER_DECISION_TREE.md", encoding="utf-8").read()
        assert "WHICH READ GOVERNS" in doc
        assert "POOLABLE_FAMILIES" in doc
