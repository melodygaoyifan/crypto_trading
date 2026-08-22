"""[P239] Hygiene batch from the 2026-08-08 full-codebase read-through.

Four items, each small, each pinned so it cannot silently regress:
  1. main.py's header no longer claims "Kraken ONLY" (falsified 2026-06-13,
     stood for two months) or carries its own version literal (drifted from
     VERSION for six months) — and the LIVE_CONFIG_PROFILE line it now
     carries is asserted against docker-compose, not just written down.
  2. TradeIntentV36 declares confidence / confidence_multiplier /
     lead_lag_amplifier_applied (previously dynamic attributes, invisible to
     serialization; defaults equal every consumer's getattr default so the
     declaration is behavior-neutral — P85 contract).
  3. AUTHORITY_MATRIX_HIGH_VOL is unwired BY RECORDED DECISION: the selector
     must never return it until a deliberate wiring with shadow evidence
     lands (P141/P177 family).
  4. The sleeve's drawdown-halt pct and contract cap are config-wired with
     defaults equal to the ctor defaults they replace.
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAIN = (REPO / "main.py").read_text(encoding="utf-8-sig", errors="replace")
HEADER = MAIN[:3000]


class TestHeaderHonesty:
    def test_no_kraken_only_claim(self):
        """Assert the CLAIM is gone, not the substring — the correction note
        legitimately quotes the retired words (the P177 comment-trap; this
        test's first version failed on its own explanation)."""
        assert "Execution: Kraken ONLY" not in HEADER, (
            "the header re-acquired the venue claim that was false for two "
            "months — the sole directional venue is the Coinbase sleeve (P152)"
        )
        assert "SINGLE EXCHANGE MODE (LOCKED)" not in HEADER

    def test_no_second_version_literal(self):
        """The docstring carried 'Version: 6.5.1-...' while VERSION said
        6.8.0 — a second copy of a version is how the first one stops being
        read. Assert no 'Version: <digits>' line, not the bare substring
        (the correction note quotes the old literal)."""
        assert not re.search(r"^Version: [0-9]", HEADER, re.M)
        assert re.search(r"Version: see the VERSION constant", HEADER)

    def test_live_config_profile_line_matches_compose(self):
        """The header names the live profile — assert it against what
        docker-compose actually runs, so this line cannot become the next
        stale claim in the same block it fixed."""
        assert "configs/live_high_risk.json" in HEADER
        compose = (REPO / "docker-compose.hetzner.yml").read_text(
            encoding="utf-8-sig", errors="replace")
        assert "configs/live_high_risk.json" in compose


class TestIntentDeclaredFields:
    def _intent(self):
        from integration.integration_v36 import TradeIntentV36
        return TradeIntentV36()

    def test_fields_exist_with_consumer_matching_defaults(self):
        it = self._intent()
        assert it.confidence == 0.5
        assert it.confidence_multiplier == 1.0
        assert it.lead_lag_amplifier_applied is False

    def test_defaults_still_match_the_consumers_getattr_defaults(self):
        """The declaration is behavior-neutral ONLY while these defaults
        equal the getattr defaults at the read sites. If a consumer's
        default changes, this test forces the field default to be
        reconsidered rather than silently diverging."""
        assert "getattr(intent, 'confidence', 0.5)" in MAIN
        assert "getattr(intent, 'confidence_multiplier', 1.0)" in MAIN
        assert "getattr(intent, 'lead_lag_amplifier_applied', False)" in MAIN


class TestHighVolMatrixStaysUnwired:
    def test_selector_never_returns_the_high_vol_matrix(self):
        """Wiring the high-vol authority downgrade re-points the DECIDER in
        the regimes where money moves fastest — it needs shadow evidence and
        its own P-entry, not a selector edit (see the P239 note at the
        matrix). quant is the discriminator: HIGH_VOL demotes it to ADVISE."""
        from signals.authority_fusion import (
            get_authority_matrix, Authority)
        for mode in ("NORMAL", "OPPORTUNITY", "NO_TRADE", "HIGH_VOL",
                     "EXTREME_VOLATILITY", "garbage", ""):
            m = get_authority_matrix(mode)
            if mode == "NO_TRADE":
                continue
            assert m["quant"] != Authority.ADVISE, (
                f"mode {mode!r}: quant demoted to ADVISE — the HIGH_VOL "
                f"matrix (or an equivalent) is being returned; that wiring "
                f"must be a deliberate, evidenced change"
            )

    def test_the_decision_note_is_present(self):
        src = (REPO / "signals" / "authority_fusion.py").read_text(
            encoding="utf-8-sig", errors="replace")
        i = src.find("AUTHORITY_MATRIX_HIGH_VOL = {")
        assert "[P239] NOT WIRED" in src[:i], (
            "the not-wired decision note above the matrix is gone — without "
            "it the matrix reads as a live control again (P177 family)"
        )


class TestSleeveKnobsConfigWired:
    def test_config_defaults_equal_ctor_defaults(self):
        """Behavior-neutral only while these match — introspected from the
        real signature, not restated (the drift-by-restatement trap)."""
        import inspect
        from exchange.coinbase_sleeve import CoinbaseSleeve
        sig = inspect.signature(CoinbaseSleeve.__init__)
        assert re.search(
            r"^\s+coinbase_max_sleeve_drawdown_pct: float = "
            + re.escape(str(sig.parameters["max_sleeve_drawdown_pct"].default)),
            MAIN, re.M)
        assert re.search(
            r"^\s+coinbase_max_contracts_per_asset: int = "
            + re.escape(str(sig.parameters["max_contracts_per_asset"].default)),
            MAIN, re.M)

    def test_parsed_and_passed(self):
        """P201: declared-but-unparsed silently no-ops; parsed-but-unpassed
        is the same defect one hop later."""
        assert 'data.get("coinbase_max_sleeve_drawdown_pct"' in MAIN
        assert 'data.get("coinbase_max_contracts_per_asset"' in MAIN
        i = MAIN.find("self._coinbase_sleeve = CoinbaseSleeve(")
        assert i > 0
        ctor = MAIN[i:MAIN.find("snapshot()", i)]
        assert "max_sleeve_drawdown_pct=" in ctor
        assert "max_contracts_per_asset=" in ctor

    def test_absent_from_live_profile(self):
        """Absent = ctor defaults = today's behavior. Setting them is an
        operator risk decision, not a side effect of this wiring.

        [P370] That decision has now been MADE for the drawdown halt, by
        explicit operator instruction on the P369 six-year backtest (15% trips
        SOL 48-77x in 6y and removes 60-85% of its return; 25% is a tolerable
        premium on all three). So this pin moves from must-be-ABSENT to the
        DECIDED value (the P237/P270 pattern): a silent revert to 0.15 AND a
        silent loosening past 0.25 both fail. The contracts-per-asset knob is
        still undecided and stays pinned absent."""
        live = json.loads((REPO / "configs" / "live_high_risk.json"
                           ).read_text(encoding="utf-8-sig"))
        assert live.get("coinbase_max_sleeve_drawdown_pct") == 0.25, (
            "P370 decided 0.25; a different value is a new decision needing "
            "its own P-entry")
        assert "coinbase_max_contracts_per_asset" not in live
