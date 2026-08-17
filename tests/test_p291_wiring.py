"""[P291] Parent-owned wiring pins: the config trio for both new flags, the
live-profile DECIDED values, and the two main.py hooks the forks depend on.

The forks built `venue_true_hold_enabled` (constitution) and `maker_reprice`
(sleeve) but could not own main.py or configs/, so nothing they wrote proves
the features are reachable. These are those proofs.

The load-bearing asymmetry pinned here: the maker ladder is ON (execution
cost only — worst case is today's cross-at-timeout) while venue-true hold is
OFF (it crosses ETH and SOL below the asserted-alpha ceiling, i.e. it OPENS
two assets — a P141 decision). If someone flips the hold flag without a
recorded P-entry, `test_venue_true_hold_is_off_in_the_live_profile` fails and
names why.
"""
import dataclasses
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _live_profile() -> dict:
    return json.loads(
        (REPO / "configs" / "live_high_risk.json").read_text(encoding="utf-8-sig"))


class TestConfigTrio:
    @pytest.mark.parametrize("key", ["coinbase_maker_reprice",
                                     "coinbase_venue_true_hold"])
    def test_declared_default_off(self, key):
        import main as m
        f = {x.name: x for x in dataclasses.fields(m.ProductionConfig)}
        assert key in f, f"{key} not declared on ProductionConfig"
        assert f[key].default is False, (
            f"{key} must DEFAULT off — every non-live profile must behave "
            f"exactly as it did before P291")

    @pytest.mark.parametrize("key", ["coinbase_maker_reprice",
                                     "coinbase_venue_true_hold"])
    def test_parsed_from_file(self, key, tmp_path):
        import main as m
        p = tmp_path / "c.json"
        p.write_text(json.dumps({key: True}), encoding="utf-8")
        cfg = m.ProductionConfig.from_file(p)
        assert getattr(cfg, key) is True, (
            f"{key} is declared but not PARSED — the P201 trio: a flag read "
            f"via getattr that from_file never sets is inert forever")


class TestLiveProfileDecidedValues:
    def test_maker_reprice_is_on(self):
        prof = _live_profile()
        assert prof.get("coinbase_maker_reprice") is True
        assert "_coinbase_maker_reprice_note" in prof

    def test_venue_true_hold_is_the_decided_value(self):
        # [P237 pattern] This pin started as "must be OFF" and became the
        # DECIDED value when the operator armed it 2026-08-17 (P291b) after
        # being shown the asset-opening consequence verbatim. It now fails on
        # a silent revert as well as a silent re-flip — either direction is a
        # live-money change that must travel with a recorded decision.
        prof = _live_profile()
        assert prof.get("coinbase_venue_true_hold") is True, (
            "coinbase_venue_true_hold was turned OFF. That reverts the P291b "
            "operator decision (ETH/SOL close again, BTC's threshold returns "
            "to 26.4bps). If deliberate, record it and update this pin.")
        assert "_coinbase_venue_true_hold_armed_note" in prof, (
            "the ARMED note carrying the decision + consequences must travel "
            "with the flag — a bare `true` loses why it was armed")
        assert "_coinbase_venue_true_hold_note" in prof, (
            "the note carrying the arming arithmetic must travel with the flag")


class TestMainWiringHooks:
    def test_funding_rate_call_is_asset_tagged(self):
        # Without asset=, the venue-true hold branch is inert BY DESIGN
        # (untagged funding could belong to another asset). This pin is what
        # makes the feature reachable when the flag is armed.
        from tests._source_scan import code_only
        src = code_only(REPO / "main.py")
        assert "update_funding_rate(" in src
        i = src.index("update_funding_rate(")
        assert "asset=asset" in src[i:i + 200], (
            "update_funding_rate is called WITHOUT asset= — the venue-true "
            "hold branch can never fire")

    def test_hold_gate_is_set_from_config_and_is_independent_of_the_spread_flag(self):
        from tests._source_scan import code_only
        src = code_only(REPO / "main.py")
        assert "venue_true_hold_enabled" in src, (
            "the hold gate is never set from config — the flag would be dead")
        i = src.index("venue_true_hold_enabled")
        blk = src[i:i + 300]
        assert "coinbase_venue_true_hold" in blk
        # Independence: the hold gate must NOT be driven by the spread flag,
        # which is already live. Riding it would open ETH/SOL as a side
        # effect of a cost correction that deliberately did not.
        assert "coinbase_venue_aware_spreads" not in blk, (
            "the hold gate is being driven by the SPREAD flag — that flag is "
            "live, so this would silently arm the asset-opening change")

    def test_sleeve_receives_maker_reprice(self):
        from tests._source_scan import code_only
        src = code_only(REPO / "main.py")
        assert "maker_reprice=bool(getattr(" in src, (
            "the sleeve ctor never receives maker_reprice — the ladder would "
            "sit behind a default-False kwarg forever")
