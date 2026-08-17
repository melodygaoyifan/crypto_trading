"""[P289] Venue-true spreads for Coinbase-routed assets.

The Kraken-era ASSET_SPREAD_BPS constants ({BTC:3, ETH:5, SOL:10} per leg)
were applied to CDE nano contracts for the sleeve's whole life. A live
read-only order-book probe (2026-08-16, weekend book) measured median FULL
spreads BTC 1.58 / ETH 5.26 / SOL 2.65 bps — SOL's constant was ~4x the
measured full spread. The CDE table charges the FULL spread rounded
conservatively (true taker cost is the half-spread, so ~2x buffer, P167).

Load-bearing properties pinned here:
  1. Venue memory: only set_spread_venue('coinbase') switches an asset to
     the CDE table; anything else (absent, None, garbage) keeps Kraken —
     a lookup failure must OVERCHARGE, never undercharge (P167/P172).
  2. The measurement is honest in both directions: ETH's CDE value (5.5)
     is HIGHER than its Kraken value (5.0) — this change is a re-pricing,
     not a loosening-by-fiat, and this test fails if someone quietly
     lowers ETH's CDE entry below the measured full spread.
  3. The re-pricing does NOT re-open ETH/SOL: even at CDE spreads their
     gate thresholds stay above the 30bps asserted-alpha ceiling. Pinned
     so a later constant change that WOULD re-open them fails loudly and
     gets its own recorded decision instead of arriving silently.
"""
import pytest

from defense.constitution import FrictionComponents


def _fc() -> FrictionComponents:
    return FrictionComponents()


class TestVenueMemory:
    def test_default_is_kraken_table(self):
        f = _fc()
        for asset, want in (("BTC", 3.0), ("ETH", 5.0), ("SOL", 10.0)):
            f.update_for_asset(asset)
            assert f.slippage_bps == want

    def test_coinbase_venue_switches_to_cde_table(self):
        f = _fc()
        for asset, want in (("BTC", 2.0), ("ETH", 5.5), ("SOL", 4.0)):
            f.set_spread_venue(asset, "coinbase")
            f.update_for_asset(asset)
            assert f.slippage_bps == want, asset

    def test_clearing_the_venue_reverts_to_kraken(self):
        f = _fc()
        f.set_spread_venue("SOL", "coinbase")
        f.update_for_asset("SOL")
        assert f.slippage_bps == 4.0
        f.set_spread_venue("SOL", None)
        f.update_for_asset("SOL")
        assert f.slippage_bps == 10.0

    @pytest.mark.parametrize("bad", [None, "", "kraken", "KRAKEN", "cde",
                                     "binance", "coinbase-intx", " "])
    def test_anything_but_coinbase_keeps_kraken(self, bad):
        # Fail direction: an unrecognized venue string must OVERCHARGE.
        # 'cde' is deliberately NOT accepted — the caller passes the
        # resolve_venue_fee_bps venue name ('coinbase'); accepting aliases
        # invites drift between the fee and spread resolvers.
        f = _fc()
        f.set_spread_venue("SOL", bad)
        f.update_for_asset("SOL")
        assert f.slippage_bps == 10.0, bad

    def test_per_asset_independence(self):
        f = _fc()
        f.set_spread_venue("BTC", "coinbase")
        f.update_for_asset("SOL")
        assert f.slippage_bps == 10.0  # SOL untouched by BTC's venue
        f.update_for_asset("BTC")
        assert f.slippage_bps == 2.0

    def test_pair_suffix_normalization(self):
        f = _fc()
        f.set_spread_venue("SOL/USD", "coinbase")
        f.update_for_asset("SOLUSD")
        assert f.slippage_bps == 4.0

    def test_unknown_asset_on_cde_falls_back_without_undercharging(self):
        f = _fc()
        f.set_spread_venue("XRP", "coinbase")
        f.update_for_asset("XRP")
        # XRP is in neither table: falls to the 5.0 default, same as Kraken.
        assert f.slippage_bps == 5.0


class TestHonestInBothDirections:
    def test_eth_cde_is_not_below_the_measured_full_spread(self):
        # Measured ETH full spread 5.26bps; the CDE entry must stay >= that
        # (the ONLY direction this table may drift without a new probe is UP).
        f = _fc()
        assert f.CDE_SPREAD_BPS["ETH"] >= 5.26

    def test_cde_values_carry_the_2x_buffer_over_half_spread(self):
        # True taker cost = half-spread (BTC 0.79 / ETH 2.63 / SOL 1.32).
        # Each CDE entry must be >= 2x the measured half-spread, i.e. >= the
        # full spread — quietly shrinking the buffer needs a new probe + entry.
        f = _fc()
        measured_half = {"BTC": 0.79, "ETH": 2.63, "SOL": 1.32}
        for a, h in measured_half.items():
            assert f.CDE_SPREAD_BPS[a] >= 2 * h, a


class TestDoesNotReopenEthSol:
    """The P289 finding the entry records: honest spreads move SOL's
    threshold ~55 -> ~40bps and ETH ~43bps — both still above the 30bps
    asserted-alpha ceiling (40 x |sig| x 0.75 at |sig|=1). If the constants
    ever change such that this arithmetic flips, that is a gate REOPENING
    and needs its own recorded decision, not a silent constant edit."""

    ALPHA_CEILING_BPS = 40.0 * 1.0 * 0.75  # trend base_edge x max |sig| x feedback
    EV_MULT = 1.1                          # NORMAL_MULTIPLIER
    # Per-asset hold/margin cost as observed in the live gate logs
    # 2026-08-16: BTC "margin=7", ETH/SOL "14.0bps hold".
    HOLD_BPS = {"BTC": 7.0, "ETH": 14.0, "SOL": 14.0}
    FEE_PLUS_LAT = 3.0 + 2.0               # taker fee + latency per leg
    SMART_BETA_MULT = 1.1435               # live gate-mult observed (P230)

    def _threshold(self, asset: str, slip: float) -> float:
        per_leg = self.FEE_PLUS_LAT + slip
        return (self.EV_MULT * (2 * per_leg + self.HOLD_BPS[asset])
                * self.SMART_BETA_MULT)

    def test_eth_and_sol_stay_closed_at_cde_spreads(self):
        f = _fc()
        for asset in ("ETH", "SOL"):
            thr = self._threshold(asset, f.CDE_SPREAD_BPS[asset])
            assert thr > self.ALPHA_CEILING_BPS, (
                f"{asset}: threshold {thr:.1f} <= ceiling "
                f"{self.ALPHA_CEILING_BPS:.1f} — the venue-true spread "
                f"re-pricing would RE-OPEN this asset; that is a P141 "
                f"decision, record it before changing the constant")

    def test_btc_pass_becomes_robust_not_marginal(self):
        f = _fc()
        thr = self._threshold("BTC", f.CDE_SPREAD_BPS["BTC"])
        # ~26.4bps vs the 30bps ceiling: passes with margin, where the
        # Kraken-spread threshold (~28.9) needed ALLOW_EPSILON.
        assert thr < self.ALPHA_CEILING_BPS
        kraken_thr = self._threshold("BTC", f.ASSET_SPREAD_BPS["BTC"])
        assert thr < kraken_thr


class TestConfigTrioAndWiring:
    """[P201 trio] declared + parsed + consumed, and the live profile pins
    the DECIDED value (P237 pattern — the flag was enabled by explicit
    operator instruction with the probe provenance annotated in-file)."""

    def test_declared_with_default_off(self):
        import main as m
        import dataclasses
        f = {x.name: x for x in dataclasses.fields(m.ProductionConfig)}
        assert "coinbase_venue_aware_spreads" in f
        assert f["coinbase_venue_aware_spreads"].default is False

    def test_parsed_from_file(self, tmp_path):
        import json as _json
        import main as m
        p = tmp_path / "c.json"
        base = {"coinbase_venue_aware_spreads": True}
        p.write_text(_json.dumps(base), encoding="utf-8")
        cfg = m.ProductionConfig.from_file(p)
        assert cfg.coinbase_venue_aware_spreads is True

    def test_live_profile_pins_the_decided_value(self):
        import json as _json
        from pathlib import Path
        prof = _json.loads(
            (Path(__file__).resolve().parent.parent / "configs" /
             "live_high_risk.json").read_text(encoding="utf-8-sig"))
        assert prof.get("coinbase_venue_aware_spreads") is True
        assert "_coinbase_venue_aware_spreads_note" in prof

    def test_venue_fee_block_wires_the_spread_memory(self):
        # The set_spread_venue call must sit in the [VENUE-FEE] block and be
        # fed by the SAME resolution (_vf_venue/_venue_fee_applied) the fee
        # half uses — a second venue lookup is the P172 twin-resolver trap.
        from pathlib import Path
        from tests._source_scan import code_only
        src = code_only(Path(__file__).resolve().parent.parent / "main.py")
        assert "set_spread_venue(" in src
        # Anchor on the CALL SITE (the flag string's first occurrence is the
        # from_file parse line, not the block). Window spans the whole
        # try/else so both branches are visible.
        anchor = src.index("set_spread_venue(")
        blk = src[max(0, anchor - 600):anchor + 900]
        assert "_vf_venue if _venue_fee_applied else None" in blk
        assert "coinbase_venue_aware_spreads" in blk  # flag-gated here
        # flag-off branch must CLEAR the memory (revert must be one flag)
        assert blk.count("set_spread_venue(") >= 2
