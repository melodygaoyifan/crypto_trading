"""[P292] XRP (and the breadth set) routing prerequisites — the LAST code
change before a September widening becomes a pure config flip.

P291-C probed the venue and shipped contract specs into the adapter's fallback
tables but deliberately left `SYMBOL_MAP` untouched, pinning its ABSENCE as an
inertness lock. P292 adds those entries, which trades a weak lock for a
readiness property — and this file is the proof that the trade was safe.

WHY A SYMBOL_MAP ENTRY IS NOT A RISK. It only lets an asset be RESOLVED to a
product id. Nothing in the runtime iterates `SYMBOL_MAP` to decide what to
trade; the sleeve driver iterates `config.assets`, and the sleeve's own
`_pid_to_asset` is built from the assets it was constructed with. So the three
locks below are what actually keep XRP flat, and this file pins all three:

    1. config.assets            -> the driver never considers the asset
    2. fractions/caps absent    -> `_sized_contracts` cannot size it
    3. routing state            -> `_coinbase_routed()` is False

Each is independently sufficient. A widening must move #1 and #2 together
(routing state is the operator's one-command step), which is exactly the
"config flip, not a build" property P291-C was aiming at.

Verified at source while writing this: `data_mgmt/feeds/coinbase_funding_feed.py`
is the only other `to_venue_symbol` consumer, it is a disabled scaffold with no
production caller, and it is caller-driven (takes `asset` as an argument) — so
new map entries cannot make it start polling new assets.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# The five P262-certified breadth assets, with the P291-C probe's product ids.
# Restated here (not imported) so this file fails loudly on drift rather than
# agreeing with a corrupted source — the P192 two-file discipline.
BREADTH = {
    "XRP":  "XPP-20DEC30-CDE",
    "ADA":  "ADP-20DEC30-CDE",
    "LTC":  "LCP-20DEC30-CDE",
    "DOGE": "DOP-20DEC30-CDE",
    "BNB":  "BNB-20DEC30-CDE",
}
INCUMBENTS = {"BTC": "BIP-20DEC30-CDE",
              "ETH": "ETP-20DEC30-CDE",
              "SOL": "SLP-20DEC30-CDE"}


def _live_profile() -> dict:
    return json.loads((REPO / "configs" / "live_high_risk.json")
                      .read_text(encoding="utf-8-sig"))


class TestSymbolMapRoundTrip:
    def test_to_venue_symbol_resolves_every_breadth_asset(self):
        from exchange.symbol_mapping import to_venue_symbol
        for asset, pid in BREADTH.items():
            assert to_venue_symbol(asset, "coinbase", "perp") == pid

    def test_from_venue_symbol_inverts_cleanly(self):
        # The sleeve maps venue positions BACK to assets via `_pid_to_asset`;
        # a non-invertible entry would strand a real position under an
        # unknown asset (the P139 book-desync shape).
        from exchange.symbol_mapping import from_venue_symbol
        for asset, pid in BREADTH.items():
            assert from_venue_symbol(pid, "coinbase", "perp") == asset

    def test_no_duplicate_product_ids(self):
        # Two assets sharing a pid would make `from_venue_symbol` ambiguous
        # and silently attribute one asset's position to the other.
        from exchange.symbol_mapping import SYMBOL_MAP
        pids = list(SYMBOL_MAP["coinbase"]["perp"].values())
        assert len(pids) == len(set(pids)), f"duplicate pids: {pids}"

    def test_incumbents_are_untouched(self):
        from exchange.symbol_mapping import SYMBOL_MAP
        perp = SYMBOL_MAP["coinbase"]["perp"]
        for asset, pid in INCUMBENTS.items():
            assert perp[asset] == pid, (
                f"{asset} moved to {perp[asset]} — the breadth edit was "
                f"supposed to be additive")

    def test_contract_specs_agree_across_both_files(self):
        # [P192 two-file class, extended to the new rows] SYMBOL_MAP resolves
        # the pid; the adapter's tables price and size it. A pid present in
        # one and absent from the other is how a fabricated unit enters
        # (P265h: a wrong contract size is a 10-100x order).
        from exchange.coinbase_adapter import CoinbaseAdapter
        from exchange.symbol_mapping import SYMBOL_MAP
        perp_pids = set(SYMBOL_MAP["coinbase"]["perp"].values())
        for table_name in ("_CONTRACT_SIZE_FALLBACK", "_PRICE_INCREMENT_FALLBACK"):
            table = getattr(CoinbaseAdapter, table_name)
            missing = perp_pids - set(table)
            assert not missing, (
                f"{missing} resolve via SYMBOL_MAP but have no {table_name} "
                f"row — on an API hiccup those assets refuse every tick")

    def test_p265h_guard_direction_is_satisfied(self):
        # The guard is one-directional (SYMBOL_MAP pids must be covered by the
        # tables), which is why P291-C landed the table rows FIRST. This pins
        # that the ordering reasoning held: adding map entries did not break it.
        from exchange.coinbase_adapter import CoinbaseAdapter
        from exchange.symbol_mapping import SYMBOL_MAP
        perp_pids = set(SYMBOL_MAP["coinbase"]["perp"].values())
        assert perp_pids >= set(BREADTH.values())
        assert not perp_pids - set(CoinbaseAdapter._CONTRACT_SIZE_FALLBACK)
        assert not perp_pids - set(CoinbaseAdapter._PRICE_INCREMENT_FALLBACK)


class TestBreadthRemainsInert:
    """The three locks. Each is independently sufficient; all three hold."""

    def test_lock1_config_assets_excludes_breadth(self):
        # THE primary lock: main.py's sleeve driver loops `config.assets`.
        # [P412] XRP is the FIRST breadth asset activated (one-first, P197) and
        # is DELIBERATELY in config.assets; the remaining four stay excluded,
        # each needing its own recorded decision after XRP's cycle.
        live_assets = set(_live_profile().get("assets") or [])
        assert live_assets, "live profile has no `assets` — read it again"
        assert "XRP" in live_assets, "[P412] XRP activation regressed"
        for asset in set(BREADTH) - {"XRP"}:
            assert asset not in live_assets, (
                f"{asset} entered config.assets without its own decision — the "
                f"sleeve driver manages it every tick now. Widening past XRP "
                f"needs watching XRP's first cycle (P197) plus a P-entry")

    def test_lock1b_sleeve_only_maps_the_assets_it_was_given(self):
        # Behavioural proof that a SYMBOL_MAP entry alone routes nothing: a
        # sleeve constructed with the home trio has no XRP pid mapping, so a
        # venue position in XRP could not even be attributed to it.
        from exchange.coinbase_sleeve import CoinbaseSleeve
        s = CoinbaseSleeve(object(), assets=("BTC", "ETH", "SOL"))
        assert set(s._pid_to_asset.values()) == {"BTC", "ETH", "SOL"}
        for pid in BREADTH.values():
            assert pid not in s._pid_to_asset

    def test_lock2_no_sizing_entries_exist(self):
        # [P412] XRP is activated and deliberately sized; the remaining four
        # must have no fraction/cap entry.
        prof = _live_profile()
        for key in ("coinbase_target_fraction_by_asset",
                    "coinbase_max_contracts_by_asset"):
            assert "XRP" in (prof.get(key) or {}), (
                f"[P412] XRP activation regressed — missing from {key}")
            for asset in set(BREADTH) - {"XRP"}:
                assert asset not in (prof.get(key) or {}), (
                    f"{asset} has a {key} entry — sizing exists for an asset "
                    f"whose forward read has not happened")

    def test_lock3_routing_state_excludes_breadth(self):
        # The operator's one-command step. Absent file == nothing routed,
        # which is also inert, so this reads the file only if it exists.
        sf = REPO / "data" / "coinbase_routing_state.json"
        if not sf.exists():
            pytest.skip("no local routing state (server-side file)")
        routed = set(json.loads(sf.read_text(encoding="utf-8-sig"))
                     .get("coinbase_assets") or [])
        # [P412] XRP is activated and MAY be routed (its config fractions/caps
        # exist); the remaining four must not be routed without their decision.
        for asset in set(BREADTH) - {"XRP"}:
            assert asset not in routed, (
                f"{asset} is in coinbase_assets — routing was widened without "
                f"the config fractions/caps that make it sizable")

    def test_the_widening_requires_moving_two_things_not_one(self):
        # The readiness property, stated as a test: a widening must move BOTH
        # the asset and its sizing together. [P412] XRP's activation did
        # exactly that (both present) — the discipline followed, not bypassed;
        # the remaining four must have neither-fully so no half-move slips in.
        prof = _live_profile()
        live_assets = set(prof.get("assets") or [])
        fracs = set((prof.get("coinbase_target_fraction_by_asset") or {}))
        assert "XRP" in live_assets and "XRP" in fracs, (
            "[P412] XRP must be fully wired (assets AND sizing) — the "
            "two-things-move discipline is how a widening is done, not skipped")
        for asset in set(BREADTH) - {"XRP"}:
            assert not (asset in live_assets and asset in fracs), (
                f"{asset} is fully wired for trading without its own decision")


class TestSpotSideUntouched:
    def test_breadth_gained_no_kraken_spot_entries(self):
        from exchange.symbol_mapping import SYMBOL_MAP
        spot = SYMBOL_MAP["kraken"]["spot"]
        for asset in BREADTH:
            assert asset not in spot, (
                f"{asset} gained a Kraken spot entry — this widening is a "
                f"CDE perp decision; Kraken spot is structurally flat (P152)")

    def test_breadth_gained_no_coinbase_spot_entries(self):
        # Unverified by any probe, and unused: the sleeve trades perp only.
        from exchange.symbol_mapping import SYMBOL_MAP
        spot = SYMBOL_MAP["coinbase"]["spot"]
        for asset in BREADTH:
            assert asset not in spot

    def test_the_deliberate_sol_usdt_asymmetry_survives(self):
        # [P253d] BTC/ETH are USD, SOL is DELIBERATELY USDT (P133/P135/P137 —
        # Kraken's SOL/USD pair went OnMaintenance). Do not "fix" this.
        from exchange.symbol_mapping import SYMBOL_MAP
        spot = SYMBOL_MAP["kraken"]["spot"]
        assert spot["BTC"] == "BTC/USD"
        assert spot["ETH"] == "ETH/USD"
        assert spot["SOL"] == "SOL/USDT"
