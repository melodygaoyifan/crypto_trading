"""P272 — passive-holdings advisory + per-asset contract caps, pinned.

Operator decisions recorded 2026-08-16: the ~12,300-XRP Kraken bag and ~$3k
BTC are DELIBERATE holds the system does not trade; the Coinbase key stays
as-is. What shipped instead: (a) the certified book label for those held
assets in the 4H heartbeat (informational, trades nothing), and (b) per-
asset contract caps as config — the vol-parity PREP whose defaults are
byte-identical to the flat scalar cap.
"""

import json
import re
from pathlib import Path

import pytest

from tests._source_scan import read_source
from defense.regime_book_shadow import RegimeBookShadow
from exchange.coinbase_sleeve import CoinbaseSleeve

REPO = Path(__file__).resolve().parent.parent
MAIN = REPO / "main.py"


# ===========================================================================
# advisory snapshot
# ===========================================================================

def _rbs_with(records):
    s = object.__new__(RegimeBookShadow)
    s._last_records = records
    return s


class TestAdvisorySnapshot:
    def test_labels_come_from_the_last_record(self):
        s = _rbs_with({"XRP": {"regime": "bear", "direction": 0.0},
                       "BTC": {"regime": "bull", "direction": 1.0}})
        snap = s.advisory_snapshot(["XRP", "BTC"])
        assert snap["XRP"]["regime"] == "bear"
        assert snap["BTC"]["direction"] == 1.0

    def test_absent_asset_is_omitted_never_a_confident_label(self):
        # P2: absence is not a signal — an asset with no record must be
        # missing from the snapshot, not rendered as flat/bull/anything
        s = _rbs_with({})
        assert s.advisory_snapshot(["XRP"]) == {}

    def test_flip_flags_exactly_once(self):
        s = _rbs_with({"XRP": {"regime": "bear", "direction": 0.0}})
        first = s.advisory_snapshot(["XRP"])
        assert first["XRP"]["changed"] is False  # no prior — nothing flipped
        # same label again: still not changed
        assert s.advisory_snapshot(["XRP"])["XRP"]["changed"] is False
        # the book flips to bull
        s._last_records["XRP"] = {"regime": "bull", "direction": 1.0}
        assert s.advisory_snapshot(["XRP"])["XRP"]["changed"] is True, (
            "a label change must be flagged on the first snapshot after it")
        # and only once — the next heartbeat shows the new label unflagged
        assert s.advisory_snapshot(["XRP"])["XRP"]["changed"] is False, (
            "FLIPPED must not repeat every heartbeat (P202: a steady-state "
            "condition rendered as a recurring alert becomes wallpaper)")

    def test_advisory_reads_only_it_never_writes_records(self):
        s = _rbs_with({"BTC": {"regime": "bull", "direction": 1.0}})
        before = dict(s._last_records["BTC"])
        s.advisory_snapshot(["BTC"])
        assert s._last_records["BTC"] == before


class TestAdvisoryWiring:
    def test_heartbeat_builds_and_publishes_the_advisory_field(self):
        src = read_source(MAIN)
        assert "advisory_snapshot(" in src, "heartbeat never calls the snapshot"
        assert '"Passive books (advisory)"' in src, (
            "the advisory field vanished from the Discord heartbeat details")

    def test_config_trio(self):
        src = read_source(MAIN)
        assert re.search(
            r"passive_advisory_assets: List\[str\] = field\(default_factory=list\)",
            src)
        assert 'data.get("passive_advisory_assets", [])' in src

    def test_live_profile_advises_the_operators_actual_holdings(self):
        live = json.loads((REPO / "configs" / "live_high_risk.json")
                          .read_text(encoding="utf-8"))
        assert set(live.get("passive_advisory_assets", [])) == {"BTC", "XRP"}, (
            "the advisory roster is the operator's recorded holdings "
            "(2026-08-16: XRP bag + BTC, both deliberate holds) — changing "
            "it is an operator decision, update the note in the same commit")


# ===========================================================================
# per-asset contract caps
# ===========================================================================

def _sleeve(cur, scalar_cap=1, by_asset=None):
    s = object.__new__(CoinbaseSleeve)
    s._halted = False
    s._halt_reason = ""
    s._max_contracts_per_asset = scalar_cap
    s._max_contracts_by_asset = dict(by_asset or {})
    s._max_net_exposure = None
    s._max_asset_exposure = {}
    s.signed_contracts = lambda asset: cur  # type: ignore[assignment]
    return s


class TestPerAssetContractCaps:
    def test_empty_dict_is_byte_identical_to_the_scalar(self):
        allowed, _ = _sleeve(0.0, scalar_cap=1).can_trade("ETH", 1)
        assert allowed
        blocked, reason = _sleeve(1.0, scalar_cap=1).can_trade("ETH", 1)
        assert not blocked and "coinbase_contract_cap" in reason

    def test_configured_asset_uses_its_own_cap(self):
        s = _sleeve(1.0, scalar_cap=1, by_asset={"ETH": 3})
        allowed, _ = s.can_trade("ETH", 1)   # 1 -> 2, cap 3
        assert allowed, "the per-asset cap must override the scalar"
        blocked, reason = _sleeve(3.0, scalar_cap=1,
                                  by_asset={"ETH": 3}).can_trade("ETH", 1)
        assert not blocked and "> 3" in reason

    def test_unconfigured_asset_falls_back_to_the_scalar(self):
        s = _sleeve(1.0, scalar_cap=1, by_asset={"ETH": 3})
        blocked, reason = s.can_trade("BTC", 1)
        assert not blocked and "> 1" in reason, (
            "an asset absent from the dict must take the SCALAR cap — a "
            "silent default to the dict's values would widen risk for "
            "every unlisted asset")

    def test_derisking_is_always_free_over_any_cap(self):
        # the P195 invariant survives the per-asset form: an over-cap
        # position (venue drift, a lowered cap) can always be trimmed.
        # Values chosen so the trim's RESULT is still above the cap
        # (4 -> 3, cap 2) — the case only the resulting>abs(cur) clause
        # permits (the first version of this test used 4 -> 2, which
        # passes the plain cap check and probed GREEN — a falsification
        # that doesn't fail must be distrusted in both directions, P238)
        s = _sleeve(4.0, scalar_cap=1, by_asset={"ETH": 2})
        allowed, _ = s.can_trade("ETH", -1)
        assert allowed, ("a reduce that stays over-cap was refused — the "
                         "cap is trapping the position it should shrink")

    def test_missing_attr_falls_back_not_refuses(self):
        # P85: a new attribute read on the live order path must be
        # getattr-defended — an old pickle/fixture without the dict must
        # not refuse every order
        s = _sleeve(0.0, scalar_cap=1)
        del s._max_contracts_by_asset
        allowed, _ = s.can_trade("BTC", 1)
        assert allowed

    def test_sized_targets_truth_table(self):
        # [P273] target_for: sign x configured size; the deadband flatten
        # and the P85 missing-attr fallback must both survive sizing
        s = _sleeve(0.0, scalar_cap=5, by_asset={"ETH": 3})
        assert s.target_for("ETH", 0.9) == 3
        assert s.target_for("ETH", -0.9) == -3
        assert s.target_for("BTC", 0.9) == 1, (
            "an unlisted asset must target 1 (the historical size), never "
            "inherit another asset's size")
        assert s.target_for("ETH", 0.05) == 0, (
            "the deadband flatten (|dir|<0.15 -> 0) must survive sizing — "
            "it is what makes the sleeve EXIT on hold")
        del s._max_contracts_by_asset
        assert s.target_for("ETH", 0.9) == 1, (
            "missing attr must fall back to size 1, not refuse (P85)")

    def test_driver_uses_sized_targets(self):
        src_sleeve = read_source(REPO / "exchange" / "coinbase_sleeve.py")
        assert re.search(r"target = self\.target_for\(asset, direction",
                         src_sleeve), (
            "manage_to_signal reverted to the unsized target_for_signal — "
            "the P273 sizing activation is silently dead")
        src_main = read_source(MAIN)
        assert "_sl.target_for(_m_a, _m_dir)" in src_main, (
            "main.py's driver-adjacent reads (ma_filter raw target, "
            "cooldown sent-flat, stop reconcile intent) must use the SIZED "
            "target or they reason about a different book than the driver "
            "trades (P2 reader/writer class)")
        assert "_sl.target_for_signal(" not in src_main

    def _fraction_sleeve(self, equity, prices, fractions,
                         fixed=None, cur=0.0):
        s = _sleeve(cur, scalar_cap=1, by_asset=fixed or {})
        s._target_fraction_by_asset = dict(fractions)
        s.sleeve_equity_usd = lambda: equity  # type: ignore[assignment]
        s._notional_usd = (  # type: ignore[assignment]
            lambda asset, n: prices.get(asset, 0.0) * n)
        return s

    def test_fractions_reproduce_the_p273_book_at_activation_equity(self):
        # [P274] the identity that makes this a safe activation: at the
        # 2026-08-16 equity, 0.15 fractions == the fixed {1,3,1} book
        # exactly — behavior changes ONLY when capital actually moves
        prices = {"BTC": 630.0, "ETH": 188.0, "SOL": 377.0}
        s = self._fraction_sleeve(3775.0, prices,
                                  {"BTC": .15, "ETH": .15, "SOL": .15})
        assert s.target_for("BTC", 0.9) == 1
        assert s.target_for("ETH", 0.9) == 3
        assert s.target_for("SOL", -0.9) == -1

    def test_a_deposit_deploys_itself(self):
        prices = {"BTC": 630.0, "ETH": 188.0, "SOL": 377.0}
        s = self._fraction_sleeve(10900.0, prices,
                                  {"BTC": .15, "ETH": .15, "SOL": .15})
        assert s.target_for("BTC", 0.9) == 2
        assert s.target_for("ETH", 0.9) == 8
        assert s.target_for("SOL", 0.9) == 4
        # and the per-asset contract cap FOLLOWS the size (cap == target)
        # or can_trade would block the scaled book the fixed caps predate
        allowed, _ = s.can_trade("ETH", 8)
        assert allowed, ("the fixed contract cap blocked the equity-scaled "
                         "target — cap must derive from the same sizing "
                         "arithmetic (P274)")

    def test_unpriceable_falls_back_to_the_fixed_book_never_fabricates(self):
        # P2/P208: sizing from missing data is fabrication — an unreadable
        # price or equity must fall back to the known-safe FIXED size
        s = self._fraction_sleeve(0.0, {}, {"ETH": .15}, fixed={"ETH": 3})
        assert s.target_for("ETH", 0.9) == 3
        s2 = self._fraction_sleeve(10900.0, {}, {"ETH": .15},
                                   fixed={"ETH": 3})
        assert s2.target_for("ETH", 0.9) == 3

    def test_fraction_clamp_defuses_fat_fingers(self):
        from exchange.coinbase_sleeve import CoinbaseSleeve as _CS
        import inspect
        # ctor clamps to <=0.25 (the largest standing per-asset gross cap)
        src = inspect.getsource(_CS.__init__)
        assert "min(0.25, max(0.0" in src, (
            "the fraction clamp is gone — a fat-fingered 1.5 would size a "
            "10x book through the target path")

    def test_config_trio_and_decided_live_values(self):
        src = read_source(MAIN)
        assert "coinbase_max_contracts_by_asset: Dict[str, int]" in src
        assert 'data.get("coinbase_max_contracts_by_asset", {})' in src
        assert re.search(r"max_contracts_by_asset=dict\(getattr\(\s*"
                         r"self\.config", src), (
            "declared + parsed but never passed to the ctor — the "
            "P16/P201 dead-flag shape")
        # [P273] The operator flip HAPPENED ("use as much money as
        # possible", 2026-08-16) — the live profile now pins the DECIDED
        # vol-parity values, chosen to fill the STANDING policy caps
        # (max all-same-direction book ~0.42x net vs the 0.50 net cap).
        # A silent revert AND a silent widening both fail loudly (P237).
        live = json.loads((REPO / "configs" / "live_high_risk.json")
                          .read_text(encoding="utf-8"))
        assert live.get("coinbase_max_contracts_by_asset") == {
            "BTC": 1, "ETH": 3, "SOL": 1, "XRP": 1}, (  # [P412] XRP breadth
            "live per-asset sizes changed — if deliberate, update this pin "
            "+ the sizing note + re-check the net-cap arithmetic in the "
            "same commit")
        # [P274] equity-scaled fractions — reproduces the fixed book at
        # activation equity, deploys deposits automatically; max
        # all-same-direction ~0.45x net vs the 0.50 cap.
        # [P370] moved from flat 0.15 x3 to VOL-PARITY {.20,.15,.095} on the
        # strategy-threshold backtest: flat 0.15 put 48% of book risk in SOL.
        # Same 0.445 aggregate budget; NOT a loosening (SOL down, ETH same,
        # BTC up but BTC cannot pass the gate). Pinned to the decided value.
        # [P412] XRP breadth activated (one-asset-first, P197): home
        # UNCHANGED, XRP 0.01 under its post_leverage_cap 0.10, aggregate 0.455
        # < 0.50 net cap. A silent revert or further widening both fail.
        assert live.get("coinbase_target_fraction_by_asset") == {
            "BTC": 0.20, "ETH": 0.15, "SOL": 0.095, "XRP": 0.01}
