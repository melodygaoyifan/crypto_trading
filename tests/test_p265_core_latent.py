"""[P265] core/execution latent defects.

Most of these sit on the Kraken exec stack (dormant for routed assets since
P152) but several are on the SANCTIONED revival paths — exits/unwinds, the
sleeve's contract-size lookups, the 30s watchdog — or corrupt shared state:

  * [FIX-H3]'s "reject" never returned: sub-minimum orders executed carrying
    a forged veto (a vetoed-and-filled contradiction).
  * Four swallowed NameErrors: AgentOBSnapshot (never imported here — the
    W7/WIRE-5 RL execution advisory never ran once since extraction, P72
    class); `_sizing` on tranche adds (the sizer cap silently skipped when
    it binds); `exit_price` in the flip-close ledger; `_c11_drl_rec` when
    pnl_attribution is absent.
  * Post-fill fee venue mislabeled Kraken fills of routed assets as
    "coinbase" (every order this function places is a Kraken order).
  * Full exits sized the close as entry_notional / CURRENT price — wrong
    unit count; a margin-short close could over-buy into a residual LONG.
  * FLIP_GATE ran AFTER the fill and discarded an executed order from the
    books (P139/P140 desync class) — moved pre-order.
  * Sub-50% timeout partials were erased (CANCELLED result without
    filled_size) — real venue inventory in no internal book.
  * The legacy clamp read USD for every BUY — wrong book for SOL/USDT.
  * thesis_budget_max_reentry: declared 1, parsed 0 — FIX-M1 never live
    from a config boot.
  * SOL forced-exit was long-only by sign convention (signed exposure
    <= 0.01 read a short as "no position").
  * trade_gate's data-health EXIT_ONLY returned REJECT for everything
    (proposal.is_exit never existed) — closes were rejected too (P195 shape).
  * The sleeve's `or 1.0` contract-size fallback fabricated a UNIT (10-100x
    orders on a transient lookup failure); the fallback tables have no drift
    guard against SYMBOL_MAP (a contract roll empties them, P192 class).
  * FastRiskTick evaluation failures were swallowed at DEBUG — a dead 30s
    watchdog indistinguishable from a quiet one.
"""

import asyncio
import re
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MAIN_SRC = (REPO / "main.py").read_text(encoding="utf-8-sig")
EXEC_SRC = (REPO / "core" / "execution_service.py").read_text(encoding="utf-8")
EM_SRC = (REPO / "execution" / "execution_manager.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# source pins (dormant-path fixes; each verified red-on-revert)
# ---------------------------------------------------------------------------

class TestExecutionServicePins:
    def test_fix_h3_actually_returns(self):
        i = EXEC_SRC.index('"EXPOSURE_BELOW_MINIMUM_VIABLE"')
        seg = EXEC_SRC[i:i + 600]
        assert '"status": "REJECTED"' in seg, (
            "[FIX-H3] stamps the veto and falls through again — sub-minimum "
            "orders execute carrying veto_active=True")

    def test_agent_ob_snapshot_is_importable_here(self):
        import core.execution_service as es
        assert getattr(es, "AgentOBSnapshot", None) is not None, (
            "AgentOBSnapshot is not bound in execution_service — every W7 "
            "evaluation NameErrors into a debug swallow and WIRE-5 never runs")

    def test_sizer_log_does_not_read_sizing_on_tranche_adds(self):
        i = EXEC_SRC.index("[SIZER] {asset}: capped")
        seg = EXEC_SRC[max(0, i - 1200):i]
        assert "_cap_label" in seg, (
            "the SIZER cap log reads _sizing.cap_applied unconditionally "
            "again — NameError on tranche adds, silently skipping the cap "
            "exactly when it binds")

    def test_flip_close_ledger_uses_fill_price(self):
        i = EXEC_SRC.index("_edrl_ledger_flip.record_close")
        seg = EXEC_SRC[i:i + 400]
        assert "float(fill_price)" in seg
        assert "float(exit_price)" not in seg, (
            "the flip-close ledger reads exit_price (assigned only in "
            "BRANCH A) — every flip close NameErrors and the record never "
            "closes")

    def test_c11_drl_fields_are_hoisted(self):
        hoist = EXEC_SRC.index("_c11_drl_rec = None")
        gate = EXEC_SRC.index("if ctx.pnl_attribution is not None:")
        assert hoist < gate, (
            "_c11_drl_rec is defined only inside the pnl_attribution try — "
            "with it absent, DRL-COUNT NameErrors and "
            "promotion_gate.record_trade never runs")

    def test_fee_venue_is_kraken_truth(self):
        assert '_venue = "kraken"' in EXEC_SRC
        assert re.search(
            r'_venue = "coinbase" if _coinbase_routed', EXEC_SRC) is None, (
            "the post-fill fee venue is routed-asset-derived again — a "
            "Kraken fill of a routed asset gets modelled at Coinbase 0/3bps "
            "and stamped venue=coinbase into attribution")

    def test_full_exit_sizes_by_entry_price(self):
        i = EXEC_SRC.index("_close_entry_px")
        seg = EXEC_SRC[i:i + 1600]
        assert "base_quantity = _close_notional / (" in seg, (
            "full exits size by entry_notional / current_price again — "
            "wrong unit count; a margin-short close over-buys into a "
            "residual long")

    def test_flip_gate_runs_pre_order(self):
        gate = EXEC_SRC.index('"status": "FLIP_BLOCKED"')
        order = EXEC_SRC.index("result = ctx.execution_manager.execute_order(")
        assert gate < order, (
            "the FLIP_GATE returns FLIP_BLOCKED after the venue fill again "
            "— an executed order is discarded from the books (P139/P140)")
        assert EXEC_SRC.count('"status": "FLIP_BLOCKED"') == 1, (
            "a second FLIP_BLOCKED return exists — check it is not the old "
            "post-fill variant")

    def test_flip_gate_does_not_mutate_the_intent(self):
        i = EXEC_SRC.index("FLIP GATE — moved PRE-ORDER")
        seg = EXEC_SRC[i:i + 1800]
        assert "intent.target_exposure = 0" not in seg
        assert "intent.direction = 0" not in seg


class TestExecutionManagerPins:
    def test_cancelled_partial_carries_its_fill(self):
        i = EM_SRC.index("Limit order timeout (partial=")
        seg = EM_SRC[max(0, i - 1200):i]
        assert "filled_size=filled_qty" in seg, (
            "the timeout-CANCELLED result omits filled_size again — up to "
            "49.9% real venue inventory lands in no internal book")


class TestClampQuoteCurrency:
    def _manager(self, balances):
        from execution.execution_manager import ExecutionManager
        m = object.__new__(ExecutionManager)
        m.dry_run = False
        m.logger = types.SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: None)
        m.exchange = types.SimpleNamespace(
            fetch_balance=lambda: balances,
            fetch_ticker=lambda s: {"last": 100.0})
        return m

    def test_usdt_quoted_buy_clamps_against_usdt(self):
        from execution.execution_manager import OrderSide
        m = self._manager({"USDT": {"free": 1000.0}, "USD": {"free": 0.0}})
        out = m._clamp_size_to_balance("SOL/USDT", OrderSide.BUY, 5.0, 100.0)
        # size 5 < max 9 (1000*0.9/100) -> passes through unclamped
        assert out == pytest.approx(5.0), (
            f"got {out} — a funded USDT book was judged by the empty USD "
            "book (P265: the clamp hardcoded USD as every pair's quote)")

    def test_usdt_buy_is_not_zeroed_by_an_empty_usd_book(self):
        from execution.execution_manager import OrderSide
        m = self._manager({"USDT": {"free": 1000.0}, "USD": {"free": 0.0}})
        out = m._clamp_size_to_balance("SOL/USDT", OrderSide.BUY, 50.0, 100.0)
        assert out > 0.0, (
            "a SOL/USDT BUY with a funded USDT book was zeroed against the "
            "USD balance")
        assert out == pytest.approx(9.0)  # 1000*0.9/100

    def test_usd_quoted_buy_still_clamps_against_usd(self):
        from execution.execution_manager import OrderSide
        m = self._manager({"USD": {"free": 500.0}, "USDT": {"free": 0.0}})
        out = m._clamp_size_to_balance("BTC/USD", OrderSide.BUY, 50.0, 100.0)
        assert out == pytest.approx(4.5)


class TestConfigAndForcedExit:
    def test_thesis_budget_parse_default_matches_the_declared_default(self):
        # declared: `thesis_budget_max_reentry: int = 1  # [FIX-M1]`
        m = re.search(r"thesis_budget_max_reentry:\s*int\s*=\s*(\d+)", MAIN_SRC)
        declared = int(m.group(1))
        m2 = re.search(
            r'"max_reentry_after_budget_hit",\s*(\d+)\)', MAIN_SRC)
        parsed = int(m2.group(1))
        assert parsed == declared == 1, (
            f"declared={declared} parsed={parsed} — every config-file boot "
            "silently reverts FIX-M1 again (the live profile has no "
            "thesis_budget section, P239 class)")

    def test_sol_forced_exit_sees_shorts(self):
        from defense.production_reliability import SOLDominanceForcedExit
        fe = SOLDominanceForcedExit()
        sig = fe.check_forced_exit(
            asset="SOL", dominance_active=False, dominance_ttl=0,
            current_phase="DISTRIBUTION", vpin=0.95, correlation=0.5,
            price_move_pct=0.0, current_exposure=-0.30)
        assert sig.should_exit, (
            "a SHORT position with VPIN 0.95 did not force-exit — signed "
            "exposure <= 0.01 reads a short as 'no position' again "
            "(long-only protection by sign convention)")

    def test_sol_forced_exit_still_noops_when_flat(self):
        from defense.production_reliability import SOLDominanceForcedExit
        fe = SOLDominanceForcedExit()
        sig = fe.check_forced_exit(
            asset="SOL", dominance_active=False, dominance_ttl=0,
            current_phase="DISTRIBUTION", vpin=0.95, correlation=0.5,
            price_move_pct=0.0, current_exposure=0.0)
        assert not sig.should_exit


class TestTradeGateExitOnly:
    def test_data_health_exit_only_returns_the_exit_only_decision(self):
        src = (REPO / "defense" / "trade_gate.py").read_text(encoding="utf-8")
        i = src.index("entries blocked, exits allowed")
        seg = src[max(0, i - 800):i + 800]
        assert "GateDecision.EXIT_ONLY" in seg, (
            "data-health EXIT_ONLY returns REJECT again — the p0 consumer "
            "maps that to allow_exit=False: position CLOSES rejected when "
            "data degrades (the P195 shape)")

    def test_the_sleeve_holds_on_data_health_vetoes(self):
        from main import SLEEVE_HOLD, sleeve_direction_from_intent
        d, _ = sleeve_direction_from_intent(
            types.SimpleNamespace(
                direction=0.9, target_exposure=0.3, veto_active=True,
                veto_reason="[TRADE_GATE] DATA_HEALTH_EXIT_ONLY"), 0.0)
        assert d is SLEEVE_HOLD


class TestContractSizeRefusal:
    def _sleeve(self, adapter):
        from exchange.coinbase_sleeve import CoinbaseSleeve
        s = object.__new__(CoinbaseSleeve)
        s._adapter = adapter
        s._assets = ("SOL",)
        s._reconcile_ok = True
        s._halted = False
        s._halt_reason = ""
        s._max_contracts_per_asset = 1
        s._max_net_exposure = None
        s._max_asset_exposure = {}
        s._flip_persist_ticks = 0
        s._flip_pending = {}
        s._protective_stop_pct = 0.10
        s._protective_stop_assets = None
        s._last_positions = {"SOL": {"product_id": "SLP-20DEC30-CDE",
                                     "signed_contracts": 1,
                                     "entry_vwap": 72.0}}
        s.reconcile_positions = lambda: s._last_positions  # type: ignore[assignment]
        return s

    def _adapter(self, cs=None):
        class _A:
            def __init__(self):
                self.placed = []
                self._client = types.SimpleNamespace(
                    get_product=lambda product_id: {"mid_market_price": "72.0"})

            def is_connected(self): return True
            def to_venue_symbol(self, a, m="perp"): return "SLP-20DEC30-CDE"
            def _contract_size(self, pid): return cs
            async def fetch_open_orders(self, symbol=None): return []
            async def cancel_order(self, oid, sym): return True

            async def place_order(self, req):
                self.placed.append(req)
                return types.SimpleNamespace(success=True, error_code="",
                                             error_message="")
        return _A()

    def test_execute_target_refuses_without_a_contract_size(self):
        a = self._adapter(cs=None)
        s = self._sleeve(a)
        r = asyncio.run(s.execute_target("SOL", 0))  # even a flatten
        assert r["status"] == "ERROR" and "no_contract_size" in r["reason"], (
            f"got {r} — an unknown contract size fabricated a unit; the "
            "resulting order is 10-100x the intended size")
        assert a.placed == []

    def test_stop_refuses_without_a_contract_size(self):
        a = self._adapter(cs=None)
        s = self._sleeve(a)
        r = asyncio.run(s.ensure_protective_stop("SOL"))
        assert r["status"] == "NO_CONTRACT_SIZE"
        assert a.placed == []

    def test_fallback_tables_cover_every_symbol_map_pid(self):
        # [P192 two-file class] a contract roll updates SYMBOL_MAP and
        # silently empties the adapter's fallback tables for the new pids.
        from exchange.coinbase_adapter import CoinbaseAdapter
        from exchange.symbol_mapping import SYMBOL_MAP
        perp_pids = set(SYMBOL_MAP["coinbase"]["perp"].values())
        assert perp_pids, "no coinbase perp pids found in SYMBOL_MAP"
        missing_cs = perp_pids - set(CoinbaseAdapter._CONTRACT_SIZE_FALLBACK)
        missing_pi = perp_pids - set(CoinbaseAdapter._PRICE_INCREMENT_FALLBACK)
        assert not missing_cs, (
            f"{missing_cs} in SYMBOL_MAP but not _CONTRACT_SIZE_FALLBACK — "
            "on an API hiccup those assets hit the refusal path every tick")
        assert not missing_pi, (
            f"{missing_pi} missing from _PRICE_INCREMENT_FALLBACK")


class TestFastRiskEscalation:
    def test_eval_failures_are_no_longer_debug_swallowed(self):
        assert "eval skipped" not in MAIN_SRC.replace(
            "[SOTA-ACT] Weight eval skipped", ""), (
            "the FastRiskTick evaluation swallow is back at DEBUG — a dead "
            "30s watchdog is indistinguishable from a quiet one")
        assert MAIN_SRC.count("inter-tick watchdog is NOT running") == 2, (
            "the escalation must exist at BOTH loop sites (paper + live)")
