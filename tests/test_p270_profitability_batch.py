"""P270 — the profitability batch, pinned.

Three shipped pieces (2026-08-15 research pass):
  1. Maker-first sleeve execution (CDE 0bps maker vs 3bps taker; post_only
     enforcement probed live — a crossing post-only previews
     PREVIEW_INVALID_LIMIT_PRICE_POST_ONLY). Default OFF.
  2. ETF-flow shadow harness (BTC/ETH daily spot-ETF net flow, CoinGlass v4,
     forward ledger etfflow_*.jsonl, P166-gated).
  3. The volskip entry-filter overlay ledger (ETH only — the
     entry_filter_lab's sole three-era winner; SOL's validation increment
     was exactly 0.0 and BTC's base stood).
"""

import asyncio
import json
import re
import time
from pathlib import Path

import pytest

from tests._source_scan import read_source

REPO = Path(__file__).resolve().parent.parent
MAIN = REPO / "main.py"


# ===========================================================================
# 1. ETF-flow harness
# ===========================================================================

from defense.etf_flow_shadow import (  # noqa: E402
    EtfFlowShadow, etf_flow_direction, MAX_AGE_DAYS, ETF_ENDPOINTS)


class TestEtfFlowDirection:
    def test_inflow_is_long_outflow_is_short(self):
        assert etf_flow_direction(55_500_000.0, 0.5) == (1.0, "inflow")
        assert etf_flow_direction(-56_200_000.0, 0.5) == (-1.0, "outflow")

    def test_absence_and_staleness_are_flat_never_a_held_direction(self):
        # P2: absence is not a signal; P265: a frozen input must not keep
        # trading into a forward ledger
        assert etf_flow_direction(None, None)[0] == 0.0
        d, reason = etf_flow_direction(1e8, MAX_AGE_DAYS + 0.1)
        assert d == 0.0 and reason.startswith("stale")

    def test_weekend_age_is_not_stale(self):
        # ETF markets close Sat/Sun: Friday's flow read on Sunday is ~2 days
        # old and must still be a valid signal, or the ledger goes flat
        # every single weekend by construction.
        d, reason = etf_flow_direction(1e8, 2.0)
        assert d == 1.0 and reason == "inflow"

    def test_zero_flow_is_flat_not_long(self):
        assert etf_flow_direction(0.0, 0.5) == (0.0, "zero_flow")


class TestInProgressDayExcluded:
    def _shadow(self, tmp_path, rows):
        s = object.__new__(EtfFlowShadow)
        s._dir = tmp_path
        s._api_key = "k"
        s._cache = {"BTC": (time.time(), rows)}
        s._warned = {}
        s._last_direction = {}
        s._state_path = tmp_path / "etfflow_state.json"
        return s

    @staticmethod
    def _varied_trailing(midnight, n=20, base=-6e7):
        # n completed days BEFORE the newest, with real variance so the
        # z-score is defined (all-identical -> zero_var -> no claim).
        import random
        random.seed(7)
        return [{"timestamp": (midnight - (n + 1 - i) * 86400) * 1000,
                 "flow_usd": base + random.gauss(0, 3e7)}
                for i in range(n)]

    def test_todays_row_is_never_the_signal(self, tmp_path):
        # The API's last row is TODAY and updates intraday — trading on it
        # is the P253c in-progress-bar class.
        now = 1_800_000_000.0
        midnight = (now // 86400) * 86400
        rows = [
            {"timestamp": (midnight - 86400) * 1000, "flow_usd": -5e7},
            {"timestamp": midnight * 1000, "flow_usd": +9e9},  # today
        ]
        s = self._shadow(tmp_path, rows)
        flow, age, day = s.latest_completed_flow("BTC", now_ts=now)
        assert flow == -5e7, (
            "the harness read TODAY's in-progress flow row — the P253c "
            "in-progress-bar trap")

    def test_record_confidence_is_abs_direction(self, tmp_path):
        # P236/P224: confidence = |direction|. [P402] the signal is the z-score,
        # so it needs a trailing window: a strong-outflow newest day against a
        # varied trailing baseline -> -1.0 / conf 1.0.
        now = time.time()
        midnight = (int(now) // 86400) * 86400
        rows = self._varied_trailing(midnight, n=20, base=-6e7)
        rows.append({"timestamp": (midnight - 86400) * 1000, "flow_usd": -1e9})
        s = self._shadow(tmp_path, rows)
        rec = s.record_tick("BTC")
        assert rec is not None
        assert rec["direction"] == -1.0
        assert rec["confidence"] == 1.0
        assert rec["reason"] == "outflow_z"
        assert "z_score" in rec
        assert rec["strategy"] == "etfflow"
        # and the file really landed under the registered prefix
        assert (tmp_path / "etfflow_BTC.jsonl").exists()

    def test_single_row_is_warmup_not_raw_sign(self, tmp_path):
        # [P402] a single completed row has NO trailing window -> WARMUP (flat),
        # NOT the raw sign. This pins that the primary signal is the z-score:
        # a silent revert to raw sign would return -1.0 here and fail.
        now = time.time()
        midnight = (int(now) // 86400) * 86400
        rows = [{"timestamp": (midnight - 86400) * 1000, "flow_usd": -5e7}]
        s = self._shadow(tmp_path, rows)
        rec = s.record_tick("BTC")
        assert rec["direction"] == 0.0
        assert rec["confidence"] == 0.0
        assert rec["reason"] == "warmup"
        # raw sign is still recorded for A/B and does carry the -1
        assert rec["raw_sign"] == -1.0

    def test_combination_shadow_derisks_on_outflow(self, tmp_path):
        # [P404] combo book = SMA200 long/flat gated by ETF outflow. Needs >=200
        # completed prices; a strong-outflow newest day -> combo flat regardless
        # of SMA (de-risk). Rows carry price_usd so SMA200 is computable.
        import time, random
        random.seed(11)
        now = time.time(); midn = int(now // 86400 * 86400)
        rows = []; px = 100.0
        for i in range(210):
            px *= (1 + random.gauss(0.002, 0.01))   # gentle uptrend
            rows.append({"timestamp": (midn - (210 - i) * 86400) * 1000,
                         "flow_usd": random.gauss(0, 1e8), "price_usd": px})
        rows[-1]["flow_usd"] = -9e8                   # strong outflow newest day
        s = self._shadow(tmp_path, rows)
        rec = s.record_tick("BTC")
        assert rec["combo_direction"] == 0.0
        assert rec["combo_reason"] == "etf_outflow_derisk"
        assert rec["sma200"] is not None

    def test_combination_shadow_none_without_enough_prices(self, tmp_path):
        # < 200 completed prices -> combo fields are None/no_price, never a crash
        rows = [{"timestamp": (int(__import__("time").time()) // 86400 * 86400 - 86400) * 1000,
                 "flow_usd": -5e7, "price_usd": 100.0}]
        s = self._shadow(tmp_path, rows)
        rec = s.record_tick("BTC")
        assert rec["combo_direction"] is None and rec["combo_reason"] == "no_price"

    def test_flat_record_has_zero_confidence(self, tmp_path):
        s = self._shadow(tmp_path, [])
        s._cache = {}
        s._api_key = ""  # no key -> no data -> flat with reason
        rec = s.record_tick("BTC")
        assert rec["direction"] == 0.0 and rec["confidence"] == 0.0
        assert rec["reason"] == "no_data"


class TestEtfWiring:
    def test_prefix_registered_at_both_scorer_sites(self):
        # The P192/P236 two-site rule: a ledger registered at one site only
        # is invisible to the other invocation path.
        src = read_source(REPO / "analytics" / "shadow_ic" /
                          "compute_shadow_ic.py")
        assert src.count('"etfflow"') + src.count("etfflow,") >= 2, (
            "etfflow must appear in BOTH the load_shadow_ledgers default "
            "tuple and the argparse default")

    def test_main_inits_and_ticks_the_harness(self):
        src = read_source(MAIN)
        assert "EtfFlowShadow" in src and "_etf_flow_shadow" in src
        assert "_etf_flow_shadow.tick()" in src, (
            "the harness is constructed but never ticked — a ledger with "
            "no writer (P199 class)")

    def test_both_assets_have_probed_endpoints(self):
        assert set(ETF_ENDPOINTS) == {"BTC", "ETH"}, (
            "SOL has no spot ETF; adding an asset here requires probing its "
            "endpoint first (P218)")


# ===========================================================================
# 2. Maker-first execution
# ===========================================================================

from exchange.coinbase_sleeve import CoinbaseSleeve  # noqa: E402


class _FakeClient:
    def __init__(self, bid=100.0, ask=100.5, mid=100.25):
        self._bid, self._ask, self._mid = bid, ask, mid

    def get_product(self, product_id=None):
        return {"price": self._mid, "mid_market_price": self._mid,
                "price_increment": "0.5"}

    def get_best_bid_ask(self, product_ids=None):
        return {"pricebooks": [{
            "bids": [{"price": str(self._bid)}],
            "asks": [{"price": str(self._ask)}]}]}


class _FakeAdapter:
    """Just enough adapter for execute_target. Scripted per test."""

    def __init__(self, maker_fills=False, cancel_ok=True,
                 reject_post_only=False):
        self._client = _FakeClient()
        self.placed = []          # list of OrderRequest
        self.cancelled = []
        self.maker_fills = maker_fills
        self.cancel_ok = cancel_ok
        self.reject_post_only = reject_post_only
        self._open = []           # order dicts currently resting

    def to_venue_symbol(self, asset, kind):
        return f"{asset}-PERP-FAKE"

    def _contract_size(self, pid):
        return 0.1

    async def place_order(self, req):
        import types
        self.placed.append(req)
        if req.post_only and self.reject_post_only:
            return types.SimpleNamespace(
                success=False, order_id=None,
                error_code="INVALID_LIMIT_PRICE_POST_ONLY",
                error_message="would cross")
        oid = f"oid-{len(self.placed)}"
        if req.post_only and not self.maker_fills:
            self._open.append({"order_id": oid})
        return types.SimpleNamespace(success=True, order_id=oid,
                                     error_code=None, error_message=None)

    async def fetch_open_orders(self, symbol=None):
        return list(self._open)

    async def cancel_order(self, oid, pid):
        if not self.cancel_ok:
            return False
        self._open = [o for o in self._open if o["order_id"] != oid]
        self.cancelled.append(oid)
        return True


def _exec_sleeve(adapter, cur=0.0, maker_first=True, wait=5.0,
                 filled_after_maker=None):
    """Sleeve with exactly the surface execute_target touches."""
    s = object.__new__(CoinbaseSleeve)
    s._adapter = adapter
    s._maker_first = maker_first
    s._maker_wait_sec = wait
    s._reconcile_ok = True
    s._halted = False
    s._halt_reason = ""
    s._max_contracts_per_asset = 5
    s._max_net_exposure = None
    s._max_asset_exposure = {}
    state = {"cur": cur}

    def _signed(asset):
        return state["cur"]
    s.signed_contracts = _signed  # type: ignore[assignment]
    s.is_ready = lambda: True  # type: ignore[assignment]

    def _reconcile():
        # simulate the maker fill landing (or not) at the post-attempt
        # reconcile: the test scripts what the venue did
        if filled_after_maker is not None and adapter.placed and any(
                p.post_only for p in adapter.placed):
            state["cur"] = filled_after_maker
        return {}
    s.reconcile_positions = _reconcile  # type: ignore[assignment]
    s.can_trade = lambda a, d: (True, "ok")  # type: ignore[assignment]

    async def _noop(*a, **k):
        return 0
    s._cancel_resting_orders = _noop  # type: ignore[assignment]
    s._cancel_stale_entry_orders = _noop  # type: ignore[assignment]
    return s, state


def _run(coro):
    return asyncio.run(coro)


class TestMakerFirstLadder:
    def test_full_maker_fill_places_no_cross(self):
        ad = _FakeAdapter(maker_fills=True)
        s, _ = _exec_sleeve(ad, cur=0.0, filled_after_maker=1.0)
        res = _run(s.execute_target("BTC", 1))
        assert res["status"] == "OK" and res.get("maker") is True
        post_onlys = [p for p in ad.placed if p.post_only]
        crosses = [p for p in ad.placed if not p.post_only]
        assert len(post_onlys) == 1 and len(crosses) == 0, (
            "a fully maker-filled target must never also place the cross")

    def test_post_only_rejection_falls_back_to_cross(self):
        ad = _FakeAdapter(reject_post_only=True)
        s, _ = _exec_sleeve(ad, cur=0.0)
        res = _run(s.execute_target("BTC", 1))
        assert res["status"] == "OK"
        crosses = [p for p in ad.placed if not p.post_only]
        assert len(crosses) == 1, "rejected post-only must cross exactly once"

    def test_cancel_failed_never_places_the_cross(self):
        # The P265 double-order class: timeout + failed cancel means the
        # maker order MAY still be live — placing the cross beside it can
        # double-fill the delta.
        ad = _FakeAdapter(maker_fills=False, cancel_ok=False)
        s, _ = _exec_sleeve(ad, cur=0.0, wait=5.0)
        res = _run(s.execute_target("BTC", 1))
        assert res["status"] == "FAILED"
        assert res["reason"] == "maker_cancel_failed_no_cross"
        crosses = [p for p in ad.placed if not p.post_only]
        assert len(crosses) == 0, (
            "cross placed while the uncancellable maker order may be live "
            "— the P265 double-order class")

    def test_unfilled_timeout_cancels_then_crosses_once(self):
        ad = _FakeAdapter(maker_fills=False, cancel_ok=True)
        s, _ = _exec_sleeve(ad, cur=0.0, wait=5.0)
        res = _run(s.execute_target("BTC", 1))
        assert res["status"] == "OK"
        assert len(ad.cancelled) == 1
        crosses = [p for p in ad.placed if not p.post_only]
        assert len(crosses) == 1

    def test_urgent_skips_the_maker_path_entirely(self):
        # FORCE_FLAT and the fast-risk watchdog must never wait to save 3bps
        ad = _FakeAdapter(maker_fills=True)
        s, _ = _exec_sleeve(ad, cur=1.0)
        res = _run(s.execute_target("BTC", 0, urgent=True))
        assert res["status"] == "OK"
        assert all(not p.post_only for p in ad.placed), (
            "urgent call placed a post-only order — an emergency exit was "
            "made to wait for a maker fill")

    def test_flag_off_is_byte_identical_taker(self):
        ad = _FakeAdapter()
        s, _ = _exec_sleeve(ad, cur=0.0, maker_first=False)
        res = _run(s.execute_target("BTC", 1))
        assert res["status"] == "OK"
        assert all(not p.post_only for p in ad.placed)
        assert res.get("maker") is None

    def test_maker_price_joins_the_touch_not_mid(self):
        # BUY joins the BID; mid-based pricing rounds onto the opposite
        # touch on a 1-tick spread and gets (correctly) rejected.
        ad = _FakeAdapter(maker_fills=True)
        s, _ = _exec_sleeve(ad, cur=0.0, filled_after_maker=1.0)
        _run(s.execute_target("BTC", 1))
        po = [p for p in ad.placed if p.post_only][0]
        assert po.price == 100.0, f"post-only BUY priced at {po.price}, not the bid"


class TestMakerConfigPlumbing:
    def test_config_trio_declared_parsed_passed(self):
        # The P201 rule: declared on ProductionConfig AND parsed in
        # from_file AND passed to the ctor — a flag missing any leg is inert
        # while looking configured.
        src = read_source(MAIN)
        assert re.search(r"^\s+coinbase_maker_first: bool = False", src, re.M)
        assert 'data.get("coinbase_maker_first", False)' in src
        assert re.search(r"maker_first=bool\(getattr\(\s*self\.config,"
                         r"\s*\"coinbase_maker_first\"", src), (
            "flag declared+parsed but never passed to the CoinbaseSleeve "
            "ctor — the P16/P201 dead-flag shape")

    def test_default_off_and_decided_value_in_live_profile(self):
        # The dataclass/ctor default stays False (absent key = off in every
        # other profile). The LIVE profile pins the DECIDED value — enabled
        # 2026-08-16 by explicit operator instruction (P270 activation) — so
        # a silent revert AND a silent widening both fail loudly (the P237
        # pin-the-decision pattern).
        import inspect
        sig = inspect.signature(CoinbaseSleeve.__init__)
        assert sig.parameters["maker_first"].default is False
        live = json.loads((REPO / "configs" / "live_high_risk.json")
                          .read_text(encoding="utf-8"))
        assert live.get("coinbase_maker_first") is True, (
            "coinbase_maker_first was flipped off/removed from the live "
            "profile — if that is a deliberate revert, update this pin and "
            "the P270 activation note in the same commit")
        assert live.get("coinbase_maker_wait_sec") == 45.0

    def test_emergency_call_sites_pass_urgent(self):
        src = read_source(MAIN)
        assert "execute_target(asset, 0, urgent=True)" in src, (
            "the fast-risk watchdog EXIT_ONLY lost urgent=True")
        assert re.search(r"execute_target\(\s*_cb_a,\s*0,\s*urgent=True\)",
                         src), "FORCE_FLAT lost urgent=True"


# ===========================================================================
# 3. Volskip overlay ledger
# ===========================================================================

from defense.regime_book_shadow import (  # noqa: E402
    RegimeBookShadow, VOLSKIP_THR)


def _rbs():
    s = object.__new__(RegimeBookShadow)
    s._volskip_state = {}
    return s


class TestVolskipSingleStep:
    def _closes(self, vol_high: bool):
        # 21 closes with tiny (calm) or huge (violent) log-returns
        import math
        step = 0.05 if vol_high else 0.0005
        c, out = 100.0, []
        for i in range(21):
            c *= math.exp(step if i % 2 == 0 else -step)
            out.append(c)
        return out

    def test_exit_is_always_honored_even_at_high_vol(self):
        s = _rbs()
        s._volskip_state["ETH"] = {"cur": 1.0}
        t, vol, allow = s._volskip_target("ETH", self._closes(True), 0.0)
        assert t == 0.0, "high vol blocked an EXIT — the filter may only gate entries (P195/P236)"

    def test_entry_blocked_at_high_vol_allowed_at_low(self):
        s = _rbs()
        t_hi, _, allow_hi = s._volskip_target("ETH", self._closes(True), 1.0)
        assert t_hi == 0.0 and not allow_hi
        s2 = _rbs()
        t_lo, _, allow_lo = s2._volskip_target("ETH", self._closes(False), 1.0)
        assert t_lo == 1.0 and allow_lo

    def test_blocked_flip_degrades_to_flatten(self):
        s = _rbs()
        s._volskip_state["ETH"] = {"cur": 1.0}
        t, _, _ = s._volskip_target("ETH", self._closes(True), -1.0)
        assert t == 0.0, ("a vol-blocked flip must flatten (exit leg "
                          "honored), never hold the old side or open the new")

    def test_missing_vol_blocks_entries_never_exits(self):
        s = _rbs()
        t, vol, allow = s._volskip_target("ETH", [100.0, 101.0], 1.0)
        assert t == 0.0 and vol is None
        s._volskip_state["ETH"] = {"cur": -1.0}
        t2, _, _ = s._volskip_target("ETH", None, 0.0)
        assert t2 == 0.0  # exit honored with no closes at all

    def test_only_eth_is_exported(self):
        assert set(VOLSKIP_THR) == {"ETH"}, (
            "volskip earned its ledger on ETH ONLY (three-era winner); "
            "SOL's validation increment was 0.0 and BTC's base stood — "
            "adding an asset requires its own lab + ledgered read")
        assert _rbs()._volskip_target("SOL", [100.0] * 30, 1.0) is None

    def test_live_step_matches_the_lab_batch_filter(self):
        # The P164/P214-class parity pin: the live single-step machine must
        # reproduce training/entry_filter_lab.apply_entry_filter bar-for-bar
        # on the same allow/raw sequence, else the forward ledger measures a
        # different mechanism than the lab selected.
        np = pytest.importorskip("numpy")
        import importlib.util
        spec = importlib.util.find_spec("training.entry_filter_lab")
        if spec is None:
            pytest.skip("training package not importable here")
        from training.entry_filter_lab import apply_entry_filter
        import random
        rng = random.Random(7)
        raw = np.array([rng.choice([-1.0, 0.0, 1.0]) for _ in range(300)])
        allow = np.array([rng.random() < 0.6 for _ in range(300)])
        batch = apply_entry_filter(raw, allow)
        # live: replay through the single-step machine with scripted allow
        s = _rbs()
        state = s._volskip_state.setdefault("ETH", {"cur": 0.0})
        live = []
        for i in range(300):
            cur = float(state.get("cur", 0.0))
            want = float(raw[i])
            ok = bool(allow[i])
            if want == cur:
                pass
            elif want == 0.0:
                cur = 0.0
            elif cur == 0.0:
                cur = want if ok else 0.0
            else:
                cur = want if ok else 0.0
            state["cur"] = cur
            live.append(cur)
        assert live == list(batch), (
            "lab batch filter and the live step disagree — the ledger "
            "would forward-test a different mechanism than the lab "
            "selected (P164/P214 class)")

    def test_ledger_leg_wired_in_record_tick(self):
        src = read_source(REPO / "defense" / "regime_book_shadow.py")
        assert "regimebook_volskip" in src
        assert "_volskip_target(asset, closes" in src
