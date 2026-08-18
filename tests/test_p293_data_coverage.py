"""
[P293] Data-coverage repair batch.

The defects this pins are all ONE SHAPE: a consumer wired to a cache that
no producer ever fills, or a neutral default that is indistinguishable from
a measurement. Every test below is written to go RED if its fix is reverted;
the source-level ones exist because the behavioural half needs a live tick
loop that a unit test cannot build.

Structure:
  TestFetchDriversExist          - FRED/LunarCrush/new feeds are DRIVEN
  TestProvenanceIsFalsifiable    - source_status can report "not available"
  TestDeribitFeed                - PCR/DVOL parsing, and SOL absence
  TestExchangeNetflow            - netflow parsing, sign convention, SOL absence
  TestFearGreedHistory           - real z-score, refusal on thin history
  TestOptionsExternalPcr         - external PCR reuses the agent's own scoring
  TestConfigTrioDefaultsOff      - every new flag is declared, parsed, OFF
  TestGciMacroPath               - FRED-backed indicators, ETF refuses to mock
"""

import ast
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAIN = REPO / "main.py"


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="ignore")


def _strip_comments(src: str) -> str:
    """Blank out # comments so a test cannot pass on its own explanation.

    The P177 trap: these fixes are documented in comments that quote the
    very symbols being asserted, so a naive substring scan matches the
    prose rather than the code.
    """
    out = []
    for line in src.splitlines():
        in_s = False
        quote = ""
        cut = None
        for i, ch in enumerate(line):
            if in_s:
                if ch == quote and (i == 0 or line[i - 1] != "\\"):
                    in_s = False
            elif ch in "\"'":
                in_s = True
                quote = ch
            elif ch == "#":
                cut = i
                break
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


# =============================================================================
# 1. The producers are actually driven
# =============================================================================

class TestFetchDriversExist:
    """FRED and LunarCrush had valid keys, non-mock init, and NO caller of
    fetch()/start() anywhere in the tree — so every consumer read a cache
    that was never filled. These pin the drivers into existence."""

    @pytest.mark.parametrize("feed_attr,method", [
        ("self.fred_feed", "fetch_if_stale"),
        ("self.lunarcrush_feed", "fetch_if_stale"),
        ("self._deribit_feed", "fetch_if_stale"),
        ("self._exchange_netflow_feed", "fetch_if_stale"),
        ("self._fng_history", "refresh_if_stale"),
    ])
    def test_feed_is_driven_from_the_tick(self, feed_attr, method):
        src = _strip_comments(_src(MAIN))
        assert f"{feed_attr}.{method}()" in src, (
            f"{feed_attr}.{method}() is not called from main.py. A feed whose "
            f"fetch is never driven serves its neutral defaults forever, which "
            f"is exactly the defect P293 repairs."
        )

    def test_drivers_are_registered_with_a_feed_name(self):
        """A task appended without a name misaligns the failure reporting,
        because failures are attributed by index against _feed_names."""
        src = _strip_comments(_src(MAIN))
        block = src[src.index("_feed_tasks = []"): src.index("if _feed_tasks:")]
        assert block.count("_feed_tasks.append(") == block.count("_feed_names.append("), (
            "every _feed_tasks.append must have a matching _feed_names.append — "
            "they are zipped by index when reporting failures"
        )

    @pytest.mark.parametrize("mod,cls", [
        ("data_mgmt.feeds.fred_feed", "FREDFeed"),
        ("data_mgmt.feeds.lunarcrush_feed", "LunarCrushFeed"),
    ])
    def test_fetch_if_stale_exists_and_throttles(self, mod, cls):
        import importlib
        m = importlib.import_module(mod)
        klass = getattr(m, cls, None)
        if klass is None:  # tolerate a class rename, find the feed class
            klass = next(
                v for k, v in vars(m).items()
                if isinstance(v, type) and k.endswith("Feed")
            )
        assert hasattr(klass, "fetch_if_stale")
        assert hasattr(klass, "cache_age_sec")

        obj = klass.__new__(klass)
        obj._last_fetch_time = None
        assert obj.cache_age_sec() is None, (
            "never-fetched must be None, not a number — 'never fetched' and "
            "'fetched long ago' have different fixes"
        )
        obj._last_fetch_time = datetime.now(timezone.utc) - timedelta(seconds=30)
        age = obj.cache_age_sec()
        assert 25 <= age <= 60

    def test_cache_age_survives_a_naive_timestamp(self):
        """P40/P97: a naive datetime here would raise inside whatever try
        block the caller happens to be in, and be swallowed."""
        from data_mgmt.feeds.fred_feed import FREDFeed
        obj = FREDFeed.__new__(FREDFeed)
        obj._last_fetch_time = datetime.now() - timedelta(seconds=10)  # naive
        assert obj.cache_age_sec() is not None


# =============================================================================
# 2. source_status must be able to say "no"
# =============================================================================

class TestProvenanceIsFalsifiable:
    """The adapter always returns a populated dict, so the old
    `if adapter_context.get("macro")` test was True unconditionally and
    marked hardcoded defaults as 'available'."""

    def test_feed_provenance_reports_absence(self):
        from signals.macro_crowd_adapter import MacroCrowdAdapter

        class _Empty:
            def get_latest(self):
                return None

        class _Full:
            def get_latest(self):
                return object()

        a = MacroCrowdAdapter(fred_feed=_Empty(), coinglass_feed=_Full())
        prov = a.feed_provenance()
        assert prov["fred"] is False, "an empty cache must report False"
        assert prov["coinglass"] is True
        assert prov["lunarcrush"] is False, "a missing feed is not 'available'"

    def test_provenance_survives_a_raising_feed(self):
        from signals.macro_crowd_adapter import MacroCrowdAdapter

        class _Boom:
            def get_latest(self):
                raise RuntimeError("venue down")

        a = MacroCrowdAdapter(fred_feed=_Boom())
        assert a.feed_provenance()["fred"] is False, (
            "a feed whose accessor raises cannot be certified as holding data"
        )

    def test_main_derives_status_from_provenance_not_dict_shape(self):
        src = _strip_comments(_src(MAIN))
        assert "feed_provenance()" in src
        assert '"defaults_only"' in src, (
            "there must be a status distinct from 'available' for the "
            "defaults-only case, or the flag remains a constant"
        )


# =============================================================================
# 3. Deribit
# =============================================================================

class TestDeribitFeed:
    def test_sol_is_not_in_supported_currencies(self):
        """Deribit lists SOL as a currency but carries NO SOL options book
        (measured: 0 instruments). Adding it would produce a fabricated PCR."""
        from data_mgmt.feeds.deribit_feed import SUPPORTED_OPTION_CURRENCIES
        assert "SOL" not in SUPPORTED_OPTION_CURRENCIES
        assert set(SUPPORTED_OPTION_CURRENCIES) == {"BTC", "ETH"}

    def test_metrics_absent_is_none_never_neutral(self):
        from data_mgmt.feeds.deribit_feed import DeribitFeed
        f = DeribitFeed(mock_mode=True)
        assert f.get_options_metrics("SOL") is None, (
            "no data must be None. A neutral 1.0 for SOL is indistinguishable "
            "from a measured balanced book — the P218 defect"
        )

    def test_pcr_is_computed_from_the_book(self):
        from data_mgmt.feeds.deribit_feed import DeribitOptionsMetrics
        m = DeribitOptionsMetrics(
            currency="BTC", put_call_ratio_oi=0.556,
            total_oi_calls=216162.0, total_oi_puts=120179.0,
            instrument_count=782,
        )
        assert m.is_usable()
        assert abs(m.put_call_ratio_oi - (120179.0 / 216162.0)) < 0.01

    def test_zero_call_oi_gives_none_not_zero(self):
        """Dividing by zero call OI must yield 'undefined', not 0.0 — a PCR
        of 0.0 reads as an extreme call skew."""
        from data_mgmt.feeds.deribit_feed import DeribitOptionsMetrics
        m = DeribitOptionsMetrics(currency="BTC", total_oi_calls=0.0,
                                  total_oi_puts=500.0, instrument_count=3)
        assert m.put_call_ratio_oi is None
        assert not m.is_usable()

    def test_as_float_rejects_nan_and_inf(self):
        from data_mgmt.feeds.deribit_feed import _as_float
        assert _as_float(float("nan")) == 0.0
        assert _as_float(float("inf")) == 0.0
        assert _as_float(None, default=None) is None
        assert _as_float("1.5") == 1.5


# =============================================================================
# 4. Exchange netflow
# =============================================================================

class TestExchangeNetflow:
    def test_sol_absent_by_measurement(self):
        from data_mgmt.feeds.exchange_netflow_feed import SUPPORTED_SYMBOLS
        assert "SOL" not in SUPPORTED_SYMBOLS, (
            "the endpoint returns ZERO SOL exchanges; including it would "
            "emit a fabricated 0.0 netflow indistinguishable from flat flows"
        )

    def test_sign_convention_is_positive_equals_inflow(self):
        """The whole point of the explicit convention: the legacy consumer
        in main.py uses the OPPOSITE sign, so a silent bridge would invert
        the signal."""
        from data_mgmt.feeds.exchange_netflow_feed import ExchangeNetflowMetrics
        m = ExchangeNetflowMetrics(
            symbol="BTC", total_balance_coins=2_510_258.0,
            netflow_coins_1d=4046.0, exchange_count=21,
        )
        assert m.netflow_coins_1d > 0
        assert m.to_dict()["sign_convention"] == "positive=inflow_to_exchanges=bearish"
        assert m.balance_pct_1d() > 0

    def test_usd_requires_a_usable_price(self):
        from data_mgmt.feeds.exchange_netflow_feed import ExchangeNetflowMetrics
        m = ExchangeNetflowMetrics(symbol="BTC", total_balance_coins=100.0,
                                   netflow_coins_1d=10.0, exchange_count=2)
        assert m.netflow_usd_1d(None) is None
        assert m.netflow_usd_1d(0) is None
        assert m.netflow_usd_1d(-5) is None
        assert m.netflow_usd_1d(64000) == pytest.approx(640000.0)

    def test_empty_response_is_not_usable(self):
        from data_mgmt.feeds.exchange_netflow_feed import ExchangeNetflowMetrics
        m = ExchangeNetflowMetrics(symbol="ETH")
        assert not m.is_usable()
        assert m.balance_pct_1d() is None

    def test_consumer_sign_is_corrected_at_source(self):
        """[P293b] The producer (onchain_feed) is positive=INFLOW; the
        consumer read positive as OUTFLOW and emitted +1 (bullish) for it —
        the inverse of its own comment. They disagreed since both were
        written, invisible only because the producer always returned 0.0.

        Pins the CORRECTED predicate: inflow (positive) -> bearish (-1).
        """
        src = _strip_comments(_src(MAIN))
        assert "-1.0 if _exf > 1e6 else (1.0 if _exf < -1e6 else 0.0)" in src, (
            "flow_direction must map POSITIVE exchange_flow (= inflow to "
            "exchanges = sell pressure) to -1.0. The pre-P293b form had this "
            "inverted."
        )
        assert "-1.0 if _exf < -1e6 else (1.0 if _exf > 1e6 else 0.0)" not in src, (
            "the inverted predicate is back"
        )

    def test_bridge_does_not_re_invert(self):
        """With the consumer corrected, a surviving negation in the bridge
        would re-open the disagreement from the other side."""
        src = _strip_comments(_src(MAIN))
        assert "-_enf_usd if _enf_usd is not None else 0.0" not in src, (
            "the bridge must NOT negate now that producer and consumer share "
            "one convention"
        )
        assert "_enf_usd if _enf_usd is not None else 0.0" in src

    def test_feed_direction_agrees_with_the_legacy_predicate(self):
        """The real consistency check: the same underlying event (coins
        moving ONTO exchanges) must read bearish through BOTH paths."""
        from data_mgmt.feeds.exchange_netflow_feed import ExchangeNetflowMetrics

        def legacy_flow_direction(exf: float) -> float:
            # mirrors the corrected main.py predicate
            return -1.0 if exf > 1e6 else (1.0 if exf < -1e6 else 0.0)

        inflow = ExchangeNetflowMetrics(
            symbol="BTC", total_balance_coins=2_510_258.0,
            netflow_coins_1d=4046.0, exchange_count=21)
        usd = inflow.netflow_usd_1d(64_000)
        assert usd > 1e6
        assert legacy_flow_direction(usd) == -1.0, "inflow must read BEARISH"
        # and the feed-native direction must agree
        assert inflow.balance_pct_1d() > 0.0005

        outflow = ExchangeNetflowMetrics(
            symbol="ETH", total_balance_coins=12_065_648.0,
            netflow_coins_1d=-13_594.0, exchange_count=20)
        usd_o = outflow.netflow_usd_1d(1_900)
        assert usd_o < -1e6
        assert legacy_flow_direction(usd_o) == 1.0, "outflow must read BULLISH"


# =============================================================================
# 5. Fear & Greed history / real z-score
# =============================================================================

class TestFearGreedHistory:
    def _hist(self, tmp_path, values):
        from data_mgmt.feeds.fear_greed_history import FearGreedHistory
        h = FearGreedHistory(data_dir=str(tmp_path))
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i, v in enumerate(values):
            h._series[(base + timedelta(days=i)).strftime("%Y-%m-%d")] = float(v)
        return h

    def test_refuses_below_min_samples(self, tmp_path):
        h = self._hist(tmp_path, [30] * 10)
        s = h.score(31.0)
        assert s.zscore is None, "a thin window must not yield a confident z"
        assert "insufficient_history" in s.reason
        assert not s.usable

    def test_degenerate_stdev_refuses(self, tmp_path):
        """A constant window has no scale — dividing by it is the
        Sharpe-of-a-constant defect (P265g)."""
        h = self._hist(tmp_path, [50] * 100)
        s = h.score(60.0)
        assert s.zscore is None
        assert s.reason == "degenerate_stdev"

    def test_real_zscore_differs_from_the_linear_form(self, tmp_path):
        """The live finding: at F&G=31 the linear form says -1.14
        (confidently bearish) while a year averaging ~27 puts 31 ABOVE
        average. If these ever agree, the fix is pointless."""
        h = self._hist(tmp_path, [20 + (i % 15) for i in range(365)])
        s = h.score(31.0)
        assert s.zscore is not None
        linear = (31.0 - 50.0) / 50.0 * 3.0
        assert linear < 0, "sanity: the linear form calls 31 bearish"
        assert s.zscore > linear, (
            "against a fearful year, 31 must score higher than the fixed "
            "linear rescale — that difference is the entire point"
        )
        assert 0.0 <= s.percentile <= 1.0

    def test_non_numeric_value_refuses(self, tmp_path):
        h = self._hist(tmp_path, [20 + (i % 15) for i in range(365)])
        assert h.score("banana").zscore is None

    def test_history_round_trips_through_disk(self, tmp_path):
        from data_mgmt.feeds.fear_greed_history import FearGreedHistory
        h = self._hist(tmp_path, [20 + (i % 15) for i in range(120)])
        h._last_refresh = datetime.now(timezone.utc)
        h._persist()
        h2 = FearGreedHistory(data_dir=str(tmp_path))
        assert h2.sample_count() == 120, (
            "history must survive a restart, or the z-score silently "
            "degrades to 'insufficient' on every deploy (P154)"
        )

    def test_corrupt_state_degrades_to_cold_start(self, tmp_path):
        from data_mgmt.feeds.fear_greed_history import FearGreedHistory
        (tmp_path / "fear_greed_history.json").write_text("{not json", encoding="utf-8")
        h = FearGreedHistory(data_dir=str(tmp_path))
        assert h.sample_count() == 0

    def test_parse_payload_rejects_out_of_range(self):
        from data_mgmt.feeds.fear_greed_history import FearGreedHistory
        rows = FearGreedHistory._parse_payload({"data": [
            {"value": "31", "timestamp": "1786924800"},
            {"value": "900", "timestamp": "1786838400"},   # impossible
            {"value": "bad", "timestamp": "1786752000"},
            {"value": "50"},                                # no timestamp
        ]})
        assert len(rows) == 1 and rows[0][1] == 31.0

    def test_main_logs_both_forms_and_defaults_to_linear(self):
        src = _strip_comments(_src(MAIN))
        assert 'market_data["sentiment_zscore_linear"]' in src
        assert 'market_data["sentiment_zscore_historical"]' in src
        assert 'sentiment_zscore_source' in src, (
            "which form produced the live value must be recorded, or the "
            "ledger cannot be read across the switch"
        )


# =============================================================================
# 6. Options external PCR
# =============================================================================

class TestOptionsExternalPcr:
    def _agent(self):
        from agents.options_sentiment_agent import OptionsSentimentAgent
        return OptionsSentimentAgent(api_key="test")

    def test_unsupported_asset_returns_none(self):
        assert self._agent().apply_external_pcr("SOL", 0.9) is None

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), None, "x"])
    def test_unusable_pcr_returns_none_not_neutral(self, bad):
        assert self._agent().apply_external_pcr("BTC", bad) is None, (
            "returning a neutral dict would overwrite a good reading with a "
            "fabricated balance"
        )

    def test_real_pcr_produces_a_non_neutral_signal(self):
        """Live values (BTC 0.556/0.778) must move the agent off the
        0.0/0.0 it has emitted since the CoinGlass paths 404'd."""
        r = self._agent().apply_external_pcr("BTC", 0.556, 0.778)
        assert r is not None
        assert r["put_call_ratio"] == pytest.approx(0.556, abs=1e-3)
        assert r["confidence"] > 0.0
        assert r["source"] == "options_sentiment_deribit"

    def test_conversion_is_not_reimplemented_at_the_call_site(self):
        """P172: the PCR->signal rules must have ONE home. main.py must call
        the agent, not roll its own thresholds."""
        src = _strip_comments(_src(MAIN))
        assert "apply_external_pcr(" in src
        assert "_compute_short_confirmation" not in src, (
            "main.py must not reimplement the options scoring"
        )


# =============================================================================
# 7. Config trio — declared, parsed, and OFF
# =============================================================================

P293_FLAGS = [
    "sentiment_zscore_mode",
    "options_use_deribit",
    "exchange_netflow_to_flow_agent",
    "dvol_to_market_data",
    "macro_gci_live",
]


class TestConfigTrioDefaultsOff:
    @pytest.mark.parametrize("flag", P293_FLAGS)
    def test_declared_and_parsed(self, flag):
        """P201: a JSON key read via getattr but never declared+parsed does
        NOTHING, and the default happening to match hides it."""
        src = _src(MAIN)
        assert re.search(rf"^\s+{flag}:\s", src, re.M), f"{flag} not declared"
        assert re.search(rf"{flag}=", src), f"{flag} not parsed in from_file"
        assert f'data.get("{flag}"' in src, f"{flag} not read from JSON"

    def test_defaults_preserve_current_behaviour(self):
        from main import ProductionConfig
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text("{}", encoding="utf-8")
            c = ProductionConfig.from_file(p)
        assert c.sentiment_zscore_mode == "linear"
        assert c.options_use_deribit is False
        assert c.exchange_netflow_to_flow_agent is False
        assert c.dvol_to_market_data is False
        assert c.macro_gci_live is False

    def test_unknown_zscore_mode_falls_back_to_linear(self):
        """An unrecognised mode must never silently switch a live signal."""
        from main import ProductionConfig
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps({"sentiment_zscore_mode": "banana"}), encoding="utf-8")
            assert ProductionConfig.from_file(p).sentiment_zscore_mode == "linear"
            p.write_text(json.dumps({"sentiment_zscore_mode": "HISTORICAL"}), encoding="utf-8")
            assert ProductionConfig.from_file(p).sentiment_zscore_mode == "historical"

    @pytest.mark.parametrize("flag", P293_FLAGS)
    def test_not_enabled_in_the_live_profile(self, flag):
        """Enabling any of these changes what a live agent emits. Adding the
        key to the live profile IS the activation action (P141) and must be
        a recorded decision, not a side effect of this batch."""
        prof = REPO / "configs" / "live_high_risk.json"
        if not prof.exists():
            pytest.skip("live profile not present")
        data = json.loads(prof.read_text(encoding="utf-8-sig"))
        # [P298] Flipped by explicit operator instruction ("make a plan on enabling all items, i don't want to wait"). The pin now asserts the DECIDED value rather than OFF, so a silent revert fails too - either direction is a live-money change (the P237/P270 pattern).
        #
        # dvol_to_market_data is the one that did NOT flip, and for a reason
        # worth keeping: Deribit publishes DVOL as an INDEX LEVEL (BTC 34.5)
        # while the constitution aliases dvol -> dvol_zscore and fires
        # EXTREME_DVOL at >= 5.0, which is not in the sleeve's HOLD set. It
        # would have flattened the book on every tick, permanently.
        DECIDED = {
            "sentiment_zscore_mode": "historical",
            "options_use_deribit": True,
            "exchange_netflow_to_flow_agent": True,
            "macro_gci_live": True,
            # [P306] dvol_to_market_data flipped LAST, and only after the
            # units bug this comment block describes was fixed: what is
            # published is now a z-score computed against a fetched trailing
            # year (BTC 34.13 -> z -1.38), not the Deribit index level.
            "dvol_to_market_data": True,
        }
        if flag == "dvol_to_market_data" and data.get(flag):
            # the precondition, asserted with the decision (P306): the flag
            # being on while the raw LEVEL is published is the failure mode.
            main_src = (REPO / "main.py").read_text(encoding="utf-8")
            assert 'market_data["dvol"] = float(_dvz)' in main_src
            assert 'market_data["dvol"] = float(_drb_m.dvol)' not in main_src
        if flag in DECIDED:
            assert data.get(flag) == DECIDED[flag], (
                f"{flag} is not at its decided value {DECIDED[flag]!r} — a "
                f"silent revert is as much a live-money change as the flip was"
            )
        else:
            assert flag not in data, (
                f"{flag} appeared in the live profile — that is an activation "
                f"decision and needs its own P-entry"
            )


# =============================================================================
# 8. GCI macro path
# =============================================================================

class TestGciMacroPath:
    def test_fred_series_map_omits_gold(self):
        """FRED's daily gold series is discontinued. An unmapped ticker must
        stay unmapped rather than get a plausible substitute."""
        from data_mgmt.feeds.fred_macro_series import FRED_SERIES_MAP, fred_series_for
        assert "GOLD" not in FRED_SERIES_MAP
        assert fred_series_for("GOLD") is None
        assert fred_series_for("VIX") == "VIXCLS"

    def test_dxy_is_labelled_a_proxy(self):
        """DTWEXBGS is a different basket at a different level (~119 vs
        ~104). Substituting it silently would mislead anyone reading the
        level; only its z-score is comparable."""
        from data_mgmt.feeds.fred_macro_series import is_proxy
        assert is_proxy("DXY") is True
        assert is_proxy("VIX") is False

    def test_unmapped_ticker_returns_empty_not_zero(self):
        from data_mgmt.feeds.fred_macro_series import fetch_daily_closes
        assert fetch_daily_closes("GOLD") == []

    def test_missing_key_returns_empty(self, monkeypatch):
        from data_mgmt.feeds import fred_macro_series as fms
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        assert fms.fetch_daily_closes("VIX", api_key="") == []

    def test_gci_has_a_throttled_updater(self):
        """update_async() had NO caller in the whole tree — the live log has
        never once emitted 'GlobalContext updated'."""
        from data_mgmt.global_context_informer import GlobalContextInformer
        assert hasattr(GlobalContextInformer, "update_if_stale")

    def test_etf_refuses_rather_than_mocking(self):
        """Driving the updater with a mock ETF flow would inject invented
        fund flows into macro_regime."""
        src = _strip_comments(_src(REPO / "data_mgmt" / "global_context_informer.py"))
        m = re.search(
            r"def _fetch_etf_data_sync.*?(?=\n    def )", src, re.S
        )
        assert m, "_fetch_etf_data_sync not found"
        head = m.group(0).split("try:")[0]
        assert "_generate_mock_flow" not in head, (
            "the yfinance-absent branch must REFUSE (return None), not "
            "fabricate a flow"
        )

    def test_gci_driver_is_gated(self):
        """The leverage-cap map is two-directional (CRISIS 1.5 LOOSENS),
        so driving GCI is not a one-way safety fix."""
        src = _strip_comments(_src(MAIN))
        assert 'getattr(self.config, "macro_gci_live", False)' in src
        assert "update_if_stale()" in src


# =============================================================================
# 9. [P293b] Quota exhaustion, restart re-spend, and real ETF flows
# =============================================================================

class TestCryptoPanicQuotaVsRateLimit:
    """A MONTHLY QUOTA is not a rate limit. CryptoPanic returns 429 with no
    Retry-After and a body saying "API monthly quota exceeded", so the 900s
    default had the engine re-asking every 15 min for the rest of the month."""

    def test_detects_the_real_quota_body(self):
        from data_mgmt.feeds.cryptopanic_feed import _is_monthly_quota_error
        real = ('{"status":"api_error","info":"API monthly quota exceeded - '
                'Upgrade your API plan: /developers/api/plans/"}')
        assert _is_monthly_quota_error(real) is True

    @pytest.mark.parametrize("body", [
        '{"detail":"Request was throttled. Expected available in 30 seconds."}',
        "too many requests",
        "",
        None,
    ])
    def test_transient_429s_are_not_treated_as_monthly(self, body):
        """Fail direction: retry sooner than needed, NEVER go dark by
        mistake for weeks."""
        from data_mgmt.feeds.cryptopanic_feed import _is_monthly_quota_error
        assert _is_monthly_quota_error(body) is False

    def test_backoff_target_is_next_month_utc(self):
        from data_mgmt.feeds.cryptopanic_feed import _next_month_start_utc
        assert _next_month_start_utc(
            datetime(2026, 8, 17, 14, 30, tzinfo=timezone.utc)
        ) == datetime(2026, 9, 1, tzinfo=timezone.utc)

    def test_december_rolls_the_year(self):
        from data_mgmt.feeds.cryptopanic_feed import _next_month_start_utc
        assert _next_month_start_utc(
            datetime(2026, 12, 31, 23, 59, tzinfo=timezone.utc)
        ) == datetime(2027, 1, 1, tzinfo=timezone.utc)

    def test_quota_branch_is_wired_into_the_429_handler(self):
        src = _strip_comments(_src(REPO / "data_mgmt" / "feeds" / "cryptopanic_feed.py"))
        assert "_is_monthly_quota_error(_body)" in src
        assert "_next_month_start_utc(" in src


class TestSessionIdentifiesItself:
    """Cloudflare-fronted vendors reject clients by signature: probed live
    from the server, the default stdlib UA gets HTTP 403 'error code: 1010'
    while any ordinary UA reaches the API."""

    def test_default_user_agent_is_set(self):
        from data_mgmt.feeds._http import DEFAULT_USER_AGENT
        assert DEFAULT_USER_AGENT and "hmats" in DEFAULT_USER_AGENT.lower()

    def test_create_session_injects_it(self):
        import asyncio
        from data_mgmt.feeds._http import create_session, DEFAULT_USER_AGENT

        async def _go():
            s = create_session()
            try:
                return dict(s.headers)
            finally:
                await s.close()

        headers = asyncio.run(_go())
        assert headers.get("User-Agent") == DEFAULT_USER_AGENT

    def test_caller_supplied_user_agent_wins(self):
        import asyncio
        from data_mgmt.feeds._http import create_session

        async def _go():
            s = create_session(headers={"User-Agent": "custom/1.0"})
            try:
                return dict(s.headers)
            finally:
                await s.close()

        assert asyncio.run(_go()).get("User-Agent") == "custom/1.0"


class TestQuotaSurvivesRestart:
    """The measured cause of exhausting a 100/month account in nine days:
    both CryptoCompare feeds kept their THROTTLE CLOCK in RAM, so each of
    the 14 process starts re-spent calls. P253d persisted the backoff and
    left the throttle one line above it in memory."""

    def test_onchain_restores_clock_and_payload(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        import importlib
        import data_mgmt.feeds.cryptocompare_onchain as cco
        importlib.reload(cco)
        Feed = next(v for k, v in vars(cco).items()
                    if isinstance(v, type) and "OnChain" in k and "Data" not in k)
        f = Feed(api_key="k")
        f._data["BTC"] = cco.CryptoCompareOnChainData(
            symbol="BTC", large_transaction_count=1234, is_mock=False)
        f._last_fetch_time = __import__("time").time()
        f._persist_backoff()

        g = Feed(api_key="k")
        assert g._last_fetch_time > 0, "throttle clock must survive a restart"
        assert "BTC" in g._data, (
            "the payload must come back too — restoring the clock alone "
            "leaves `and self._data` False and the persistence decorative"
        )
        assert g._data["BTC"].large_transaction_count == 1234

    def test_onchain_never_persists_mock_rows(self, tmp_path, monkeypatch):
        """A restored mock would survive as if it were a reading (P2)."""
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        import importlib
        import data_mgmt.feeds.cryptocompare_onchain as cco
        importlib.reload(cco)
        Feed = next(v for k, v in vars(cco).items()
                    if isinstance(v, type) and "OnChain" in k and "Data" not in k)
        f = Feed(api_key="k")
        f._data["ETH"] = cco.CryptoCompareOnChainData(symbol="ETH", is_mock=True)
        f._last_fetch_time = __import__("time").time()
        f._persist_backoff()
        assert "ETH" not in Feed(api_key="k")._data

    def test_news_cache_survives_restart(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        import importlib, time as _t
        import data_mgmt.feeds.cryptocompare_news_feed as ccn
        importlib.reload(ccn)
        f = ccn.CCNewsFeed(api_key="k")
        f._cache["BTC"] = (_t.time(), [ccn.CCNewsItem(
            id=1, title="t", body="b", url="u", source="s",
            published_at=datetime.now(timezone.utc),
            categories=["BTC"], sentiment="POSITIVE")])
        f._persist_state()

        g = ccn.CCNewsFeed(api_key="k")
        assert "BTC" in g._cache
        ts, items = g._cache["BTC"]
        assert len(items) == 1 and items[0].title == "t"
        assert (_t.time() - ts) < ccn.MIN_FETCH_INTERVAL, (
            "the restored timestamp must be able to bind the 12h throttle"
        )

    @pytest.mark.parametrize("mod,success_marker,fn", [
        ("data_mgmt/feeds/cryptocompare_onchain.py",
         "self._fetch_count += 1", "self._persist_backoff()"),
        ("data_mgmt/feeds/cryptocompare_news_feed.py",
         "1 call served", "self._persist_state()"),
    ])
    def test_persist_runs_on_success_not_only_on_429(self, mod, success_marker, fn):
        """Persisting only on the rate-limit path means a healthy fetch
        followed by a restart re-spends quota.

        NOTE this pin was rewritten after a falsification probe: the first
        version asserted `src.count(fn) >= 2`, which stayed GREEN when the
        success-path call was deleted, because the two rate-limit call sites
        already satisfied the count. A count is not a location — assert the
        call happens inside the SUCCESS block (P174/P238).
        """
        src = _strip_comments(_src(REPO / mod))
        i = src.find(success_marker)
        assert i > 0, f"success marker {success_marker!r} not found in {mod}"
        # Window from the success marker to the end of that try-block.
        tail = src[i:]
        stop = tail.find("\n        except")
        window = tail[:stop] if stop > 0 else tail[:800]
        assert fn in window, (
            f"{fn} is not called on the SUCCESS path of {mod} — a healthy "
            f"fetch followed by a restart would re-spend account quota"
        )


class TestGciEtfUsesRealFlows:
    def test_aggregate_is_one_record_per_asset(self):
        """The caller SUMS the list, so one aggregate per *ticker* would
        multiply the real flow by the number of tickers."""
        src = _strip_comments(
            _src(REPO / "data_mgmt" / "global_context_informer.py"))
        assert "_fetch_aggregate_flow_coinglass" in src
        # Anchor INSIDE fetch_all_flows_async — a bare `if not
        # YFINANCE_AVAILABLE:` matches an earlier site and would swallow
        # unrelated code (the P238 non-unique-anchor trap).
        fn = re.search(
            r"async def fetch_all_flows_async.*?return btc_flows, eth_flows",
            src, re.S)
        assert fn, "fetch_all_flows_async not found"
        m = re.search(
            r"if not YFINANCE_AVAILABLE:.*?return btc_flows, eth_flows",
            fn.group(0), re.S)
        assert m, "the aggregate early-return path is missing"
        block = m.group(0)
        assert "for _asset, _bucket in" in block
        assert "SPOT_ETF_TICKERS" not in block, (
            "the aggregate path must NOT iterate per-ticker"
        )

    def test_streak_is_keyed_by_asset_not_ticker(self):
        """_calculate_streak buckets any non-BTC ticker as ETH, so a
        synthetic aggregate ticker would land in the wrong bucket."""
        from data_mgmt.global_context_informer import ETFFlowTracker
        t = ETFFlowTracker()
        t._calculate_streak_for_asset("BTC", -100.0)
        assert t.get_streak("BTC") == -1
        assert t.get_streak("ETH") == 0, "BTC flow must not move ETH's streak"

    def test_synthetic_ticker_would_have_been_misbucketed(self):
        """Pins WHY the asset-keyed entry point exists — this is the bug
        that would have shipped without it."""
        from data_mgmt.global_context_informer import ETFFlowTracker
        t = ETFFlowTracker()
        t._calculate_streak("BTC_AGGREGATE_COINGLASS", -100.0)
        assert t.get_streak("ETH") == -1, (
            "the ticker heuristic still mis-buckets unknown tickers as ETH — "
            "which is exactly why the aggregate path must not use it"
        )

    def test_stale_flow_reads_as_absent(self, monkeypatch):
        from data_mgmt.global_context_informer import ETFFlowTracker
        t = ETFFlowTracker()

        class _Stale:
            def latest_completed_flow(self, asset):
                return 1.0e8, 9.0, "2026-08-08"   # 9 days old

        t._cg_etf_reader = _Stale()
        assert t._fetch_aggregate_flow_coinglass("BTC") is None, (
            "a flow that old is a reporting gap, not a reading about today"
        )

    def test_missing_flow_returns_none_not_zero(self):
        from data_mgmt.global_context_informer import ETFFlowTracker
        t = ETFFlowTracker()

        class _Empty:
            def latest_completed_flow(self, asset):
                return None, None, None

        t._cg_etf_reader = _Empty()
        assert t._fetch_aggregate_flow_coinglass("BTC") is None


# =============================================================================
# 10. [P293c] Free RSS news — the cheaper CryptoPanic answer
# =============================================================================

class TestRssNewsFeed:
    """Both paid news sources are dark (CryptoPanic monthly quota,
    CryptoCompare 90/90 on an unraisable cap) and headline_count==0 is what
    makes llm_sentiment untradeable. RSS is free, keyless and unquota'd."""

    def test_sol_matching_is_word_bounded(self):
        """A naive `"sol" in title` matches solution/sold/console/solar and
        would flood SOL with unrelated news — corrupting the one input this
        feed exists to supply."""
        from data_mgmt.feeds.rss_news_feed import RSSNewsItem
        traps = [
            "A simple solution to scaling",
            "Shares were sold at a premium",
            "Console makers eye blockchain",
            "Solar farms power mining rigs",
            "Consolidation continues",
        ]
        for t in traps:
            assert not RSSNewsItem(t, None, "x").matches("SOL"), f"false match: {t}"
        for t in ["Solana fee overhaul lands", "SOL rallies 4%"]:
            assert RSSNewsItem(t, None, "x").matches("SOL"), f"missed: {t}"

    def test_btc_and_eth_matching(self):
        from data_mgmt.feeds.rss_news_feed import RSSNewsItem
        assert RSSNewsItem("Bitcoin ETF flows turn", None, "x").matches("BTC")
        assert RSSNewsItem("BTC price loses trend line", None, "x").matches("BTC")
        assert RSSNewsItem("Ethereum devs narrow proposals", None, "x").matches("ETH")
        assert not RSSNewsItem("Ethereum devs narrow proposals", None, "x").matches("BTC")

    def test_undated_item_is_none_not_now(self):
        """Stamping an unparseable date as NOW lets an old article pass a
        freshness window (the P287 defect)."""
        from data_mgmt.feeds.rss_news_feed import _parse_date
        assert _parse_date(None) is None
        assert _parse_date("not a date") is None
        assert _parse_date("") is None

    def test_parses_rfc822_and_iso(self):
        from data_mgmt.feeds.rss_news_feed import _parse_date
        d1 = _parse_date("Mon, 17 Aug 2026 12:00:00 +0000")
        d2 = _parse_date("2026-08-17T12:00:00Z")
        assert d1 is not None and d1.tzinfo is not None
        assert d2 is not None and d2.tzinfo is not None
        assert d1 == d2

    def test_parses_rss_and_atom_shapes(self):
        from data_mgmt.feeds.rss_news_feed import parse_feed
        rss = """<?xml version="1.0"?><rss><channel>
          <item><title>Bitcoin surges</title>
                <pubDate>Mon, 17 Aug 2026 12:00:00 +0000</pubDate></item>
          <item><title>Solana upgrade</title></item>
        </channel></rss>"""
        items = parse_feed(rss, "t")
        assert [i.title for i in items] == ["Bitcoin surges", "Solana upgrade"]
        assert items[0].published_at is not None
        assert items[1].published_at is None

        atom = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
          <entry><title>Ether climbs</title>
                 <updated>2026-08-17T10:00:00Z</updated></entry></feed>"""
        a = parse_feed(atom, "t")
        assert len(a) == 1 and a[0].title == "Ether climbs"

    def test_malformed_xml_returns_empty_not_raise(self):
        from data_mgmt.feeds.rss_news_feed import parse_feed
        assert parse_feed("<not xml", "t") == []

    def test_failed_sweep_keeps_cache_and_does_not_stamp_fresh(self):
        """A failed sweep stamped as fresh reads as 'no news happened'
        (the P265f defect)."""
        import asyncio
        from data_mgmt.feeds.rss_news_feed import RSSNewsFeed, RSSNewsItem
        f = RSSNewsFeed(sources=(("dead", "http://127.0.0.1:9/none"),))
        f._items = [RSSNewsItem("cached BTC story", None, "old")]
        f._last_fetch = 0.0
        out = asyncio.run(f.fetch())
        assert len(out) == 1, "cache must survive a total failure"
        assert f._last_fetch == 0.0, "a failed sweep must not stamp fresh"

    def test_blend_is_additive_and_last(self):
        """RSS must only ADD what the curated sources did not supply."""
        src = _strip_comments(_src(REPO / "agents" / "sentiment_llm_agent.py"))
        i_cc = src.find("cc_news_added")
        i_rss = src.find("rss_added")
        assert i_cc > 0 and i_rss > i_cc, "the RSS blend must come after CC News"
        assert "if not tl or tl in seen_titles:" in src, "dedup against prior sources"

    def test_blend_respects_the_same_window(self):
        """An RSS item older than the window must age out exactly like a CC
        News one, or stale copy satisfies the starvation gate."""
        src = _strip_comments(_src(REPO / "agents" / "sentiment_llm_agent.py"))
        m = re.search(r"rss_added.*?return meta", src, re.S)
        assert m, "RSS blend block not found"
        assert "ri.published_at < cutoff" in m.group(0), (
            "the RSS blend must apply the same cutoff as the other sources"
        )

    def test_no_fabricated_panic_metrics(self):
        """RSS has no vote data; inventing panic/velocity from a headline
        count would be a fabricated measurement (P2)."""
        src = _src(REPO / "data_mgmt" / "feeds" / "rss_news_feed.py")
        for forbidden in ("panic_score", "news_velocity", "narrative_intensity"):
            assert f'"{forbidden}"' not in src, (
                f"RSS must not synthesise {forbidden}"
            )


# =============================================================================
# 11. Anti-vacuity — the scan-based tests must be able to fail
# =============================================================================

class TestScansAreNotVacuous:
    def test_comment_stripper_actually_strips(self):
        assert "secret" not in _strip_comments("x = 1  # secret")
        assert "keep" in _strip_comments("x = 'keep'  # drop")

    def test_main_source_is_readable_and_large(self):
        """A scanner that silently reads nothing reports exactly what a
        clean scan reports (P171/P226)."""
        src = _strip_comments(_src(MAIN))
        assert len(src) > 100_000, "main.py scan returned too little to trust"
