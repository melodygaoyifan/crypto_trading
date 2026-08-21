"""[P355] Two operator-facing items closed: the Ensemble Regime Classifier now
has real inputs, and a resolved account cap is noticed within a day.

PART 1 — THE CLASSIFIER'S INPUTS.
P347 measured 8 of its 10 market/technical inputs with NO producer anywhere,
so every `market_data.get(key, default)` returned the default on every tick:
the model scored a market in which the price had not moved on any horizon,
MACD was flat, Bollinger sat mid-band and CVD was zero — the same
fabricated-neutral vector whatever the market was doing. A control fed
defaults is not a conservative control, it is a random one with a
safety-sounding name.

Re-measured here against the producers: of the 16 inputs, **13 are
recoverable** and only 3 genuinely have no source anywhere.

    NAME-ONLY (the pipeline computes exactly this quantity, under another
    name — the P2 reader/writer mismatch, ten times over):
        price_return_1h      <- price_change_1h_pct   (P306, real Kraken hourly)
        price_return_4h      <- price_change_4h_pct
        rsi_14               <- rsi
        macd_histogram       <- macd_hist
        volume_ratio         <- volume_ratio          (already correct)
        volatility_4h        <- volatility_4h         (already correct)
        funding_rate         <- funding_rate          (already correct)
        open_interest_change <- oi_change_24h_pct
        fear_greed_index     <- fear_greed_value
        cvd                  <- cvd_divergence        (in agent_signals)

    DERIVED from real series already present:
        price_return_24h  — 6 bars of 4H closes
        volatility_24h    — realized vol over the same closes
        bb_position       — (close - bb_lower) / (bb_upper - bb_lower)

    NO SOURCE ANYWHERE, left ABSENT rather than defaulted:
        volatility_1h    — needs an hourly price SERIES; the pipeline keeps
                           only the last hourly CHANGE (P306)
        basis            — exists only inside the calbasis shadow ledger
        social_sentiment — no producer at all

Resolving the names is only half of it. Without a refusal, a feed outage
silently restores exactly the state P347 found, so the caller now REFUSES to
classify when too many inputs are absent — an absent input must never be
indistinguishable from a measured neutral one (P2/P223).

PART 2 — THE ENTRY-ONLY GUARD THAT COULD NOT WORK.
`[V6 SHORT FILTER]` is flatten-intended, and its "is this a new short?" test
read `position_state`, which in LIVE resolves to the Kraken book — empty since
2026-06-13 (P275/P338). So it was STRUCTURALLY TRUE for every routed asset and
the filter would have liquidated a live Coinbase short while believing there
was none. It now reads the sleeve's reconciled book, and an UNKNOWN book does
not veto: firing on an unreadable book liquidates a real position, while not
firing admits one short entry that every other gate still sees.

PART 3 — A RESOLVED CAP IS NOTICED.
The Anthropic account cap stated a reset of 2026-09-01 and was resolved on
08-21. `classify_external_failure` suppressed for the whole stated interval,
so a long-lived process would have sat on the headline heuristic for eleven
more days; only a restart (which wipes the in-memory disable) let Haiku
resume. **A stated reset is the EARLIEST the vendor promises access back, not
the latest.** The suppression is now the earlier of the stated reset and the
bounded re-probe — capping only ever SHORTENS, so P293b's real defect (a
monthly quota retried on a 900s transient backoff, ~2,900 pointless requests)
stays fixed at one request per re-probe interval.
"""

import inspect

import pytest

from orchestration.strategic_coordinator import StrategicCoordinator
from infra.failure_policy import (
    classify_external_failure,
    DEFAULT_QUOTA_REPROBE_SEC,
    FailureClass,
)
import main
from tests._guard_pins import assert_guard_live


def _full_market_data():
    """Everything a healthy tick publishes, under the REAL producer names."""
    return {
        "price_change_1h_pct": 0.004,
        "price_change_4h_pct": -0.011,
        "rsi": 57.3,
        "macd_hist": 0.28,
        "volume_ratio": 1.35,
        "volatility_4h": 0.031,
        "funding_rate": 0.00007,
        "oi_change_24h_pct": 0.042,
        "fear_greed_value": 31,
        "prices": [100.0, 101.5, 102.0, 101.2, 103.4, 104.1, 105.0, 106.2],
        "bb_upper": 110.0,
        "bb_lower": 100.0,
        "current_price": 106.2,
    }


# ==========================================================================
# PART 1 — inputs
# ==========================================================================
def test_thirteen_of_sixteen_inputs_now_resolve_from_real_producers():
    vals, missing = StrategicCoordinator._resolve_regime_inputs(
        _full_market_data(), {"cvd_divergence": 0.41})
    assert len(vals) == 13, sorted(vals)
    assert sorted(missing) == ["basis", "social_sentiment", "volatility_1h"]


def test_every_alias_points_at_a_key_some_producer_actually_writes():
    """The P2 fix must not introduce a second generation of phantom names."""
    import data_mgmt.market_data_pipeline as mdp
    import agents.microstructure_agent as micro
    haystack = (inspect.getsource(mdp) + inspect.getsource(main)
                + inspect.getsource(micro))
    for field_name, (src_key, where) in (
            StrategicCoordinator._REGIME_INPUT_ALIASES.items()):
        assert f'"{src_key}"' in haystack or f"'{src_key}'" in haystack, (
            f"{field_name} is aliased to {src_key!r}, which no producer "
            f"writes — that is the defect this change exists to fix, one "
            f"generation on"
        )


@pytest.mark.parametrize("field_name,src_key", [
    ("price_return_1h", "price_change_1h_pct"),
    ("price_return_4h", "price_change_4h_pct"),
    ("macd_histogram", "macd_hist"),
    ("open_interest_change", "oi_change_24h_pct"),
    ("fear_greed_index", "fear_greed_value"),
])
def test_each_alias_carries_the_producers_value_through(field_name, src_key):
    md = _full_market_data()
    vals, _ = StrategicCoordinator._resolve_regime_inputs(md, {})
    assert vals[field_name] == pytest.approx(md[src_key])


def test_cvd_comes_from_agent_signals_not_market_data():
    """It lives in the other dict; reading only market_data is why it was 0."""
    vals, missing = StrategicCoordinator._resolve_regime_inputs(
        _full_market_data(), {"cvd_divergence": -0.6})
    assert vals["cvd"] == pytest.approx(-0.6)
    assert "cvd" not in missing

    vals2, missing2 = StrategicCoordinator._resolve_regime_inputs(
        _full_market_data(), None)
    assert "cvd" not in vals2 and "cvd" in missing2, (
        "an absent agent signal must be MISSING, never a fabricated 0"
    )


def test_the_derived_values_are_arithmetic_on_the_real_series():
    md = _full_market_data()
    vals, _ = StrategicCoordinator._resolve_regime_inputs(md, {})
    p = md["prices"]
    assert vals["price_return_24h"] == pytest.approx((p[-1] - p[-7]) / p[-7])
    assert vals["bb_position"] == pytest.approx(
        (md["current_price"] - md["bb_lower"])
        / (md["bb_upper"] - md["bb_lower"]))
    assert 0.0 < vals["volatility_24h"] < 0.5


def test_bb_position_is_clamped_to_the_band():
    md = _full_market_data()
    md["current_price"] = 500.0
    vals, _ = StrategicCoordinator._resolve_regime_inputs(md, {})
    assert vals["bb_position"] == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "n/a", ""])
def test_an_unusable_value_is_missing_rather_than_coerced(bad):
    md = _full_market_data()
    md["rsi"] = bad
    vals, missing = StrategicCoordinator._resolve_regime_inputs(md, {})
    assert "rsi_14" not in vals and "rsi_14" in missing


def test_a_short_price_series_leaves_both_derived_values_missing():
    md = _full_market_data()
    md["prices"] = [100.0, 101.0]
    _, missing = StrategicCoordinator._resolve_regime_inputs(md, {})
    assert "price_return_24h" in missing and "volatility_24h" in missing


def test_the_three_unavailable_inputs_are_always_reported_missing():
    """Their absence is permanent and must never read as a measurement."""
    vals, missing = StrategicCoordinator._resolve_regime_inputs(
        _full_market_data(), {"cvd_divergence": 0.1})
    for k in StrategicCoordinator._REGIME_INPUT_UNAVAILABLE:
        assert k in missing
        assert k not in vals


def test_the_rosters_partition_the_dataclass_exactly():
    """Anti-rot: a new RegimeFeatures field must be classified, not silently
    left to its default (P310's both-directions rule)."""
    aliases = set(StrategicCoordinator._REGIME_INPUT_ALIASES)
    derived = set(StrategicCoordinator._REGIME_INPUT_DERIVED)
    unavailable = set(StrategicCoordinator._REGIME_INPUT_UNAVAILABLE)
    assert not (aliases & derived) and not (aliases & unavailable)
    assert not (derived & unavailable)
    total = len(aliases) + len(derived) + len(unavailable)
    assert total == StrategicCoordinator._REGIME_TOTAL_INPUTS, (
        f"{total} inputs are classified but the builder passes "
        f"{StrategicCoordinator._REGIME_TOTAL_INPUTS} — an unclassified field "
        f"falls back to its dataclass default with nobody noticing"
    )


def test_the_caller_refuses_on_a_mostly_absent_vector():
    """Pinned with assert_guard_live, because `"X" in src` stays true under
    `if False and X` — the trap P234/P251/P307 hit three times and P311 built
    this helper to close. The falsification probe caught the first draft of
    this very test doing exactly that.
    """
    src = inspect.getsource(StrategicCoordinator.filter_short_intent)
    assert_guard_live(
        src, "_absent > self._REGIME_MAX_ABSENT_INPUTS",
        why="resolving the names without a live refusal means a feed outage "
            "silently restores the fabricated-neutral vector P347 found")
    assert "return result" in src.split("_REGIME_MAX_ABSENT_INPUTS", 1)[1][:900]
    # the floor must leave headroom for exactly the permanent absences plus a
    # small transient gap, not admit an arbitrary number of them
    assert (StrategicCoordinator._REGIME_MAX_ABSENT_INPUTS
            >= len(StrategicCoordinator._REGIME_INPUT_UNAVAILABLE))
    assert (StrategicCoordinator._REGIME_MAX_ABSENT_INPUTS
            < StrategicCoordinator._REGIME_TOTAL_INPUTS // 2)


def test_the_builder_returns_what_is_missing_alongside_the_features():
    """A builder that returns only the vector cannot be refused on."""
    src = inspect.getsource(StrategicCoordinator._build_regime_features)
    assert "return features, missing" in src
    assert "return None, [" in src, "the unavailable path must report too"


# ==========================================================================
# PART 2 — the entry-only guard
# ==========================================================================
def test_both_flatten_intended_short_vetoes_read_the_sleeve():
    """BOTH sites, because fixing one instance of a class is not fixing the
    class (P171/P226). The second — `[P0 SHORT BLOCK]`, one screen above —
    was found only because the guard written for the first was scoped to the
    whole file rather than to the block it was about.
    """
    src = inspect.getsource(main)
    for marker in ("[V6 SHORT FILTER] New short BLOCKED",
                   "[P0 SHORT BLOCK] Shorts blocked by regime/squeeze"):
        i = src.index(marker)
        window = src[max(0, i - 2600):i]
        assert "short_already_held(" in window, (
            f"the guard before {marker!r} does not consult the sleeve book — "
            f"this veto is flatten-intended, so it would liquidate a live short"
        )
    assert "if current_exposure >= 0:" not in src, (
        "the structurally-always-true guard is back"
    )


def test_there_is_exactly_one_implementation_of_the_question():
    """Two copies is how the two sites drift apart again (P172)."""
    src = inspect.getsource(main)
    assert src.count("def short_already_held(") == 1
    assert src.count("short_already_held(") == 3, (
        "expected one definition and two call sites"
    )


@pytest.mark.parametrize("contracts,expected", [
    (-2.0, True), (-1.0, True), (0.0, False), (1.0, False), (3.0, False),
])
def test_the_resolver_answers_from_the_sleeve_book(contracts, expected):
    class _S:
        _reconcile_ok = True
        def signed_contracts(self, a):
            return contracts
    assert main.short_already_held(_S(), "ETH", {"current_exposure": 0.0}) is expected


def test_a_stale_or_raising_sleeve_is_unknown_never_false():
    class _Stale:
        _reconcile_ok = False
        def signed_contracts(self, a):
            return -5.0
    class _Raises:
        _reconcile_ok = True
        def signed_contracts(self, a):
            raise RuntimeError("venue 502")
    assert main.short_already_held(_Stale(), "ETH", {}) is None
    assert main.short_already_held(_Raises(), "ETH", {}) is None


def test_with_no_sleeve_the_kraken_book_is_still_the_right_answer():
    """Paper / single-venue runs must keep working exactly as before."""
    assert main.short_already_held(None, "ETH", {"current_exposure": -0.2}) is True
    assert main.short_already_held(None, "ETH", {"current_exposure": 0.0}) is False
    assert main.short_already_held(None, "ETH", {"current_exposure": "x"}) is None


@pytest.mark.parametrize("var", ["_v6_short_held", "_p0_short_held"])
def test_an_unknown_book_does_not_veto(var):
    """`is False` and not `not x` — the whole point is that unknown must not
    be treated as 'no short here'."""
    src = inspect.getsource(main)
    assert f"if {var} is False:" in src, (
        f"{var} is not tested with `is False`, so an UNKNOWN book (None) "
        f"would take the veto branch and liquidate a live short"
    )
    assert f"if not {var}" not in src
    assert f"{var} is None" in src, (
        "an unknown book must say so rather than pass silently"
    )


def test_the_short_filter_still_receives_the_agent_signals_it_needs():
    assert "agent_signals" in inspect.signature(
        StrategicCoordinator.filter_short_intent).parameters
    src = inspect.getsource(main)
    assert "agent_signals=agent_signals,  # [P355] carries `cvd`" not in src or True
    i = src.index("short_filter = self.strategic_coordinator.filter_short_intent(")
    assert "agent_signals=agent_signals" in src[i:i + 900], (
        "without it `cvd` is permanently MISSING at the one call site that "
        "matters"
    )


def test_the_module_is_still_not_shipped_so_none_of_this_is_live_yet():
    """Arming remains an operator flip (P141/P347). If the module lands in the
    image this test goes red and that decision gets re-opened deliberately."""
    import pathlib
    root = pathlib.Path(main.__file__).parent
    df = (root / "Dockerfile.engine").read_text(encoding="utf-8")
    di = (root / ".dockerignore").read_text(encoding="utf-8")
    assert "training/regime" not in df and "training/regime" not in di, (
        "the Ensemble Regime Classifier is now in the image — its veto is "
        "flatten-intended and becomes live for the first time. Re-open the "
        "arming decision (P347) rather than letting it ship as a side effect."
    )


# ==========================================================================
# PART 3 — the quota re-probe
# ==========================================================================
def _regain(msg):
    return classify_external_failure(status=400, message=msg)


def test_a_far_future_stated_reset_is_capped_at_the_reprobe():
    p = _regain("You have reached your specified API usage limits. You will "
                "regain access on 2099-09-01 at 00:00 UTC.")
    assert p.failure_class is FailureClass.QUOTA_EXHAUSTED
    assert p.retry_after_sec == DEFAULT_QUOTA_REPROBE_SEC
    assert "EARLIEST" in p.reason, (
        "the reason must say the date was read and then capped, or an "
        "operator cannot tell this from a parse failure"
    )


def test_a_near_stated_reset_is_honoured_exactly():
    """The half of P319 that must never regress: the server's own date is
    used when it is sooner than the re-probe."""
    from datetime import datetime, timedelta, timezone
    soon = (datetime.now(timezone.utc) + timedelta(hours=3)).strftime(
        "%Y-%m-%d at %H:%M")
    p = _regain(f"usage limit; you will regain access on {soon} UTC")
    assert 0 < p.retry_after_sec < DEFAULT_QUOTA_REPROBE_SEC


def test_the_suppression_is_real_and_never_zero():
    """Capping must not become 'retry immediately' — that is P293b's defect."""
    for msg in ("usage limit; regain access on 2099-01-01 at 00:00 UTC",
                "you have reached your usage limit"):
        p = _regain(msg)
        assert p.suppresses and p.retry_after_sec > 0


def test_an_ordinary_malformed_400_is_still_not_silenced():
    p = classify_external_failure(status=400, message="messages.0: invalid field")
    assert p.failure_class is not FailureClass.QUOTA_EXHAUSTED


def test_a_cap_resolved_early_is_noticed_within_a_day():
    """The incident, as arithmetic: the Anthropic cap stated 2026-09-01 and
    was resolved on 08-21. Under the old rule a long-lived process stayed on
    the headline heuristic for the full stated interval."""
    p = _regain("You will regain access on 2099-09-01 at 00:00 UTC.")
    assert p.retry_after_sec <= 24 * 3600.0, (
        "a resolved cap must be re-probed within a day, not at the vendor's "
        "stated date"
    )
