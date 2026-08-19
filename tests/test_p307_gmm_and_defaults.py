"""[P307] The GMM feature-set resolution, and the default-ON audit.

Two things are pinned here:

  * the 9-feature GMM contract, in every place it has to agree — the training
    builder, the runtime builder, and the DEPLOYED artifacts. P215's rule is
    that {GMM, parquets, builder} move as one versioned set; these tests are
    what makes a half-move loud instead of a silent fallback to the ADX proxy.

  * the roster of default-TRUE switches that have NO reader. Seven of those
    exist. They are not bugs today (a switch nobody consults changes nothing)
    but each reads as an armed control, and one of them sits on a component
    that binds the sleeve. The guard fails when an EIGHTH appears.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return io.open(REPO / rel, encoding="utf-8").read()


# ---------------------------------------------------------------------------
# 1. the 9-feature contract
# ---------------------------------------------------------------------------
class TestGmmFeatureContract:
    def _cols(self):
        # Load through the shared helper, NOT a plain import: rebuild_pipeline
        # does `from scripts.wavelet_denoise import ...` expecting training/
        # on sys.path, and any earlier test that imported the repo-root
        # `scripts` package binds sys.modules["scripts"] to the wrong one.
        # A plain import here passes alone and fails in the full suite (P194).
        from tests.test_rebuild_pipeline_gmm_split import _load_rebuild_module
        return list(_load_rebuild_module().GMM_FEATURE_COLS)

    def test_the_three_measured_dead_inputs_are_gone(self):
        """return_1h was return_4h/4 exactly (corr 1.0), which in a
        full-covariance GMM double-weights return_4h in the assignment
        distance — measured ARI 0.690/0.657/0.741 and a changed k on BTC.
        cross_asset_correlation and spread_percentile were constants on both
        sides (P221)."""
        cols = self._cols()
        for gone in ("return_1h", "cross_asset_correlation",
                     "spread_percentile"):
            assert gone not in cols, f"{gone} is back in the GMM feature set"
        assert len(cols) == 9

    def test_the_two_builders_agree_on_count_and_order(self):
        """A two-file contract (P192/P215). The runtime array is positional —
        an order change with a matching count is a silent feature swap."""
        cols = self._cols()
        rt = _src("data_mgmt/market_data_pipeline.py")
        body = rt.split("features = _np.array([")[1].split("]")[0]
        toks = [t.strip() for t in body.replace("\n", " ").split(",")
                if t.strip()]
        assert len(toks) == len(cols), (
            f"runtime builds {len(toks)}, training declares {len(cols)}")
        # positional correspondence, by the runtime's own local names
        expect = ["ret_4h", "ret_24h", "ret_7d", "vol_1h", "vol_24h",
                  "vol_pct", "vov", "mom_con", "fear_idx"]
        assert toks == expect, f"runtime feature ORDER drifted: {toks}"

    def test_the_shape_guard_is_a_function_that_can_be_called(self):
        """Behavioural, not textual. The first falsification probe of this
        guard stayed GREEN against `if False and ...` because the pin only
        asserted the predicate's source text existed — P234/P251 verbatim.
        The predicate now lives in a pure function, so the truth table is
        asserted by CALLING it and the call site is pinned separately."""
        from data_mgmt.market_data_pipeline import gmm_shape_mismatch as g
        assert g([1, 2, 3], [1, 2]) is True
        assert g([1, 2], [1, 2]) is False
        # a MISSING scaler is a different condition with its own branch;
        # collapsing the two would report the wrong cause
        assert g(None, [1, 2]) is False
        assert g([1, 2], None) is False
        assert g(3, 4) is False          # unsized input must not raise

    def test_the_shape_guard_is_actually_on_the_predict_path(self):
        rt = _src("data_mgmt/market_data_pipeline.py")
        assert "if gmm_shape_mismatch(_scaler_mean, features):" in rt, (
            "the guard is defined but the predict path no longer calls it")
        assert "feature_count_mismatch" in rt
        assert "logger.error" in rt.split("feature_count_mismatch")[0][-1500:]

    @pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
    def test_the_deployed_artifacts_match_the_builder(self, asset):
        p = REPO / "models" / "regime_classifier" / asset / "gmm_config.json"
        if not p.exists():
            pytest.skip("models/ is gitignored — artifact half runs locally")
        cfg = json.loads(p.read_text(encoding="utf-8"))
        cols = self._cols()
        assert cfg["feature_cols"] == cols, (
            f"{asset}: deployed feature_cols differ from the builder — the "
            f"P215 set is half-moved")
        for key in ("scaler_mean", "scaler_scale"):
            assert len(cfg[key]) == len(cols), f"{asset}: {key} length"
        assert cfg.get("fit_policy") == "split_aware", (
            f"{asset}: a full-sample fit reached the deploy dir (P164/P280)")

    @pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
    def test_no_deployed_cluster_mean_has_a_duplicated_coordinate_pair(
            self, asset):
        """The tell that made the old artifacts diagnosable: with return_1h a
        perfect duplicate, coordinates 0 and 1 of EVERY cluster mean were
        identical in the shipped JSON."""
        p = REPO / "models" / "regime_classifier" / asset / "gmm_config.json"
        if not p.exists():
            pytest.skip("models/ is gitignored")
        cfg = json.loads(p.read_text(encoding="utf-8"))
        for i, m in enumerate(cfg["means"]):
            assert abs(float(m[0]) - float(m[1])) > 1e-12, (
                f"{asset} cluster {i}: coordinates 0 and 1 are identical — a "
                f"duplicated feature is back in the fit")


# ---------------------------------------------------------------------------
# 2. default-ON audit
# ---------------------------------------------------------------------------
_SWITCH = re.compile(
    r"^\s{4}(enable[a-z0-9_]*|[a-z0-9_]*_enabled|[a-z0-9_]*_enforce"
    r"|use_[a-z0-9_]+)\s*:\s*bool\s*=\s*True", re.M)

_ROOTS = ("core", "defense", "risk", "signals", "execution", "agents",
          "strategies", "exchange", "integration", "data_mgmt", "analytics",
          "market", "orchestration", "infra")

# Measured 2026-08-18. Each is default TRUE and consulted NOWHERE, so it
# changes nothing while reading like an armed control. The trade_gate one is
# the reason this roster exists: that component binds the sleeve (P275), and
# `use_maker_fee` reads as "the gate prices maker fills", which it does not.
# [P311] Down from seven to five. `proof_log_enabled` and
# `use_vol_percentile_adjustment` were DELETED rather than annotated: a
# config file was setting the first, and a caller was passing the second, so
# in both cases something believed it was configuring behaviour. The three
# remaining have neither a reader nor a writer, and `use_maker_fee` is the
# one that matters — it sits on a component that binds the sleeve and reads
# as "the gate prices maker fills", which it does not.
NO_READER_DEFAULT_TRUE = {
    ("agents/onchain_solana_agent.py", "use_birdeye"),
    ("agents/onchain_solana_agent.py", "use_solscan"),
    ("core/runtime_spine.py", "enable_proof_logs"),
    ("defense/trade_gate.py", "use_maker_fee"),
    ("orchestration/sota_integration.py", "alert_popup_enabled"),
}


def _scan():
    texts = {}
    for r in _ROOTS:
        for p in (REPO / r).rglob("*.py"):
            if "archive" in p.parts or "legacy" in p.parts:
                continue
            texts[str(p.relative_to(REPO)).replace("\\", "/")] = \
                io.open(p, encoding="utf-8").read()
    texts["main.py"] = _src("main.py")
    found, dead = [], set()
    for f, t in texts.items():
        for m in _SWITCH.finditer(t):
            name = m.group(1)
            found.append((f, name))
            reads = sum(len(re.findall(r"[.\[\"']" + re.escape(name) + r"\b", s))
                        for s in texts.values())
            if reads == 0:
                dead.add((f, name))
    return found, dead


class TestDefaultOnAudit:
    def test_the_scan_finds_something(self):
        """P174: an empty scan makes every assertion below vacuous."""
        found, _ = _scan()
        assert len(found) > 50, len(found)

    def test_no_new_switch_is_armed_by_default_with_nobody_reading_it(self):
        _, dead = _scan()
        new = dead - NO_READER_DEFAULT_TRUE
        assert not new, (
            "new default-TRUE switch(es) with no reader anywhere: "
            f"{sorted(new)} — either wire it or default it False; a control "
            "that looks armed and does nothing is the P177 shape")

    def test_the_roster_has_not_rotted(self):
        """An entry that gained a reader, or moved, must leave the roster —
        otherwise this becomes a parking spot rather than a guard."""
        _, dead = _scan()
        stale = NO_READER_DEFAULT_TRUE - dead
        assert not stale, (
            f"roster entries no longer match reality: {sorted(stale)} — they "
            f"gained a reader or moved; update the roster with the reason")

    def test_the_profitmax_family_still_holds_rather_than_liquidates(self):
        """13 profit_max_* switches default TRUE with no live-profile key.
        P287 found their vetoes were an unclassified sleeve-liquidation door;
        the classification is what makes leaving them on defensible."""
        src = _src("main.py")
        blk = src[src.index("_SLEEVE_HOLD_VETOES"):][:4000]
        for veto in ("[FALSE_BREAKOUT_VETO]", "[LOSS_STREAK_HALT]"):
            assert veto in blk, f"{veto} left the sleeve HOLD roster"


# ---------------------------------------------------------------------------
# 3. the liveness finding
# ---------------------------------------------------------------------------
def test_soldex_confidence_is_zero_on_a_flat_direction():
    """Seen live on SOL: dir=+0.00 conf=1.00. Confidence was the raw
    liquidity score, so a deeply liquid DEX with no arbitrage opportunity
    emitted a maximally-confident non-signal — and both fusion and the IC
    scorer consume direction x confidence. The P224 defect in a second
    agent."""
    src = _src("main.py")
    i = src.index('agent_signals["soldex_confidence"]')
    stmt = src[i:i + 200]
    assert "abs(_sd_dir)" in stmt, (
        "soldex confidence no longer collapses on a flat direction")
    assert 'agent_signals["soldex_liquidity_score"] = _sd_liq' in src, (
        "the undiluted liquidity reading must stay available separately")


# ---------------------------------------------------------------------------
# 4. the recurring source-pin trap, made mechanical
# ---------------------------------------------------------------------------
_COND_PIN = re.compile(
    r"""assert\s+["']([^"']*(?:==|!=|>=|<=|\s>\s|\s<\s|\bnot\b|\band\b|\bor\b|\bis\b)[^"']*)["']\s+in\s+""",
    re.M)

# [P311] 14 -> 9. The five DEFEATABLE ones (a bare `if ...:` line, where
# `if False and ...` keeps the substring) now go through
# tests/_guard_pins.assert_guard_live, which requires the condition to be the
# WHOLE condition of its statement. Converting them immediately paid: the
# feed-degradation pin turned out to be one clause of a conjunction, which
# the substring form could not distinguish and now has to state.
#
# The ten that remain pin EXPRESSIONS (ternaries, comparisons inside a
# return) rather than guards, so the `if False and` trap does not apply to
# them; the ceiling stays as an anti-rot bound.
#
# Accepted 2026-08-18. A pin of the shape `assert "<condition>" in src`
# survives `if False and <condition>` — it proves the code was WRITTEN, not
# that it RUNS. That has now bitten three times (P234's gate-hysteresis block,
# P251's stale-snapshot guard, and P307's GMM shape guard, whose first
# falsification probe stayed green). Converting every existing instance would
# mean extracting six predicates across six live modules — disproportionate
# for guards that are currently correct. So the roster is frozen instead: it
# may SHRINK, never grow, and a new one sends its author to the fix that
# works (extract the predicate into a pure function and CALL it).
_ACCEPTED_COND_PINS = 9


def _cond_pins():
    out = []
    for p in sorted((REPO / "tests").glob("test_p3*.py")) +             sorted((REPO / "tests").glob("test_p29*.py")):
        t = io.open(p, encoding="utf-8").read()
        for m in _COND_PIN.finditer(t):
            out.append((p.name, t[:m.start()].count("\n") + 1, m.group(1)))
    return out


def test_the_condition_pin_scan_finds_something():
    """P174: a regex that matches nothing makes the guard below vacuous."""
    assert len(_cond_pins()) > 5


def test_no_new_source_text_condition_pins():
    n = len(_cond_pins())
    assert n <= _ACCEPTED_COND_PINS, (
        f"{n - _ACCEPTED_COND_PINS} new source-text pin(s) of a CONDITION: "
        f"{[h for h in _cond_pins()][_ACCEPTED_COND_PINS:]}. Such a pin stays "
        f"green against `if False and <condition>` — it proves the code was "
        f"written, not that it runs (P234/P251/P307). Extract the predicate "
        f"into a pure function and assert its truth table by CALLING it, then "
        f"pin the call site separately.")


def test_the_accepted_count_is_not_stale():
    """If the roster shrinks, lower it — otherwise the ceiling silently makes
    room for a new one."""
    n = len(_cond_pins())
    assert n == _ACCEPTED_COND_PINS, (
        f"only {n} condition pins remain (accepted {_ACCEPTED_COND_PINS}) — "
        f"lower _ACCEPTED_COND_PINS to {n} so the guard keeps its teeth")
