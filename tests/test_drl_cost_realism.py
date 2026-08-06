"""[P179-P184] The DRL training harness measured the wrong things.

Five defects, all the same shape — a number that looked like a measurement and
could not have been one:

  P179  `_compute_trade_cost_bps` hardcoded `fee_bps = 0.0  # within free tier
        ($10K/mo)`. KRAKEN_PRO_FEES was defined and never read; `grep -rn`
        returned one line, its own definition. Every DRL model was validated
        against slippage and impact alone (3-10 bps) while the live Kraken
        taker fee is 26 bps.

  P180  `best_fold = max(mean_reward)`. mean_reward is the shaped training
        signal, which included a bonus for holding a position aligned with the
        GMM regime label whether or not it made money. Fold selection was
        therefore partly a statement about the reward function.

  P181  `_evaluate` ran ten `deterministic=True` rollouts over one fixed
        window. A deterministic policy on a fixed window is a pure function,
        so all ten were identical and `std_reward` was exactly 0.0 on every
        fold of every asset of every run.

  P182  No baseline was ever run through this environment, so
        "under-performs buy-and-hold" was not an observable outcome.

  P184  `_regime_to_name` returned "2.0" for a float regime id, silently
        disabling every regime-conditional reward term.

Two layers here. The source-level gates run everywhere and carry the weight.
The behavioural tests need gymnasium/stable_baselines3 and skip on machines
that do not train — which is most of them, so they must not be the only thing
standing behind these fixes.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source_scan import code_only, read_source  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER = REPO_ROOT / "training" / "train_drl_full.py"
COUNTERFACTUAL = (REPO_ROOT / "analytics" / "drl_realized"
                  / "drl_counterfactual_sharpe.py")


@pytest.fixture(scope="module")
def src():
    """Trainer source with comments AND docstrings blanked.

    Docstrings too: the P179 fix documents the removed line inside a
    docstring, so a comments-only scan matches the explanation. See
    tests/_source_scan.py.
    """
    return code_only(TRAINER, strip_docstrings=True)


# =============================================================================
# P179 - the fee is charged
# =============================================================================

class TestTheTrainerChargesRealFees:
    def test_the_hardcoded_zero_fee_is_gone(self, src):
        assert not re.search(r"\bfee_bps\s*=\s*0(\.0)?\b", src), (
            "`fee_bps = 0.0` is back in the trainer. Every model trained "
            "while that line exists is validated at roughly half the friction "
            "it will pay on Kraken."
        )

    def test_the_fee_comes_from_the_venue_table(self, src):
        assert re.search(r"fee_bps\s*=\s*self\._fee_bps", src), (
            "_compute_trade_cost_bps no longer reads the resolved venue fee"
        )

    def test_the_fallback_table_matches_the_live_one(self):
        """A mirrored constant with no equality check is a future divergence.

        This is the P170/P176 lesson applied pre-emptively: the trainer keeps
        a literal fallback for boxes that cannot import the repo root, and
        nothing else would notice if the live table moved.

        This test found the fix's own bug: `_load_venue_fees` imported
        `_VENUE_FEES`, which does not exist, and `except Exception` returned
        the fallback in silence. Both tables read 26bps, so nothing looked
        wrong — the trainer was simply never reading the live one.
        """
        from core.paper_fee_service import VENUE_FEE_STD

        fallback = _extract_fallback_table()
        for venue, sides in fallback.items():
            assert venue in VENUE_FEE_STD, (
                f"trainer prices {venue!r}, core/paper_fee_service.py does not"
            )
            for side, bps in sides.items():
                live_bps = float(VENUE_FEE_STD[venue][side]) * 10000.0
                assert bps == pytest.approx(live_bps, abs=1e-6), (
                    f"{venue}/{side}: trainer fallback says {bps} bps, live "
                    f"venue table says {live_bps} bps. Training and execution "
                    f"now disagree about what a trade costs."
                )

    def test_the_live_table_import_names_a_symbol_that_exists(self, src):
        """The silent-fallback trap, pinned.

        `_load_venue_fees` catches Exception and returns the hardcoded copy.
        Because the copy is correct, a broken import produces the right
        numbers from the wrong source — which is how the original version,
        importing a symbol that has never existed, went unnoticed.
        """
        import core.paper_fee_service as pfs

        m = re.search(r"from core\.paper_fee_service import (\w+)", src)
        assert m, "the trainer no longer imports the live fee table at all"
        name = m.group(1)
        assert hasattr(pfs, name), (
            f"training/train_drl_full.py imports "
            f"core.paper_fee_service.{name}, which does not exist. The "
            f"ImportError is swallowed and the hardcoded fallback is used, so "
            f"this fails silently and the fee numbers still look right."
        )

    def test_the_fee_table_source_is_recorded(self, src):
        """Which table was used is not inferable from the values."""
        assert "VENUE_FEES_SOURCE" in src, (
            "the loader no longer reports whether it read the live table or "
            "the fallback; the two are identical by value, so nothing else "
            "can tell them apart"
        )

    def test_the_dead_fee_table_is_not_back(self, src):
        assert "KRAKEN_PRO_FEES" not in src, (
            "KRAKEN_PRO_FEES is defined again. It is a second, unread fee "
            "table; the first one being unread for the whole life of the "
            "trainer is what P179 was."
        )

    def test_free_tier_is_opt_in_not_the_default(self, src):
        assert re.search(r"assume_free_tier\s*:\s*bool\s*=\s*False", src), (
            "assume_free_tier defaults to True again, which silently restores "
            "zero-fee training for anyone who does not pass the flag"
        )


# =============================================================================
# P180 - selection and reward shaping
# =============================================================================

class TestFoldSelectionFollowsMoney:
    def test_best_fold_is_chosen_by_after_cost_pnl(self, src):
        """The primary selection, not the degraded one.

        A plain "mean_reward does not appear" assertion fails against the
        deliberate fallback for folds restored from cache, which have no
        after-cost figure. So pin the ordering instead: PnL selection first,
        the reward fallback only after it, and only under a guard.
        """
        primary = src.find("best_fold = max(self.results, key=_sel)")
        assert primary != -1, (
            "best_fold is no longer selected by _sel (after-cost PnL). "
            "mean_reward is the shaped training signal, not realized money."
        )
        fallback = src.find('key=lambda k: self.results[k]["mean_reward"]')
        if fallback != -1:
            assert fallback > primary, (
                "the mean_reward fallback now runs before the PnL selection")
            guard = src.rfind("if not np.isfinite(best_pnl):", primary, fallback)
            assert guard != -1, (
                "the mean_reward fallback is no longer guarded by "
                "'no fold reported an after-cost PnL' — it is now reachable "
                "on folds that did report one")

    def test_the_selection_metric_is_recorded(self, src):
        assert '"selection_metric"' in src, (
            "the summary no longer records what selection was based on, so a "
            "results.json cannot be read without reading the code that made it"
        )

    def test_the_alignment_bonus_defaults_off(self, src):
        n = len(re.findall(
            r"regime_alignment_bonus\s*:\s*bool\s*=\s*False", src))
        assert n >= 3, (
            f"expected regime_alignment_bonus to default False in "
            f"EnhancedRewardCalculator, TradingEnvFull and FullDRLTrainer; "
            f"found {n} such defaults. If any one of them defaults True the "
            f"bonus is back on by whichever path constructs that object."
        )

    def test_both_copies_of_the_bonus_are_gated(self, src):
        """It is duplicated: the calculator and the classic branch of step()."""
        bonuses = re.findall(r"abs\(position_bias\)\s*\*\s*0\.5", src)
        assert len(bonuses) == 2, (
            f"expected exactly 2 copies of the alignment bonus, found "
            f"{len(bonuses)}; the gating below assumes both"
        )
        assert re.search(r"if\s+self\.regime_alignment_bonus\s*:", src), (
            "the EnhancedRewardCalculator copy is ungated"
        )
        assert re.search(r"if\s+self\._regime_alignment_bonus\s*:", src), (
            "the classic-branch copy in step() is ungated. It is the worse "
            "one: it adds to the raw reward with no quality_weight."
        )


# =============================================================================
# P181 - the error bar
# =============================================================================

class TestEvaluationHasAnErrorBar:
    def test_no_rollout_loop_hardcodes_deterministic(self, src):
        """Both loops, not just the one P181 was reported against.

        The trainer has two: `evaluate_policy_full` and Optuna's `_eval_nav`.
        Fixing only the first leaves hyperparameter selection ranking trials
        on five copies of one number.
        """
        bad = re.findall(
            r"model\.predict\(\s*obs\s*,\s*deterministic\s*=\s*True\s*\)", src)
        assert not bad, (
            f"{len(bad)} rollout loop(s) hardcode deterministic=True. On a "
            f"fixed validation window that makes every episode identical: "
            f"std_reward is exactly 0.0 and an n-episode mean is one sample."
        )
        assert src.count("deterministic=deterministic") >= 2, (
            "a rollout loop no longer takes its determinism from a parameter")

    def test_the_full_metrics_entry_point_exists(self, src):
        assert "def evaluate_policy_full" in src

    @pytest.mark.parametrize("key", [
        "mean_pnl_after_cost", "sharpe_after_cost",
        "sharpe_ci_low", "sharpe_ci_high", "degenerate_spread",
    ])
    def test_the_after_cost_metrics_are_reported(self, src, key):
        assert f'"{key}"' in src, f"{key} is no longer reported"

    def test_the_degenerate_case_is_still_detectable(self, src):
        """Not "we removed the bug" but "we can still see it if it returns"."""
        assert re.search(r"degenerate_spread.*np\.std\(rewards\)\)?\s*==\s*0",
                         src, re.S), (
            "degenerate_spread no longer derives from the actual spread, so "
            "it can no longer report the pre-P181 condition"
        )


# =============================================================================
# P182 - the baselines
# =============================================================================

class TestBaselinesGatePromotion:
    def test_both_baselines_exist(self, src):
        assert "def _buy_and_hold" in src
        assert "def _sma_rule" in src
        assert "def baseline_policies" in src

    def test_no_baselines_means_no_promotion(self):
        """The failure path is the one that matters.

        If the baseline run throws, the tempting default is to fall through
        and promote on the absolute number — which is precisely the gate being
        replaced.
        """
        verdict = _promotion_verdict_source()
        assert re.search(r"if\s+not\s+baselines\s*:", verdict), (
            "_promotion_verdict no longer special-cases the empty-baseline "
            "case; check that it cannot pass without a comparison"
        )
        assert re.search(r'"passes"\s*:\s*False', verdict), (
            "the empty-baseline branch must return passes=False"
        )

    def test_the_gate_requires_beating_every_baseline(self, src):
        assert re.search(r"passes\s*=\s*\(not lost_to\)", src), (
            "the promotion gate no longer requires beating every baseline"
        )

    def test_baseline_failure_returns_empty_not_partial(self, src):
        """A half-filled baseline dict would let a model 'beat' one baseline."""
        m = re.search(r"def _evaluate_baselines.*?\n    def ", src, re.S)
        body = m.group(0) if m else src
        assert body.count("return {}") >= 2, (
            "_evaluate_baselines must return {} on both the unwrap failure "
            "and the exception path, not a partially populated dict"
        )


# =============================================================================
# P184 - regime ids that do not resolve
# =============================================================================

class TestRegimeIdsResolve:
    def test_integral_floats_map_to_names(self, src):
        assert re.search(r"np\.floating", src), (
            "_regime_to_name no longer handles float regime ids. df.iloc[row] "
            "upcasts an int64 regime to float64 whenever every column in the "
            "row is numeric, and the old code then returned the string '2.0' "
            "— absent from POSITION_BIAS, BULL_REGIMES, BEAR_REGIMES and "
            "regime_weights, so every regime-conditional reward term became a "
            "no-op with no error."
        )

    def test_unresolvable_ids_are_reported(self, src):
        assert "_assert_regimes_resolve" in src
        assert re.search(r"self\._assert_regimes_resolve\(\)", src), (
            "the check exists but is never called"
        )

    def test_pandas_still_upcasts_the_way_this_fix_assumes(self):
        """Pin the platform behaviour the fix is a response to.

        If a future pandas stops upcasting, this test fails and the comment in
        _regime_to_name becomes wrong — better to be told than to leave an
        explanation that no longer describes reality.
        """
        import numpy as np
        import pandas as pd

        df = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=3, freq="4h"),
            "close": np.arange(3, dtype=float),
            "regime": np.arange(3, dtype=np.int64),
        })
        assert isinstance(df.iloc[1]["regime"], np.integer), (
            "a non-numeric column no longer preserves the int regime")
        numeric_only = df.drop(columns=["ts"])
        assert isinstance(numeric_only.iloc[1]["regime"], np.floating), (
            "pandas no longer upcasts; P184's premise has changed")


# =============================================================================
# P183 - the counterfactual can run, and refuses when it cannot
# =============================================================================

class TestTheCounterfactualRefusesEmptyInput:
    def test_the_attribution_directory_exists(self):
        d = REPO_ROOT / "logs" / "attribution"
        assert d.is_dir(), (
            "logs/attribution/ is gone. The counterfactual globs it; with the "
            "directory missing every trade lands in the no_signal bucket and "
            "the report prints '(empty)' rows next to a real ALL TRADES row — "
            "a run with zero input that reads as 'DRL contributed nothing'."
        )

    def test_it_exits_nonzero_on_a_missing_directory(self, tmp_path):
        r = subprocess.run(
            [sys.executable, str(COUNTERFACTUAL)],
            env={"PATH": "/usr/bin:/bin", "HMATS_LOG_DIR": str(tmp_path / "nope")},
            capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT),
        )
        assert r.returncode != 0, (
            "the counterfactual still produces a report with no input")
        assert "P183" in (r.stdout + r.stderr)

    def test_it_exits_nonzero_on_an_empty_directory(self, tmp_path):
        (tmp_path / "attribution").mkdir()
        r = subprocess.run(
            [sys.executable, str(COUNTERFACTUAL)],
            env={"PATH": "/usr/bin:/bin", "HMATS_LOG_DIR": str(tmp_path)},
            capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT),
        )
        assert r.returncode != 0
        assert "no signals_" in (r.stdout + r.stderr)

    def test_it_reports_coverage_not_just_buckets(self):
        text = read_source(COUNTERFACTUAL)
        assert "COVERAGE:" in text, (
            "the coverage line is gone. Without it the bucket table cannot "
            "distinguish 'DRL had no view' from 'the signal files do not "
            "overlap the trade window'."
        )


# =============================================================================
# Behavioural layer - needs the training stack
# =============================================================================

@pytest.fixture(scope="module")
def T():
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("sb3_contrib")
    sys.path.insert(0, str(REPO_ROOT))
    import training.train_drl_full as mod
    return mod


@pytest.fixture
def synth_df():
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(7)
    n = 600
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
        "close": close,
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "regime": rng.integers(0, 6, n),
    })


class TestTheFeeActuallyReachesTheCost:
    def test_venues_price_differently(self, T, synth_df):
        mk = lambda **kw: T.TradingEnvFull(  # noqa: E731
            df=synth_df, feature_cols=["f1", "f2"],
            reward_mode="classic", **kw)
        kraken = mk()._compute_trade_cost_bps(5000.0)
        coinbase = mk(venue="coinbase")._compute_trade_cost_bps(5000.0)
        free = mk(assume_free_tier=True)._compute_trade_cost_bps(5000.0)
        assert kraken - free == pytest.approx(26.0), (
            f"kraken taker fee is not reaching the trade cost "
            f"(kraken={kraken}, free_tier={free})")
        assert coinbase - free == pytest.approx(3.0)

    def test_an_unknown_venue_raises_rather_than_costing_nothing(
            self, T, synth_df):
        with pytest.raises(ValueError, match="unknown venue"):
            T.TradingEnvFull(df=synth_df, feature_cols=["f1", "f2"],
                             venue="krakken")

    def test_an_unknown_fee_side_raises(self, T, synth_df):
        with pytest.raises(ValueError, match="maker.*taker"):
            T.TradingEnvFull(df=synth_df, feature_cols=["f1", "f2"],
                             fee_side="marker")


class TestTheAlignmentBonusIsGated:
    def _total(self, T, df, actions, **kw):
        import numpy as np
        env = T.TradingEnvFull(df=df, feature_cols=["f1", "f2"], **kw)
        env.reset()
        tot = 0.0
        for a in actions:
            _, r, term, trunc, _ = env.step(np.array([a], dtype=np.float32))
            tot += r
            if term or trunc:
                break
        return tot

    @pytest.mark.parametrize("mode", ["classic", "sortino"])
    def test_the_flag_changes_the_reward(self, T, synth_df, mode):
        import numpy as np
        acts = list(np.random.default_rng(11).uniform(-1, 1, 200))
        off = self._total(T, synth_df, acts, reward_mode=mode)
        on = self._total(T, synth_df, acts, reward_mode=mode,
                         regime_alignment_bonus=True)
        assert off != pytest.approx(on), (
            f"[{mode}] the regime_alignment_bonus flag has no effect on the "
            f"reward, so turning it off did not remove the bonus"
        )


class TestRegimeResolutionEndToEnd:
    def test_a_float_regime_column_still_resolves(self, T, synth_df):
        import numpy as np
        df = synth_df.drop(columns=["timestamp"]).copy()
        env = T.TradingEnvFull(df=df, feature_cols=["f1", "f2"],
                               reward_mode="classic")
        env.reset()
        env.step(np.array([1.0], dtype=np.float32))
        assert env._get_regime() in T.REGIME_NAMES, (
            f"_get_regime returned {env._get_regime()!r} on a numeric-only "
            f"dataframe; the regime-conditional reward terms are inert"
        )


def _extract_fallback_table() -> dict:
    """Read _VENUE_FEES_FALLBACK_BPS out of the trainer without importing it.

    The trainer imports gymnasium at module scope, so the equality check
    against the live table would only run on training boxes if this imported
    normally — and that check is the one thing standing between the two tables
    drifting apart.
    """
    import ast

    tree = ast.parse(read_source(TRAINER))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_VENUE_FEES_FALLBACK_BPS"
            for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(
        "_VENUE_FEES_FALLBACK_BPS not found in training/train_drl_full.py")


def _promotion_verdict_source() -> str:
    src = code_only(TRAINER, strip_docstrings=True)
    m = re.search(r"def _promotion_verdict\(.*?\n(?=\S)", src, re.S)
    assert m, "_promotion_verdict is gone"
    return m.group(0)
