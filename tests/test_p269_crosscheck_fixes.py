"""P269 — the docs-vs-implementation cross-check fix batch, pinned.

The 2026-08-14 cross-check of GMM + DRL-retraining docs against the code
found a set of drifts in BOTH directions: docs prescribing retired/leaked-era
procedures (Makefile drl target training at the wrong venue with no fresh
tag; the guide's phantom "convert 4h -> _1d" step), and code silently
diverging from documented design (buffer cap only under --config; the
navigator's online GMM overwriting the calibrated per-asset posterior;
STEADY_UPTREND absent from the reward tables that the P221 vocabulary
requires; a latent cross-asset GMM fallback).

Every pin here is on the CLAIM, not on incidental phrasing (P177), and the
source scans use tests/_source_scan.code_only so a comment quoting the
retired string cannot satisfy or trip a check.
"""

import re
from pathlib import Path

import pytest

from tests._source_scan import code_only, read_source

REPO = Path(__file__).resolve().parent.parent
MAKEFILE = REPO / "training" / "Makefile"
RUN_TRAINING = REPO / "training" / "run_training.py"
TRAIN_DRL = REPO / "training" / "train_drl_full.py"
MAIN = REPO / "main.py"
PIPELINE = REPO / "data_mgmt" / "market_data_pipeline.py"
GUIDE = REPO / "docs" / "HMATS_TRAINING_GUIDE_V2.md"
CLAUDE_MD = REPO / "CLAUDE.md"


# ---------------------------------------------------------------------------
# 1. Makefile drl target: the honest-pipeline flags, no leaked-era overrides
# ---------------------------------------------------------------------------

class TestMakefileDrlTarget:
    def _drl_block(self) -> str:
        text = read_source(MAKEFILE)
        # the drl recipe runs until the next top-level target
        m = re.search(r"^drl:.*?(?=^\w[\w-]*:)", text, re.M | re.S)
        assert m, "drl target missing from training/Makefile"
        return m.group(0)

    def test_decision_interval_4_is_passed(self):
        assert "--decision-interval 4" in self._drl_block(), (
            "make drl lost --decision-interval 4 — it would train the "
            "retired every-bar churn formulation (P242/P258: ~480 trades/"
            "fold, friction >= the loss)")

    def test_no_stale_lr_override(self):
        assert "--lr 3e-5" not in self._drl_block(), (
            "make drl re-acquired --lr 3e-5 — the ULTIMATE preset default "
            "is 1.5255e-05; a Makefile override silently retunes every run")

    def test_no_uncapped_buffer_override(self):
        assert "--buffer-size 2000000" not in self._drl_block(), (
            "make drl re-acquired --buffer-size 2000000 (pre-Stage-9.7; "
            "with n_stack=8 that is the memory condition the 500K cap "
            "exists to prevent)")

    def test_tag_is_fresh_not_fixed(self):
        blk = self._drl_block()
        assert "makefile_$$(date" in blk, (
            "make drl must generate a FRESH tag per run — a fixed tag makes "
            "the trainer restore cached folds and report stale numbers as "
            "if it trained (P200 launch gotcha)")
        assert not re.search(r"--tag\s+makefile_default\b", blk)

    def test_venue_flags_still_present(self):
        blk = self._drl_block()
        assert "$(DRL_VENUE)" in blk and "$(DRL_FEE_SIDE)" in blk


# ---------------------------------------------------------------------------
# 2. run_training.py orchestrator
# ---------------------------------------------------------------------------

class TestRunTrainingOrchestrator:
    def test_gmm_step_uses_rebuild_pipeline(self):
        src = read_source(RUN_TRAINING)
        assert '"gmm":' in src and "scripts/rebuild_pipeline.py" in src, (
            "run_gmm must go through rebuild_pipeline so {GMM, parquets} "
            "are refit as ONE set (P215) at the strictest fold boundary")
        # strip docstrings: the module header legitimately QUOTES the
        # phantom path when explaining its removal (the P177 comment trap)
        assert "scripts/retrain_gmm.py" not in code_only(
            RUN_TRAINING, strip_docstrings=True), (
            "retrain_gmm.py trains the legacy GLOBAL model the runtime "
            "treats as fallback only (P189)")

    def test_run_drl_passes_decision_interval_and_fresh_tag(self):
        src = code_only(RUN_TRAINING)
        assert "'--decision-interval', '4'" in src or \
               '"--decision-interval", "4"' in src, (
            "orchestrated DRL runs lost --decision-interval 4")
        assert "orchestrated_" in src and "strftime" in src, (
            "orchestrated DRL runs need a FRESH per-run tag (P200 cache "
            "gotcha)")

    def test_check_data_looks_in_drl_training_dir(self):
        src = code_only(RUN_TRAINING)
        assert "drl_training" in src, (
            "check_data must look where the parquets actually live "
            "(training_data/drl_training/), not the directory root — the "
            "old path made preflight fail on a healthy tree")


# ---------------------------------------------------------------------------
# 3. train_drl_full.py: buffer cap unconditional, P221 vocabulary in tables
# ---------------------------------------------------------------------------

class TestTrainDrlFull:
    def test_buffer_cap_is_outside_the_config_block(self):
        """The 500K cap must apply to plain CLI runs, not only --config runs.

        Structural pin: in main()'s source, the cap's `if args.buffer_size`
        statement must sit at the SAME indentation depth as `if args.config:`
        (i.e. not nested inside it).
        """
        src = read_source(TRAIN_DRL)
        cfg = re.search(r"^([ \t]*)if args\.config:", src, re.M)
        cap = re.search(
            r"^([ \t]*)if args\.buffer_size and args\.buffer_size > 500_000:",
            src, re.M)
        assert cfg and cap, "config block or buffer cap not found"
        assert len(cap.group(1)) <= len(cfg.group(1)), (
            "the 500K buffer cap is nested INSIDE `if args.config:` again — "
            "a plain CLI --buffer-size flows uncapped into the model")

    def test_steady_uptrend_is_a_bull_regime_with_position_bias(self):
        src = code_only(TRAIN_DRL)
        bull = re.search(r"BULL_REGIMES\s*=\s*frozenset\(\[(.*?)\]\)", src, re.S)
        assert bull and "STEADY_UPTREND" in bull.group(1), (
            "STEADY_UPTREND (in ETH/SOL's k=7 P221 vocabulary) must be in "
            "BULL_REGIMES or the reward is regime-blind exactly where the "
            "clean GMMs label the bull")
        bias = re.search(r"POSITION_BIAS\s*[:=].*?\{(.*?)\n\}", src, re.S)
        assert bias, "POSITION_BIAS table not found"
        assert "STEADY_UPTREND" in bias.group(1)
        assert "NEUTRAL_DRIFT" in bias.group(1), (
            "NEUTRAL_DRIFT must stay mapped (0.0) so runs on pre-P221 "
            "parquets don't hit the unmapped-regime no-op path (P184)")


# ---------------------------------------------------------------------------
# 4. main.py: the navigator confidence max-overwrite stays retired
# ---------------------------------------------------------------------------

class TestNavigatorOverwriteRetired:
    def test_no_code_overwrites_regime_confidence_from_the_navigator(self):
        src = code_only(MAIN)
        # The retired shape: comparing regime_tensor.regime_confidence
        # against agent_signals regime_confidence and assigning on win.
        assert not re.search(
            r"regime_tensor\.regime_confidence\s*>\s*agent_signals",
            src), (
            "the navigator max-overwrite is back: the online-fit 6-component "
            "navigator GMM must not overwrite the calibrated per-asset "
            "posterior under the same `regime_confidence` key (P269)")

    def test_p0_keys_still_written(self):
        src = code_only(MAIN)
        assert "agent_signals['p0_regime']" in src
        assert "agent_signals['p0_regime_confidence']" in src, (
            "the navigator's own namespaced keys must survive the "
            "retirement — retiring the OVERWRITE must not delete the signal")


# ---------------------------------------------------------------------------
# 5. market_data_pipeline: cross-asset GMM fallback refuses
# ---------------------------------------------------------------------------

class TestCrossAssetGmmRefusal:
    def test_missing_per_asset_artifact_refuses_instead_of_borrowing_btc(self):
        src = read_source(PIPELINE)
        m = re.search(
            r"elif self\._gmm_configs:(.*?)(?=\n        else:)", src, re.S)
        assert m, (
            "the per-asset-era guard branch is gone: an asset with a "
            "missing artifact pair falls through to the 'legacy' fields, "
            "which are set from the FIRST loaded per-asset model — i.e. it "
            "is silently classified with ANOTHER ASSET'S GMM (P269)")
        assert "return None" in m.group(1), (
            "the guard branch must refuse toward the ADX proxy (return "
            "None), never classify")
        assert "warning" in m.group(1).lower(), (
            "the refusal must be loud — a silent ADX fallback is "
            "indistinguishable from a healthy classification")


# ---------------------------------------------------------------------------
# 6. Docs: the claims, not the phrasing
# ---------------------------------------------------------------------------

class TestDocClaims:
    def test_guide_has_no_phantom_conversion_step(self):
        text = read_source(GUIDE)
        # The instruction shape was a bash comment telling the reader to
        # convert files with a tool that does not exist. The P269 correction
        # legitimately QUOTES the retired instruction in prose, so pin the
        # executable claim: no bash fence line may instruct the conversion.
        for m in re.finditer(r"```bash(.*?)```", text, re.S):
            for line in m.group(1).splitlines():
                assert "convert 4h" not in line.lower(), (
                    "the guide re-acquired the unexecutable 'convert 4h -> "
                    "_1d' step — no such converter exists in the tree")

    def test_guide_names_the_real_funding_fetcher(self):
        assert "fetch_binance_funding.py" in read_source(GUIDE)

    def test_claude_md_training_command_carries_the_load_bearing_flags(self):
        text = read_source(CLAUDE_MD)
        m = re.search(r"# DRL \(TQC\).*?```", text, re.S)
        assert m, "Training Commands DRL block missing from CLAUDE.md"
        blk = m.group(0)
        for flag in ("--extractor lstm_film_a", "--venue coinbase",
                     "--decision-interval 4", "--tag"):
            assert flag in blk, (
                f"CLAUDE.md's canonical DRL command lost {flag} — the bare "
                "command trains at the wrong venue/extractor and restores "
                "stale cache (P269 corrected this once already)")

    @pytest.mark.parametrize("doc", [
        "docs/HMATS_Architecture_Part4_Execution_DRL_v10.md",
        "training/README.md",
        "training/README_V4.md",
    ])
    def test_historical_docs_carry_their_banner(self, doc):
        text = read_source(REPO / doc)
        assert "HISTORICAL" in text[:2000], (
            f"{doc} lost its historical banner — its DRL/GMM claims are "
            "leaked-era and contradict the live state (P269)")
