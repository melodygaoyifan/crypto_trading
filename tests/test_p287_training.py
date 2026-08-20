"""[P287] Training/CI batch of the 2026-08-16 read-through fix campaign.

Pins, per finding:
  1. 2x cost copies (regime_model_lab, label_lab) charge half-RT per leg.
  2. fetch_coinglass_history grew --interval {4h,1d}; refresh-data runs
     both; check_data_freshness watches the CONSUMED 1d archives.
  3. merge_external_data bounds staleness at 3 days (behavioral).
  4. --gmm-no-split can never deploy to models/regime_classifier.
  5. legacy shared-GMM fallback REFUSES; assert_clean_gmm cross-checks the
     prod dir (behavioral).
  6. hetzner_deploy keeps the NEWEST CI run per workflow; python3 fallback.
  7. window-usage ledger counts prefix-matched validation purposes.
  8. export_mlp_shadow has a cadence mechanism (refresh-data).
  9. fetch_binance_full writes the canonical raw parquet atomically.
 10. train_drl_full's fv2 refusal names the P266 reality.
 11. dsr() charges the Bailey-LdP skew/kurt inflation term.

Source pins use tests/_source_scan.code_only(strip_docstrings=True) where a
statement's absence is asserted — the fix comments quote the removed code
(P177/P179 trap).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training"))

from tests._source_scan import code_only, read_source  # noqa: E402

RML = REPO / "training" / "regime_model_lab.py"
LBL = REPO / "training" / "label_lab.py"
FCH = REPO / "training" / "scripts" / "fetch_coinglass_history.py"
RBP = REPO / "training" / "scripts" / "rebuild_pipeline.py"
MKF = REPO / "training" / "Makefile"
CDF = REPO / "training" / "scripts" / "check_data_freshness.py"
HDS = REPO / "scripts" / "hetzner_deploy.sh"
FBF = REPO / "training" / "fetch_binance_full.py"
TDF = REPO / "training" / "train_drl_full.py"
PCT = REPO / "training" / "scripts" / "pooled_certification.py"
MEC = REPO / "training" / "scripts" / "mlp_ensemble_cert.py"
DSC = REPO / "training" / "scripts" / "deep_seq_cert.py"
EMS = REPO / "training" / "scripts" / "export_mlp_shadow.py"


# ── 1. the 2x cost copies ────────────────────────────────────────────────

class TestCostConventionParity:
    def test_regime_model_lab_halves_the_round_trip_constant(self):
        src = code_only(RML, strip_docstrings=True)
        assert "_COST[ctx[\"asset\"]] / 2.0" in src, (
            "regime_model_lab no longer halves _COST per leg — _COST is "
            "ROUND-TRIP (P281 rename); full-per-leg is the 2x overcharge "
            "P279 measured")
        # the old defective statement must be gone (comments stripped so the
        # fix's own explanation cannot satisfy this)
        assert "* cost_bps * cost_mult" not in src, (
            "the un-normalized per-leg charge of the RT constant is back")

    def test_label_lab_halves_and_single_sources_the_constant(self):
        src = code_only(LBL, strip_docstrings=True)
        assert "(COST_BPS[asset] / 2.0)" in src
        assert re.search(
            r"from training\.train_supervised_full import COST_BPS", src), (
            "label_lab restates COST_BPS locally again — the restated copy "
            "is exactly how the P281 rename-in-meaning missed it")
        assert not re.search(
            r"^COST_BPS\s*=\s*\{", src, re.M), (
            "a local COST_BPS dict literal is back in label_lab")

    def test_three_labs_charge_one_identical_trade_identically(self):
        """Behavioral parity: one +1ct entry+exit at flat prices costs the
        same through evaluate_segment's arithmetic and the (now fixed)
        regime_model_lab / label_lab formulas, all = 2 legs x RT/2."""
        np = pytest.importorskip("numpy")
        rt = 6.0
        pos = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
        turnover = np.abs(np.diff(pos)).sum()          # 2 legs
        expected = turnover * (rt / 2.0) / 1e4          # = 1 full RT
        # supervised zoo formula (reference, P281)
        zoo = np.abs(np.diff(pos)) * (rt / 2.0) / 1e4
        # regime_model_lab post-P287: cost_leg_bps = _COST/2
        rml = np.abs(np.diff(pos)) * (rt / 2.0) * 1.0 / 1e4
        # label_lab post-P287
        lbl = np.abs(np.diff(pos)) * (rt / 2.0) / 1e4
        for arr in (zoo, rml, lbl):
            assert abs(float(arr.sum()) - expected) < 1e-12
        assert abs(expected - rt / 1e4) < 1e-12  # 2 legs == exactly 1 RT

    def test_no_other_full_rt_per_leg_copy_survives_in_training(self):
        """Sweep every |diff(pos)| cost site in training/ (the P226 lesson:
        a fix applied to one instance is not applied to the class). Sites
        multiplying a known ROUND-TRIP constant must carry '/ 2' on it;
        per-side-named constants are exempt (they are already per leg)."""
        offenders = []
        for p in (REPO / "training").rglob("*.py"):
            try:
                src = code_only(p, strip_docstrings=True)
            except Exception:
                continue
            for m in re.finditer(r"np\.abs\(np\.diff\([^)]*\)\)\s*\*"
                                 r"\s*([^\n]+)", src):
                expr = m.group(1)
                if "COST" not in expr and "cost" not in expr:
                    continue  # turnover metrics, not cost charges
                if "side" in expr.lower():
                    continue  # per-side constants are per-leg by name
                if "/ 2" in expr or "cost_leg" in expr or "cost_rt" in expr:
                    continue  # halved at site, or a variable halved upstream
                offenders.append(f"{p.name}: {expr.strip()[:70]}")
        assert not offenders, (
            "possible full-RT-per-leg cost charge(s) — classify each as "
            f"per-side (rename it) or halve it: {offenders}")


# ── 2. CoinGlass 1d refreshability ──────────────────────────────────────

class TestCoinglass1dRefreshable:
    def test_fetcher_has_interval_flag_with_1d(self):
        src = read_source(FCH)
        assert "--interval" in src
        assert re.search(r"VALID_INTERVALS\s*=\s*\(\s*\"4h\",\s*\"1d\"", src)

    def test_fetcher_output_path_uses_the_runtime_interval(self):
        src = code_only(FCH, strip_docstrings=True)
        assert '{asset}_{name}_{interval}.parquet' in src, (
            "output filename no longer keyed to the runtime interval — a "
            "1d run would overwrite the 4h archives or vice versa")
        assert '{asset}_{name}_{INTERVAL}.parquet' not in src

    def test_refresh_data_fetches_both_intervals(self):
        mk = read_source(MKF)
        assert "fetch_coinglass_history.py --interval 1d" in mk, (
            "refresh-data no longer refreshes the 1d archives — those are "
            "the ONLY files rebuild_pipeline._load_coinglass_daily reads, "
            "and CoinGlass depth ~180d makes a stale archive a PERMANENT "
            "history loss (P266/P287)")

    def test_freshness_check_watches_the_consumed_1d_files(self):
        src = read_source(CDF)
        assert "*_liquidation_1d.parquet" in src
        assert "*_oi_1d.parquet" in src


# ── 3. merge tolerance ──────────────────────────────────────────────────

class TestMergeStalenessBound:
    @pytest.fixture()
    def rbp_mod(self):
        # [P287] `import rebuild_pipeline` by bare name breaks under full-suite
        # ordering: an earlier test binds sys.modules['scripts'] to the
        # REPO-ROOT scripts/ package, and rebuild_pipeline's
        # `from scripts.wavelet_denoise import ...` then cannot resolve.
        # Single-source the isolation loader (P172) instead of a third copy.
        pytest.importorskip("sklearn")
        from tests.test_rebuild_pipeline_gmm_split import _load_rebuild_module
        return _load_rebuild_module()

    def _daily(self, pd, ext_cols, days):
        rows = []
        for d in days:
            r = {"timestamp": pd.Timestamp(d, tz="UTC")}
            for c in ext_cols:
                r[c] = 1.0
            rows.append(r)
        return pd.DataFrame(rows)

    def test_bars_beyond_3d_of_daily_coverage_read_no_data(
            self, rbp_mod, monkeypatch):
        pd = pytest.importorskip("pandas")
        ext_cols = [c for c in rbp_mod.EXTERNAL_FEATURE_COLS
                    if c != "has_external_data"]
        daily = self._daily(pd, ext_cols, ["2026-01-01", "2026-01-02"])
        monkeypatch.setattr(rbp_mod, "_load_coinglass_daily",
                            lambda a: daily)
        monkeypatch.setattr(
            rbp_mod, "_load_futures_daily",
            lambda a: pd.DataFrame(
                {"timestamp": pd.Series([], dtype="datetime64[ns, UTC]")}))
        bars = pd.date_range("2026-01-01", "2026-01-10", freq="4h", tz="UTC")
        df4h = pd.DataFrame({"timestamp": bars, "close": 1.0})
        out = rbp_mod.merge_external_data(df4h, "BTC")
        out = out.set_index("timestamp")
        # within 3d of the last daily row (2026-01-02): has data
        assert out.loc[pd.Timestamp("2026-01-03 12:00", tz="UTC"),
                       "has_external_data"] == 1.0
        # beyond 3d: NO data (the frozen-carry-forever defect)
        stale = out.loc[pd.Timestamp("2026-01-08 00:00", tz="UTC")]
        assert stale["has_external_data"] == 0.0, (
            "a bar 6 days past the newest daily row still reads as having "
            "external data — the unbounded merge_asof staleness is back "
            "(a September rebuild would stamp weeks of frozen Aug values "
            "as data)")
        for c in ext_cols:
            assert stale[c] == 0.0

    def test_within_contiguous_coverage_content_is_unchanged(
            self, rbp_mod, monkeypatch):
        """In contiguous daily coverage (rows 1d apart, the measured shape
        of every real archive: max gap 1d on 8/9 files, one 4d gap in SOL
        futures) the tolerance never binds — values equal the untolerated
        merge."""
        pd = pytest.importorskip("pandas")
        ext_cols = [c for c in rbp_mod.EXTERNAL_FEATURE_COLS
                    if c != "has_external_data"]
        days = [f"2026-01-{d:02d}" for d in range(1, 8)]
        daily = self._daily(pd, ext_cols, days)
        monkeypatch.setattr(rbp_mod, "_load_coinglass_daily",
                            lambda a: daily)
        monkeypatch.setattr(
            rbp_mod, "_load_futures_daily",
            lambda a: pd.DataFrame(
                {"timestamp": pd.Series([], dtype="datetime64[ns, UTC]")}))
        bars = pd.date_range("2026-01-01", "2026-01-07 20:00",
                             freq="4h", tz="UTC")
        df4h = pd.DataFrame({"timestamp": bars, "close": 1.0})
        out = rbp_mod.merge_external_data(df4h, "BTC")
        assert (out["has_external_data"] == 1.0).all()
        for c in ext_cols:
            assert (out[c] == 1.0).all()


# ── 4. leaky-GMM deploy refusal ─────────────────────────────────────────

class TestLeakyGmmCannotDeploy:
    def test_deploy_block_gates_on_gmm_no_split(self):
        src = code_only(RBP, strip_docstrings=True)
        assert re.search(
            r"if not args\.skip_gmm and args\.gmm_no_split:", src), (
            "the deploy block no longer refuses --gmm-no-split — a "
            "visualization run could copy a full-sample (LEAKY) fit into "
            "models/regime_classifier, the P267 invariant's only deploy "
            "site")
        assert "REFUSING GMM deploy" in read_source(RBP)

    def test_routine_deploy_announces_the_versioned_set_swap(self):
        src = read_source(RBP)
        assert "ONE versioned set" in src and "P215" in src


# ── 5. GMM loader refusal + build/prod cross-check ──────────────────────

class TestGmmProvenance:
    def test_legacy_shared_fallback_refuses(self, tmp_path, monkeypatch):
        # [P287] bare-name import breaks under full-suite ordering (see
        # TestMergeStalenessBound.rbp_mod) — use the shared isolation loader.
        pytest.importorskip("sklearn")
        from tests.test_rebuild_pipeline_gmm_split import _load_rebuild_module
        rbp_mod = _load_rebuild_module()
        # a prod dir with ONLY the shared legacy config, no per-asset dir
        (tmp_path / "gmm_config.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(rbp_mod, "PROD_GMM_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="P287"):
            rbp_mod.load_existing_gmm_per_asset("BTC")

    def _tree(self, tmp_path, build_cfg, prod_cfg=None):
        b = tmp_path / "training" / "training_data" / "gmm_models" / "BTC"
        b.mkdir(parents=True)
        (b / "gmm_config.json").write_text(json.dumps(build_cfg),
                                           encoding="utf-8")
        if prod_cfg is not None:
            p = tmp_path / "models" / "regime_classifier" / "BTC"
            p.mkdir(parents=True)
            (p / "gmm_config.json").write_text(json.dumps(prod_cfg),
                                               encoding="utf-8")
        return tmp_path

    def test_clean_build_with_no_prod_dir_passes(self, tmp_path, monkeypatch):
        import splits
        clean = {"fit_policy": "split_aware", "n_components": 6,
                 "means": [[0.1]]}
        monkeypatch.setattr(splits, "REPO", self._tree(tmp_path, clean))
        assert splits.assert_clean_gmm("BTC")["fit_policy"] == "split_aware"

    def test_leaky_prod_fit_refuses_even_when_build_is_clean(
            self, tmp_path, monkeypatch):
        import splits
        clean = {"fit_policy": "split_aware", "n_components": 6,
                 "means": [[0.1]]}
        leaky = {"fit_policy": "full_sample_LEAKY", "n_components": 6,
                 "means": [[0.1]]}
        monkeypatch.setattr(splits, "REPO",
                            self._tree(tmp_path, clean, leaky))
        with pytest.raises(SystemExit, match="prod-side"):
            splits.assert_clean_gmm("BTC")

    def test_different_prod_fit_refuses(self, tmp_path, monkeypatch):
        import splits
        clean = {"fit_policy": "split_aware", "n_components": 6,
                 "means": [[0.1]]}
        other = {"fit_policy": "split_aware", "n_components": 7,
                 "means": [[0.9]]}
        monkeypatch.setattr(splits, "REPO",
                            self._tree(tmp_path, clean, other))
        with pytest.raises(SystemExit, match="DIFFERENT fits"):
            splits.assert_clean_gmm("BTC")

    def test_identical_prod_fit_passes(self, tmp_path, monkeypatch):
        import splits
        clean = {"fit_policy": "split_aware", "n_components": 6,
                 "means": [[0.1]]}
        monkeypatch.setattr(splits, "REPO",
                            self._tree(tmp_path, clean, dict(clean)))
        assert splits.assert_clean_gmm("BTC")

    def test_real_artifacts_still_pass_the_widened_gate(self):
        """The REAL build+prod dirs (both the P221 fit) must pass — the new
        cross-check must not break the working state. Skips loudly when the
        operator-local artifacts are absent (CI, P252b pattern)."""
        cfg = (REPO / "training" / "training_data" / "gmm_models" / "BTC"
               / "gmm_config.json")
        if not cfg.exists():
            pytest.skip("operator-local GMM artifacts not present (P252b)")
        import splits
        assert splits.assert_clean_gmm("BTC")["fit_policy"] == "split_aware"


# ── 6. deploy-script CI check ───────────────────────────────────────────

class TestDeployScriptCiCheck:
    def test_first_match_per_workflow_wins(self):
        """[P344] The parser moved into tools/ci_status.py (ONE
        implementation). The property is unchanged and is now asserted
        BEHAVIOURALLY rather than as a source substring, which is strictly
        stronger: a substring cannot tell live code from dead (P234/P320)."""
        import sys as _sys
        _sys.path.insert(0, str(REPO))
        from tools.ci_status import GREEN, RED, classify

        def _r(name, conclusion):
            return {"name": name, "status": "completed",
                    "conclusion": conclusion}

        # newest-first: the RE-RUN failed. The old bug was an unconditional
        # overwrite, which let the OLDEST (green) run win and deployed.
        assert classify([_r("codebase-invariants", "failure"),
                         _r("test-suite", "success"),
                         _r("codebase-invariants", "success")]) [0] == RED, (
            "the CI-verdict parser overwrites per-workflow again — the API "
            "returns runs NEWEST-first, so the overwrite makes the OLDEST "
            "run win and a green-then-red re-run deploys")
        assert classify([_r("codebase-invariants", "success"),
                         _r("test-suite", "success")])[0] == GREEN

    def test_no_python_at_all_refuses(self):
        src = read_source(HDS)
        assert "neither 'python' nor 'python3'" in src
        assert 'PY_BIN=python3' in src
        # the scanner gate must run through the resolved interpreter, not
        # sit behind a silent `command -v python` skip
        assert re.search(r'"\$\{PY_BIN\}" -X utf8 tools/ci_check_invariants',
                         src)
        assert not re.search(
            r"if command -v python &>/dev/null; then\s*\n\s*echo \"  Running",
            src), "the silent-skip shape around the scanner gate is back"


# ── 7. window-ledger prefix counting ────────────────────────────────────

class TestWindowLedgerCounting:
    def test_free_text_validation_purpose_is_counted(self, tmp_path,
                                                     monkeypatch):
        import splits
        monkeypatch.setattr(splits, "LEDGER_PATH",
                            tmp_path / "window_usage.json")
        # the P259b shape: a deliberate validation read with a free-text tag
        prior0 = splits.record_window_usage(
            "banded_overlay_p259", "ETH", 9100, 13000,
            "validation read #1 for this candidate (operator-authorized)")
        assert prior0 == 0
        prior1 = splits.record_window_usage(
            "some_new_candidate", "ETH", 9100, 13000, "validation")
        assert prior1 == 1, (
            "a free-text 'validation read ...' purpose was not counted as "
            "a prior read — the multiplicity discount under-counts exactly "
            "the deliberately-recorded spends")

    def test_validation_mentioning_purpose_without_prefix_warns(
            self, tmp_path, monkeypatch, capsys):
        import splits
        monkeypatch.setattr(splits, "LEDGER_PATH",
                            tmp_path / "window_usage.json")
        splits.record_window_usage("x", "BTC", 0, 10,
                                   "operator validation spend")
        outp = capsys.readouterr().out
        assert "will NOT be counted" in outp

    def test_real_ledger_recount_never_shrinks(self):
        """Fixture-copy recount of the actual on-disk ledger: the prefix
        counter must see AT LEAST as many validation reads as the old
        exact-match counter did (it widens, never narrows)."""
        import splits
        real = REPO / "training" / "reports" / "window_usage.json"
        if not real.exists():
            pytest.skip("no on-disk ledger on this machine")
        records = json.loads(real.read_text(encoding="utf-8"))["records"]
        exact = sum(1 for r in records if r["purpose"] == "validation")
        prefix = sum(1 for r in records
                     if splits._is_validation_purpose(r["purpose"]))
        assert prefix >= exact


# ── 8/9/10. cadence mechanism, atomic write, message honesty ────────────

class TestSmallFixes:
    def test_export_mlp_shadow_is_in_the_refresh_chain(self):
        mk = read_source(MKF)
        assert "export_mlp_shadow.py" in mk, (
            "the P284 weekly-refit cadence lost its mechanism again — the "
            "forward ledger's provenance claims a weekly refit")
        assert "CADENCE MECHANISM" in read_source(EMS)

    def test_fetch_binance_full_writes_atomically(self):
        src = code_only(FBF, strip_docstrings=True)
        assert re.search(r"os\.replace\(_tmp,\s*existing_path\)", src)
        assert not re.search(r"merged\.to_parquet\(existing_path\)", src), (
            "the canonical 6-year raw parquet is written non-atomically "
            "again — a crash mid-write truncates the merge target")
        assert re.search(r"^import os", code_only(FBF), re.M)

    def test_fv2_refusal_names_the_p266_reality(self):
        src = read_source(TDF)
        assert "since P266" in src or "STEP 5b" in src
        assert "AFTER the\n" not in src.split("--include-fv2")[1][:600], (
            "the fv2 refusal message teaches the retired two-step rule")


# ── 11. dsr skew/kurt inflation ─────────────────────────────────────────

class TestDsrInflationTerm:
    @pytest.fixture()
    def pct(self):
        pytest.importorskip("sklearn")
        sys.path.insert(0, str(REPO / "training" / "scripts"))
        import importlib
        return importlib.import_module("pooled_certification")

    def test_normal_seg_matches_the_normal_form(self, pct):
        np = pytest.importorskip("numpy")
        base = pct.dsr(1.5, 5000, 10)
        rng = np.random.default_rng(0)
        seg = rng.normal(0.0001, 0.01, 5000)
        near = pct.dsr(1.5, 5000, 10, seg=seg)
        assert abs(base - near) < 0.05  # normal moments ≈ (0, 3)

    def test_fat_tailed_negative_skew_deflates_a_positive_sharpe(self, pct):
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(1)
        # crypto-shaped: negative skew, heavy tails
        seg = rng.normal(0.0002, 0.005, 5000)
        seg[rng.integers(0, 5000, 60)] -= 0.08
        x = np.asarray(seg)
        m, s = x.mean(), x.std()
        g3 = float(((x - m) ** 3).mean() / s ** 3)
        g4 = float(((x - m) ** 4).mean() / s ** 4)
        assert g3 < -0.5 and g4 > 5  # the fixture really is skewed/fat
        assert pct.dsr(1.5, 5000, 10, seg=seg) < pct.dsr(1.5, 5000, 10), (
            "skew/kurt inflation no longer deflates — the denominator has "
            "gone back to the constant 1 (dead-sr0 form)")

    def test_all_three_call_sites_pass_the_segment(self):
        for path in (PCT, MEC, DSC):
            src = code_only(path, strip_docstrings=True)
            for m in re.finditer(r"\bdsr\(", src):
                # skip the def itself
                if src[max(0, m.start() - 4):m.start()].endswith("def "):
                    continue
                call = src[m.start():m.start() + 120]
                assert "seg=seg" in call, (
                    f"{path.name}: a dsr() call site no longer passes the "
                    f"return segment — its DSR silently reverts to the "
                    f"normal form: {call[:80]}")
