"""[P382] The misc-fix batch — each pin names the defect it guards.

 1. seat_check: regimebook > whale > trend precedence; `quant` series labelled
    by the live primary_strategy convention.
 2. window_usage.json: the P379 regimebook_adj validation read is ledgered.
 3. scanners: exchange/, api/, portfolio/ are in LIVE_DIRS (exchange/ in mypy
    CRITICAL_DIRS); a RELATIVE CLI path no longer crashes the two linters;
    the baselines carry a _p382_note attribution.
 4. docker-compose healthcheck: stale `updated_at` FAILS, fresh passes.
 5. deribit: a PARTIAL fetch carries the failed currency forward with its
    ORIGINAL timestamp; the fresh one is fresh.
 6. ENABLE_REGIME_TRANSITION_BUFFER getattr defaults equal the declared flag.
 7. every safe_import(module, name) target resolves; a wrong name WARNs.
 8. README: the runbook no longer prescribes `--update` to re-baseline.
 9. SEPTEMBER_DECISION_TREE: regimebook section at the P298/P299/P379 state,
    tripwire prescription retired.
10. live profile notes carry the [SUPERSEDED ...] prefix; profile still loads.
11. hetzner_deploy.sh: the scan block runs under `set +e` so SCAN_RC is read.
"""
from __future__ import annotations

import ast
import asyncio
import importlib
import io
import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _read(rel: str) -> str:
    return io.open(REPO / rel, encoding="utf-8-sig").read()


# =============================================================================
# 1. seat_check precedence + quant label
# =============================================================================

class TestSeatCheckPrecedence:
    WHALE_STATS = json.dumps(
        {"whale": {"ic_4h": 0.04, "ic_16h": 0.011,
                   "t_4h": 0.98, "t_16h": 0.14, "n": 605}})

    def _run(self, capsys, cfg: dict, *extra):
        import scripts.seat_check as sc
        import tempfile
        d = tempfile.mkdtemp()
        p = Path(d) / "cfg.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        rc = sc.main(["--config", str(p), *extra])
        out = capsys.readouterr().out
        return rc, out

    def test_regimebook_outranks_whale(self, capsys):
        """P298: whale defers to a directional book, so with both enforced
        the BOOK holds the DECIDE slot."""
        rc, out = self._run(capsys, {"whale_seat_mode": "enforce",
                                     "regimebook_mode": "enforce",
                                     "trend_following_mode": "enforce"},
                            "--stats", self.WHALE_STATS)
        assert re.search(r"incumbent\s*:\s*regimebook", out), out

    def test_whale_outranks_trend_when_no_book(self, capsys):
        rc, out = self._run(capsys, {"whale_seat_mode": "enforce",
                                     "regimebook_mode": "off",
                                     "trend_following_mode": "enforce"},
                            "--stats", self.WHALE_STATS)
        assert re.search(r"incumbent\s*:\s*whale", out), out

    def test_trend_is_last(self, capsys):
        rc, out = self._run(capsys, {"trend_following_mode": "enforce"},
                            "--stats", self.WHALE_STATS)
        assert re.search(r"incumbent\s*:\s*trend", out), out

    def _ic_report(self, tmp_path) -> Path:
        rep = tmp_path / "ic.json"
        rep.write_text(json.dumps({"agents": {
            "quant": {"horizons": {"1": {"ic": 0.01, "t": 0.5, "n": 400},
                                   "4": {"ic": 0.02, "t": 0.6, "n": 390}}},
            "whale": {"horizons": {"1": {"ic": 0.04, "t": 0.98, "n": 605},
                                   "4": {"ic": 0.011, "t": 0.14, "n": 600}}},
        }}), encoding="utf-8")
        return rep

    def test_quant_series_is_labelled_regimebook_under_enforce(self, capsys, tmp_path):
        """`quant` IS whoever holds the seat; main.py stamps
        primary_strategy="regimebook" under regimebook_mode enforce (P313)."""
        rep = self._ic_report(tmp_path)
        rc, out = self._run(capsys, {"regimebook_mode": "enforce",
                                     "whale_seat_mode": "enforce"},
                            "--ic-report", str(rep))
        assert re.search(r"^\s+regimebook\s+[+-]\d", out, re.M), out
        assert not re.search(r"^\s+trend\s+[+-]\d", out, re.M), out

    def test_quant_series_is_labelled_trend_without_the_book(self, capsys, tmp_path):
        rep = self._ic_report(tmp_path)
        rc, out = self._run(capsys, {"regimebook_mode": "off",
                                     "trend_following_mode": "enforce"},
                            "--ic-report", str(rep))
        assert re.search(r"^\s+trend\s+[+-]\d", out, re.M), out

    def test_unreadable_config_still_refuses_without_incumbent(self, capsys):
        import scripts.seat_check as sc
        rc = sc.main(["--config", "/no/such/cfg.json",
                      "--stats", self.WHALE_STATS])
        assert rc == 2

    def test_the_stale_P250_sol_claim_is_gone(self):
        src = _read("scripts/seat_check.py")
        assert "structurally inert" not in src
        assert "v1_trend_only" in src and "P299" in src


# =============================================================================
# 2. window ledger backfill
# =============================================================================

class TestWindowLedgerBackfill:
    def test_the_p379_read_is_ledgered_per_asset(self):
        d = json.loads(_read("training/reports/window_usage.json"))
        recs = [r for r in d["records"]
                if r["experiment"] == "regimebook_adj_validation_read:p379"]
        assert {r["asset"] for r in recs} == {"BTC", "ETH", "SOL"}
        for r in recs:
            # the P287 prefix rule: it must COUNT as a validation read
            assert r["purpose"].lower().startswith("validation")
            assert r["start"] == 9100 and r["end"] >= 13034
            assert "backfill" in r["note"].lower()

    def test_the_counter_sees_it(self):
        from training import splits
        assert "regimebook_adj_validation_read:p379" in splits.validation_spend("BTC")


# =============================================================================
# 3. scanner coverage + relative-path crash + baselines
# =============================================================================

class TestScannerCoverage:
    @pytest.mark.parametrize("mod", [
        "tools.lint_silent_swallow", "tools.lint_naive_datetime",
        "tools.lint_self_config_undefined",
    ])
    def test_live_dirs_cover_the_venue_layer(self, mod):
        m = importlib.import_module(mod)
        for d in ("exchange", "api", "portfolio"):
            assert d in m.LIVE_DIRS, (mod, d)

    def test_silent_failure_audit_live_dirs(self):
        src = _read("scripts/silent_failure_audit.py")
        blk = src[src.index("LIVE_DIRS = ["):src.index("SKIP_DIRS")]
        for d in ("exchange", "api", "portfolio"):
            assert f'"{d}"' in blk, d

    def test_mypy_critical_dirs_include_exchange(self):
        from tools.lint_mypy_baseline import CRITICAL_DIRS
        assert "exchange" in CRITICAL_DIRS

    @pytest.mark.parametrize("tool", ["tools/lint_naive_datetime.py",
                                      "tools/lint_self_config_undefined.py"])
    def test_a_relative_cli_path_does_not_crash(self, tool):
        r = subprocess.run([sys.executable, "-X", "utf8", tool, "exchange"],
                           cwd=str(REPO), capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        assert "Traceback" not in r.stderr, r.stderr[-800:]
        assert "relative_to" not in r.stderr

    @pytest.mark.parametrize("name,keys", [
        ("silent_swallow_baseline.json", {"total_count": 704}),
        ("silent_failure_baseline.json", {"tryexcept_count": 725, "dictget_count": 44}),  # [P407j] +1 skew+ETF combo; [P409] +1 held-BTC-ridge shadow wiring
        ("mypy_baseline.json", {"total_count": 1084, "mypy_version": "2.3.0"}),
    ])
    def test_baselines_were_bumped_with_attribution(self, name, keys):
        d = json.loads(_read(f"tools/scanner_baselines/{name}"))
        assert "_p382_note" in d and "exchange" in d["_p382_note"]
        for k, v in keys.items():
            assert d[k] == v, (name, k, d[k])


# =============================================================================
# 4. docker-compose healthcheck age
# =============================================================================

class TestHealthcheckAge:
    def _probe(self):
        yaml = pytest.importorskip("yaml")
        d = yaml.safe_load(_read("docker-compose.hetzner.yml"))
        test = d["services"]["hmats-engine"]["healthcheck"]["test"]
        assert test[:3] == ["CMD", "python", "-c"]
        return test[3]

    @pytest.mark.parametrize("age_h,expect_rc", [(1, 0), (9, 1)])
    def test_stale_updated_at_fails(self, tmp_path, age_h, expect_rc):
        code = self._probe()
        p = tmp_path / "dashboard_state.json"
        ts = (datetime.now(timezone.utc) - timedelta(hours=age_h)).isoformat()
        ts = ts.replace("+00:00", "Z")  # the P323 wire format
        p.write_text(json.dumps({"updated_at": ts}), encoding="utf-8")
        code = code.replace("/opt/hmats/data/dashboard_state.json",
                            str(p).replace("\\", "/"))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        assert r.returncode == expect_rc, (r.stdout, r.stderr[-300:])

    def test_missing_stamp_fails(self, tmp_path):
        code = self._probe()
        p = tmp_path / "dashboard_state.json"
        p.write_text(json.dumps({"no": "stamp"}), encoding="utf-8")
        code = code.replace("/opt/hmats/data/dashboard_state.json",
                            str(p).replace("\\", "/"))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True)
        assert r.returncode != 0


# =============================================================================
# 5. deribit partial-fetch carry
# =============================================================================

class TestDeribitPartialCarry:
    def _feed(self, monkeypatch):
        from data_mgmt.feeds import deribit_feed as df
        from data_mgmt.feeds import _http

        class _Sess:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False

        monkeypatch.setattr(_http, "create_session", lambda **kw: _Sess())
        feed = df.DeribitFeed(poll_interval_sec=0.0)
        return df, feed

    def test_failed_currency_keeps_previous_with_original_timestamp(self, monkeypatch, caplog):
        df, feed = self._feed(monkeypatch)
        calls = {"n": 0}
        t1 = datetime(2026, 8, 22, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 8, 22, 10, 15, tzinfo=timezone.utc)

        async def fake(session, cur, now):
            if calls["n"] == 0:
                return df.DeribitOptionsMetrics(
                    currency=cur, put_call_ratio_oi=0.5 if cur == "BTC" else 0.7,
                    total_oi_calls=10.0, instrument_count=3, timestamp=t1)
            if cur == "ETH":
                raise RuntimeError("boom")
            return df.DeribitOptionsMetrics(
                currency=cur, put_call_ratio_oi=0.55, total_oi_calls=11.0,
                instrument_count=3, timestamp=t2)

        monkeypatch.setattr(feed, "_fetch_currency", fake)
        asyncio.run(feed.fetch())
        calls["n"] = 1
        with caplog.at_level(logging.WARNING):
            snap = asyncio.run(feed.fetch())

        assert snap.get("BTC").put_call_ratio_oi == 0.55
        assert snap.get("BTC").timestamp == t2, "the fresh currency is fresh"
        eth = snap.get("ETH")
        assert eth is not None, "ETH's previous reading must survive the partial fetch"
        assert eth.put_call_ratio_oi == 0.7
        assert eth.timestamp == t1, "a carried reading keeps its ORIGINAL stamp"
        assert any("carried_forward:ETH" == e for e in snap.errors), snap.errors
        assert any("carried forward" in r.message and "ETH" in r.message
                   for r in caplog.records)
        # and the carry is visible through the public accessor
        assert feed.get_options_metrics("ETH").timestamp == t1

    def test_total_failure_still_keeps_the_whole_cache(self, monkeypatch):
        df, feed = self._feed(monkeypatch)
        t1 = datetime(2026, 8, 22, 10, 0, tzinfo=timezone.utc)
        state = {"fail": False}

        async def fake(session, cur, now):
            if state["fail"]:
                raise RuntimeError("down")
            return df.DeribitOptionsMetrics(
                currency=cur, put_call_ratio_oi=0.5, total_oi_calls=10.0,
                instrument_count=3, timestamp=t1)

        monkeypatch.setattr(feed, "_fetch_currency", fake)
        asyncio.run(feed.fetch())
        state["fail"] = True
        snap = asyncio.run(feed.fetch())
        assert snap.get("BTC").timestamp == t1 and snap.get("ETH").timestamp == t1

    def test_a_currency_never_seen_is_not_fabricated(self, monkeypatch):
        """Carry only what was previously MEASURED: a currency absent from
        both cycles stays absent (P2)."""
        df, feed = self._feed(monkeypatch)

        async def fake(session, cur, now):
            if cur == "ETH":
                raise RuntimeError("never")
            return df.DeribitOptionsMetrics(
                currency=cur, put_call_ratio_oi=0.5, total_oi_calls=10.0,
                instrument_count=3, timestamp=now)

        monkeypatch.setattr(feed, "_fetch_currency", fake)
        asyncio.run(feed.fetch())
        snap = asyncio.run(feed.fetch())
        assert snap.get("ETH") is None
        assert not any(e.startswith("carried_forward") for e in snap.errors)


# =============================================================================
# 6. ENABLE_REGIME_TRANSITION_BUFFER defaults agree with the declaration
# =============================================================================

class TestRegimeBufferDefaultAgreesWithDeclaration:
    def _declared(self) -> bool:
        import dataclasses
        from configs.sota_flags import SOTAFlags
        for f in dataclasses.fields(SOTAFlags):
            if f.name == "ENABLE_REGIME_TRANSITION_BUFFER":
                return f.default
        raise AssertionError("flag not declared")

    def _getattr_defaults(self, rel: str):
        tree = ast.parse(_read(rel))
        out = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr" and len(node.args) == 3
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "ENABLE_REGIME_TRANSITION_BUFFER"):
                assert isinstance(node.args[2], ast.Constant), (rel, node.lineno)
                out.append((node.lineno, node.args[2].value))
        return out

    @pytest.mark.parametrize("rel", ["defense/trade_gate.py",
                                     "risk/regime_transition_buffer.py",
                                     "defense/governor_integration.py"])
    def test_every_getattr_default_equals_the_declared_flag(self, rel):
        declared = self._declared()
        sites = self._getattr_defaults(rel)
        assert sites, f"no getattr read of the flag in {rel} — the pin is vacuous"
        for line, default in sites:
            assert default == declared, (
                f"{rel}:{line} defaults ENABLE_REGIME_TRANSITION_BUFFER to "
                f"{default!r}; configs/sota_flags.py declares {declared!r} "
                f"(P338/P382)")

    def test_the_declared_value_is_still_False(self):
        """If the declaration flips, every default above must flip WITH it —
        this pin makes that a visible decision."""
        assert self._declared() is False


# =============================================================================
# 7. safe_import targets resolve; wrong names are loud
# =============================================================================

class TestSafeImportNames:
    ARCHIVED = {"infra.event_replay", "core.plugin_registry"}

    def test_every_named_attribute_exists_in_its_module(self):
        src = _read("orchestration/sota_integration.py")
        pairs = re.findall(r'safe_import\(\s*"([^"]+)"\s*,\s*"([^"]+)"', src)
        assert len(pairs) >= 20, "the scan found too few targets to be the real roster"
        missing = []
        for mod, name in pairs:
            if mod in self.ARCHIVED:
                continue
            m = importlib.import_module(mod)
            if not hasattr(m, name):
                missing.append(f"{mod}.{name}")
        assert not missing, f"safe_import names that do not exist: {missing}"

    def test_rl_fallback_binding_is_the_real_class(self):
        from orchestration import sota_integration as si
        from execution.rl_fallback import RLExecutionFallbackManager
        assert si.RLExecutionFallback is RLExecutionFallbackManager

    def test_a_wrong_class_name_WARNs_instead_of_returning_None_silently(self, caplog):
        from orchestration.sota_integration import safe_import
        with caplog.at_level(logging.WARNING):
            got = safe_import("execution.rl_fallback", "RLExecutionFallback")
        assert got is None
        assert any(r.levelno >= logging.WARNING and "RLExecutionFallback" in r.message
                   for r in caplog.records), caplog.text

    def test_a_correct_name_stays_quiet(self, caplog):
        from orchestration.sota_integration import safe_import
        with caplog.at_level(logging.WARNING):
            got = safe_import("execution.rl_fallback", "RLExecutionFallbackManager")
        assert got is not None
        assert not [r for r in caplog.records if "rl_fallback" in r.message]


# =============================================================================
# 8 / 9 / 10 / 11 — docs, profile notes, deploy script
# =============================================================================

class TestDocsAndNotes:
    def test_readme_runbook_no_longer_prescribes_update(self):
        md = _read("README.md")
        row = [l for l in md.splitlines()
               if l.startswith("| Scanner baseline INCREASE")]
        assert len(row) == 1, row
        assert "hand-edit ONLY the counters that moved" in row[0]
        assert "Never run" in row[0]
        assert "re-baseline: `python" not in row[0]

    def test_decision_tree_regimebook_section_is_current(self):
        md = _read("docs/SEPTEMBER_DECISION_TREE.md")
        i = md.index("## ~Sep 9 — regimebook raw + adjusted")
        j = md.index("## ~Sep 9 — derivflow")
        sec = md[i:j]
        assert 'regimebook_mode: "shadow"' in sec, "the revert must be named"
        assert "OVERFIT" in sec and "P379" in sec
        assert "stays OFF regardless" in sec
        assert "P298" in sec and "v1_trend_only" in sec

    def test_decision_tree_tripwire_prescription_is_retired(self):
        md = _read("docs/SEPTEMBER_DECISION_TREE.md")
        i = md.index("## Sep 1 — the trend tripwire")
        j = md.index("## ~Sep 7 — ma_filter")
        sec = md[i:j]
        assert "PRESCRIPTION RETIRED" in sec and "P299" in sec
        assert "seat controller" in sec.lower() or "seat_check" in sec
        row = [l for l in sec.splitlines() if l.startswith("| 4/4 GATE-CLOSED")]
        assert len(row) == 1 and "no config edit" in row[0], row

    @pytest.mark.parametrize("key", ["_description", "_drl_note",
                                     "_coinbase_use_gated_intent_note",
                                     "_trend_layer_note",
                                     "_coinbase_protective_stop_note"])
    def test_live_profile_notes_are_marked_superseded(self, key):
        d = json.loads(_read("configs/live_high_risk.json"))
        assert str(d[key]).startswith("[SUPERSEDED 2026-08-2"), key
        assert "ORIGINAL:" in d[key], "the original text must be kept, not rewritten"

    def test_live_profile_still_loads(self):
        from main import ProductionConfig
        cfg = ProductionConfig.from_file(REPO / "configs" / "live_high_risk.json")
        # the notes are documentation; the DECIDED values are unchanged
        assert cfg.regimebook_mode == "enforce"
        assert cfg.trend_following_mode == "enforce"

    def test_deploy_scan_block_disarms_errexit(self):
        sh = _read("scripts/hetzner_deploy.sh")
        i = sh.index("[P328] Scan the COMMIT")
        j = sh.index('if [ ${SCAN_RC} -ne 0 ]')
        blk = sh[i:j]
        # the first REAL (indented) assignment — the explanatory comment above
        # the block also spells `SCAN_RC=$?`, and a scanner that matches its
        # own explanation is worthless (P177)
        first_rc = re.search(r"^\s+SCAN_RC=\$\?", blk, re.M).start()
        assert re.search(r"^set \+e\s*$", blk[:first_rc], re.M), (
            "under set -e a failing scan exits before SCAN_RC is read and the "
            "failure message never prints")
        assert blk.rstrip().endswith("set -e"), "errexit must be re-armed after the block"
