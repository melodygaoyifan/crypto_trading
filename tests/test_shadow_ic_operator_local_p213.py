"""[P213] The shadow-IC gate is OPERATOR-LOCAL, and says so instead of pretending.

`analytics/` is NOT excluded by .dockerignore, so this module ships in the engine
image. `training/training_data/` IS excluded (line 41), so its price series does
not. Run it in the container and every strategy returns `ohlcv_missing`, a full
table prints, and a report gets written — output indistinguishable from "the
strategies have no signal".

That exact conflation is what hid P199 for months: a check that could not run,
reporting as though it ran. The fix is not to make it runnable server-side (the
parquets are large and refreshed from Binance monthly archives on the operator's
box) but to make the wrong-environment case a REFUSAL that names its own cause.

The distinction that matters is ALL vs ANY: one asset missing its parquet is a
genuine data gap the run should report per-strategy and continue through; every
asset missing means the tool is somewhere it cannot work.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_MOD = _REPO / "analytics" / "shadow_ic" / "compute_shadow_ic.py"
_SRC = _MOD.read_text(encoding="utf-8", errors="replace")


class TestItDeclaresWhereItRuns:

    def test_the_docstring_says_operator_local(self):
        head = _SRC[:_SRC.index('"""', 3)]
        assert "OPERATOR-LOCAL" in head
        assert ".dockerignore" in head, (
            "must name WHY it cannot run server-side, not just that it cannot"
        )

    def test_it_names_the_refresh_command(self):
        head = _SRC[:_SRC.index('"""', 3)]
        assert "refresh_ohlcv_4h.py" in head

    def test_the_claim_about_dockerignore_is_true(self):
        """The docstring's reason must stay true, or it becomes P192-shaped
        folklore. Pins both halves: the data is excluded, the module is not."""
        di = (_REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()
        stripped = [ln.strip() for ln in di]
        assert "training/training_data/" in stripped, (
            "the parquets are no longer dockerignored — the docstring's "
            "operator-local rationale is now false; revisit P213"
        )
        assert not any(ln == "analytics/" for ln in stripped), (
            "analytics/ is now excluded, so the trap this guards against "
            "(module present, data absent) no longer exists"
        )


class TestWrongEnvironmentRefuses:

    def test_all_assets_unpriced_is_a_refusal(self):
        assert "REFUSING TO REPORT" in _SRC
        i = _SRC.index("REFUSING TO REPORT")
        w = _SRC[i - 1500:i + 1500]
        assert "return 2" in w, "must exit non-zero, not just print"

    def test_the_refusal_names_the_cause_and_the_fix(self):
        i = _SRC.index("REFUSING TO REPORT")
        w = _SRC[i:i + 1200]
        assert "dockerignore" in w
        assert "refresh_ohlcv_4h.py" in w

    def test_it_refuses_on_ALL_not_ANY(self):
        """One missing asset is a data gap and must still produce a report;
        only every-asset-missing is the wrong-environment signal."""
        i = _SRC.index("_priced = [v for v in per_strategy.values()")
        w = _SRC[i:i + 400]
        assert "if per_strategy and not _priced:" in w, (
            "refusing when ANY asset is unpriced would turn a routine data gap "
            "into a hard failure"
        )

    def test_it_refuses_before_writing_a_report(self):
        """A report on disk is the artifact people trust later."""
        i_ref = _SRC.index("REFUSING TO REPORT")
        i_report = _SRC.index('"generated_at": datetime.now(timezone.utc).isoformat()')
        assert i_ref < i_report

    def test_an_empty_ledger_dir_still_returns_1_not_2(self):
        """Distinct exit codes for distinct causes: 1 = no signals to score,
        2 = no prices to score them against. Collapsing them would reintroduce
        the no-signal/no-data conflation at the shell level."""
        i = _SRC.index("No shadow records loaded from")
        assert "return 1" in _SRC[i:i + 300]


class TestEndToEnd:

    def test_running_it_without_prices_refuses_loudly(self, tmp_path):
        """Behavioural, not source-level: build a ledger with a bogus asset that
        can have no parquet, and assert the process refuses."""
        led = tmp_path / "ledgers"
        led.mkdir()
        from datetime import datetime, timedelta, timezone
        ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        (led / "microstructure_ZZZ.jsonl").write_text(
            '{"ts": "%s", "asset": "ZZZ", "strategy": "ofi", "signal": 0.5}\n' % ts,
            encoding="utf-8")
        r = subprocess.run(
            [sys.executable, "-X", "utf8", str(_MOD),
             "--ledger-dir", str(led), "--window-days", "30"],
            capture_output=True, text=True, cwd=str(_REPO), timeout=300)
        if r.returncode == 1 and "No shadow records" in (r.stderr or ""):
            pytest.skip("ledger schema differs; source-level tests cover the gate")
        assert r.returncode == 2, (
            f"expected refusal (2), got {r.returncode}\n"
            f"STDOUT:{r.stdout[-800:]}\nSTDERR:{r.stderr[-800:]}"
        )
        assert "REFUSING TO REPORT" in (r.stderr or "")


class TestTheReviewToolShipsWhereItsDataIs:
    """[P213] The companion half. `data/trend_regime_shadow.jsonl` is written to
    the hmats-data VOLUME, so unlike the IC gate its evidence IS on the server —
    which makes the reader's absence from the image a real gap rather than an
    accepted limitation. Added to the P190 allowlist.
    """

    def test_the_review_script_is_in_the_image_allowlist(self):
        df = (_REPO / "Dockerfile.engine").read_text(encoding="utf-8")
        assert "scripts/trend_regime_review.py" in df

    def test_it_survives_dockerignore(self):
        """P192: naming a file in the COPY is not enough — `scripts/` is
        excluded, so it needs a matching negation placed after it."""
        di = (_REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()
        stripped = [ln.strip() for ln in di]
        assert "!scripts/trend_regime_review.py" in stripped
        assert stripped.index("scripts/") < stripped.index(
            "!scripts/trend_regime_review.py"), (
            "Docker takes the LAST matching pattern — a negation before the "
            "exclusion does nothing"
        )

    def test_it_places_no_orders(self):
        """The allowlist criterion (P141): nothing that can trade goes in the
        live container."""
        src = (_REPO / "scripts" / "trend_regime_review.py").read_text(
            encoding="utf-8", errors="replace")
        for forbidden in ("place_order", "execute_target", "create_order",
                          "manage_to_signal"):
            assert forbidden not in src

    def test_it_is_stdlib_only(self):
        src = (_REPO / "scripts" / "trend_regime_review.py").read_text(
            encoding="utf-8", errors="replace")
        import re
        mods = set()
        for m in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)",
                             src, re.M):
            mods.add(m.group(1).split(".")[0])
        mods -= {"__future__"}
        assert mods <= set(sys.stdlib_module_names), (
            f"non-stdlib imports would need pip installs in the image: "
            f"{sorted(mods - set(sys.stdlib_module_names))}"
        )
