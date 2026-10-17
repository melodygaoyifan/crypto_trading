"""[P1-7] Shadow report verdict + Phase 1 invariants."""
import importlib.util
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PHASE1_START = "4e2078f"


def _mod():
    spec = importlib.util.spec_from_file_location("p1sr", REPO / "scripts" / "phase1_shadow_report.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _st(rt, gross, comm):
    return {"would_suppress": 3, "round_trips": rt,
            "sum_realized_gross_usd": gross, "sum_commission_usd": comm}


class TestVerdict:
    def test_not_yet_under_48h(self):
        assert _mod()._verdict(_st(5, -10.0, 4.0), 47.9)[0] == "NOT_YET"

    def test_eligible_only_when_suppressed_fills_were_net_losers(self):
        assert _mod()._verdict(_st(5, 1.0, 4.0), 60)[0] == "ENFORCE_ELIGIBLE"
        assert _mod()._verdict(_st(5, 9.0, 4.0), 60)[0] == "KEEP_SHADOW"

    def test_nothing_suppressed_keeps_shadow(self):
        assert _mod()._verdict(_st(0, 0.0, 0.0), 60)[0] == "KEEP_SHADOW"

    def test_attribution_counts_each_round_trip_once(self):
        rows = [{"ts": 100.0, "asset": "BTC", "verdict": "WOULD_SUPPRESS"},
                {"ts": 110.0, "asset": "BTC", "verdict": "WOULD_SUPPRESS"}]
        rts = [{"asset": "BTC", "entry_ts": 200.0, "fee_usd": 2.0, "gross_pnl_usd": -5.0}]
        st = _mod()._attribute(rows, rts)
        assert st["round_trips"] == 1 and st["sum_commission_usd"] == 2.0

    def test_floor_log_parse(self, tmp_path):
        log = tmp_path / "hmats.log"
        log.write_text("2026-10-18 04:00:01 | x | INFO | [P1-4-FLOOR] XRP: budget $106.00 < 1 contract "
                       "($648.35) — WOULD_SUPPRESS (shadow): still sizing 1ct\n", encoding="utf-8")
        rows = _mod()._floor_rows(str(log))
        assert len(rows) == 1 and rows[0]["asset"] == "XRP"


class TestPhase1Invariants:
    def _diff(self, *paths):
        out = subprocess.run(["git", "diff", "--stat", PHASE1_START, "--", *paths],
                             cwd=REPO, capture_output=True, text=True, encoding="utf-8")
        return out.stdout.strip()

    def test_constitution_and_training_untouched(self):
        assert self._diff("defense/constitution.py", "training/") == ""

    def test_shadow_report_ships_in_image(self):
        assert "scripts/phase1_shadow_report.py" in (REPO / "Dockerfile.engine").read_text(encoding="utf-8")
        assert "!scripts/phase1_shadow_report.py" in (REPO / ".dockerignore").read_text(encoding="utf-8")
