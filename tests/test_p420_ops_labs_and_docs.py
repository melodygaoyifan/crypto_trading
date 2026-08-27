"""[P420] Labs + docs half of the ops fix pass.

  * etf_sma_overlay_check._stats was LAG-2 while claiming lag-1 (ret[:-1] then
    ret[1:]); pinned with a synthetic series whose answer is known.
  * sizing_overlay_lab.per_bar_net was the unguarded mirror of
    mechanism_lab.pnl_after_cost (which refuses validation reads); it now
    carries the same guard with an explicit allow_validation opt-in, and the
    labs that opt in also LEDGER the read.
  * docs/SEPTEMBER_DECISION_TREE.md prescribed reversed actions (P361 class);
    the corrected claims are pinned so they cannot silently rot back.
  * scripts/september_check.py CANDIDATES carries the three unscheduled
    streams and the countdown no longer prescribes the retired P237 edit.
  * training/Makefile drl-fast matches drl (di=4, fresh tag).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from _source_scan import code_only  # noqa: E402

np = pytest.importorskip("numpy")


# ── 1. lag-1 alignment ────────────────────────────────────────────────────────

class TestEtfOverlayIsLagOne:

    def _stats(self):
        sys.path.insert(0, str(REPO / "training" / "scripts"))
        import importlib
        import etf_sma_overlay_check as m
        return importlib.reload(m)._stats

    def test_position_at_t_earns_the_t_to_t_plus_1_move(self):
        """px doubles ONLY between day 2 and day 3; a position held on day 2
        alone must capture it (lag-1). Under the old lag-2 arithmetic the
        day-2 position earned the day-3 -> day-4 move (zero) and the day-1
        position earned the double instead."""
        px = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        pos_day2 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        pos_day1 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        net2, _, _ = self._stats()(px, pos_day2, 0.0, 0)
        net1, _, _ = self._stats()(px, pos_day1, 0.0, 0)
        assert net2 == pytest.approx(100.0), "pos[t] must earn the t->t+1 move"
        assert net1 == pytest.approx(0.0), "lag-2 would credit day 1 (the defect)"

    def test_matches_the_probes_convention(self):
        """etf_flow_probe._hold is the lag-1 reference (P400); same synthetic
        series, same net."""
        sys.path.insert(0, str(REPO / "training" / "scripts"))
        import etf_flow_probe as probe
        px = np.array([1.0, 1.1, 1.3, 1.2, 1.5, 1.4, 1.6])
        sig = np.array([np.nan, 2.0, 2.0, -2.0, -2.0, 2.0, 2.0])
        net_probe, _, _, p = probe._hold(px, sig, 0.0, 1.0)
        pos = np.array([0.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
        net_check, _, _ = self._stats()(px, pos, 0.0, 0)
        assert net_check == pytest.approx(net_probe, abs=0.1)

    def test_source_pin(self):
        src = code_only(REPO / "training" / "scripts" / "etf_sma_overlay_check.py")
        i = src.index("def _stats(")
        blk = src[i:i + 600]
        assert "ret[1:] = px[1:]" in blk
        assert "ret[:-1] = px[1:]" not in blk


# ── 2. per_bar_net guard + the labs' ledgering ───────────────────────────────

class TestPerBarNetGuard:

    def test_refuses_a_validation_read_by_default(self):
        from training.sizing_overlay_lab import per_bar_net
        from training.mechanism_lab import DE
        close = np.linspace(100, 110, DE + 50)
        pos = np.ones_like(close)
        with pytest.raises(AssertionError, match="validation-era"):
            per_bar_net(close, pos, 10.0, DE - 10, DE + 10)

    def test_design_era_read_is_unchanged(self):
        from training.sizing_overlay_lab import per_bar_net
        from training.mechanism_lab import DE
        close = np.linspace(100, 110, DE + 50)
        pos = np.ones_like(close)
        s = per_bar_net(close, pos, 10.0, DE - 20, DE)
        assert len(s) == 19

    def test_explicit_opt_in_allows_it(self):
        from training.sizing_overlay_lab import per_bar_net
        from training.mechanism_lab import DE
        close = np.linspace(100, 110, DE + 50)
        pos = np.ones_like(close)
        s = per_bar_net(close, pos, 10.0, DE - 10, DE + 10, allow_validation=True)
        assert len(s) == 19

    @pytest.mark.parametrize("rel", [
        "training/conviction_sizing_lab.py", "training/gate_probes_lab.py"])
    def test_the_opting_in_labs_also_ledger(self, rel):
        src = code_only(REPO / rel)
        assert "allow_validation=True" in src, f"{rel} lost its explicit opt-in"
        assert "record_window_usage(" in src, (
            f"{rel} opts into a validation read without ledgering it (P420)")

    @pytest.mark.parametrize("rel", [
        "training/scripts/conviction_channel_lab.py",
        "training/scripts/ridge_16h_pooled_check.py"])
    def test_the_other_readers_ledger_too(self, rel):
        assert "record_window_usage(" in code_only(REPO / rel), rel

    def test_gate_probes_ledgers_once_per_asset(self):
        """Six probes read the same validation window; one spend, many looks —
        the ledger must not count it six times."""
        src = code_only(REPO / "training" / "gate_probes_lab.py")
        assert "_LEDGERED" in src
        assert src.count("record_window_usage(") == 1
        for tag in ("A1 ", "A2 ", "A3 ", "A4 ", "B2 ", "B4 "):
            i = src.index(f'_report(f"{tag}') if tag != "B2 " else src.index('_report("B2 ')
            assert "asset=" in src[i:i + 400], f"probe {tag} does not pass asset= to _report"


# ── 3. the decision tree says the true state ─────────────────────────────────

class TestDecisionTreeCorrections:

    @pytest.fixture(autouse=True)
    def _doc(self):
        self.doc = (REPO / "docs" / "SEPTEMBER_DECISION_TREE.md").read_text(encoding="utf-8")

    def _section(self, heading_prefix):
        i = self.doc.index(heading_prefix)
        j = self.doc.find("\n## ", i + 1)
        return self.doc[i:j if j > 0 else len(self.doc)]

    def test_regimebook_fail_row_no_longer_names_the_whale_seat_as_cover(self):
        sec = self._section("## ~Sep 9 — regimebook raw + adjusted")
        row = [ln for ln in sec.splitlines() if ln.startswith("| raw book FAILS")][0]
        assert "whale seat covers" not in row
        assert "whale_seat_mode" in row and "P417" in row
        assert "skew seat" in row and "ETF seat" in row

    def test_breadth_row_does_not_prescribe_routing_again(self):
        sec = self._section("## ~Sep 15 — volskip + etfflow")
        row = [ln for ln in sec.splitlines() if ln.startswith("| breadth books")][0]
        # the prescription half (after the correction note) must not prescribe
        # the widening again; the note itself quotes the OLD wording
        after = row.split("]**", 1)[1]
        assert "extend sleeve assets + routing" not in after
        assert "REJECTED" in after and "HELD BACK" in after and "P412c" in after
        assert "ROUTED" in after

    def test_etfflow_row_reads_confirm_or_disarm(self):
        sec = self._section("## ~Sep 15 — volskip + etfflow")
        row = [ln for ln in sec.splitlines() if ln.startswith("| etfflow")][0]
        # the prescription half must not say "entry tilt" any more (the quote
        # of the OLD wording inside the correction note is allowed)
        after = row.split("]**", 1)[1]
        assert "candidate for a BTC/ETH entry tilt" not in after
        assert "etf_seat_mode" in row and "confirm" in row and "DISARM" in row

    def test_sizing_ladder_states_the_floored_arithmetic(self):
        sec = self._section("## The sizing ladder")
        assert "0.15 × 3 = 0.45" not in sec.replace("replace the 0.15 × 3 = 0.45", "")
        for tok in ("{BTC .20, ETH .15, SOL .095, XRP .01, BNB .005}",
                    "_sized_contracts", "max(1,", "~6.4%", "0.50"):
            assert tok in sec, tok

    def test_three_unscheduled_streams_have_sections(self):
        for h in ("## ~Sep 24 — skewetf", "## ~Sep 26 — convsize",
                  "## regimebook_breadth"):
            assert h in self.doc, h
        assert "conviction_sizing_review.py" in self._section("## ~Sep 26 — convsize")
        assert "_4H_ohlcv_kraken.parquet" in self._section("## regimebook_breadth")

    def test_sizing_ladder_matches_the_live_profile(self):
        """The doc quotes the live fractions; if the profile moves, the doc
        must move with it (P237)."""
        import json
        live = json.loads((REPO / "configs" / "live_high_risk.json").read_text(encoding="utf-8"))
        fr = live["coinbase_target_fraction_by_asset"]
        assert fr == {"BTC": 0.2, "ETH": 0.15, "SOL": 0.095, "XRP": 0.01, "BNB": 0.005}, (
            "live fractions moved — re-derive the sizing-ladder table in "
            "docs/SEPTEMBER_DECISION_TREE.md")


# ── 4. september_check rows + countdown ──────────────────────────────────────

class TestSeptemberCheckRoster:

    def test_three_new_candidates_present(self):
        from scripts.september_check import CANDIDATES
        for name in ("skewetf", "convsize", "regimebook_breadth"):
            assert name in CANDIDATES, name
        assert CANDIDATES["skewetf"][1] == "2026-08-25"
        assert CANDIDATES["convsize"][1] == "2026-08-27"
        assert "TBD" in CANDIDATES["regimebook_breadth"][1]

    def test_countdown_lists_an_unstarted_clock_and_retires_the_p237_line(self, capsys):
        from scripts.september_check import countdown
        from datetime import date
        countdown(today_override=date(2026, 8, 27))
        out = capsys.readouterr().out
        assert "regimebook_breadth" in out and "clock unstarted" in out
        assert "trend_assets removal" not in out
        assert "NO trend_assets edit" in out

    def test_an_unstarted_clock_is_never_due(self):
        from scripts.september_check import countdown
        from datetime import date
        due = countdown(return_due=True, today_override=date(2027, 1, 1))
        assert "regimebook_breadth" not in {d[0] for d in due}


# ── 5. Makefile drl-fast aligned with drl ────────────────────────────────────

class TestMakefileDrlFast:

    def test_drl_fast_has_di4_and_a_fresh_tag(self):
        text = (REPO / "training" / "Makefile").read_text(encoding="utf-8")
        blk = re.search(r"^drl-fast:\n((?:\t.*\n)+)", text, re.M)
        assert blk, "drl-fast target is gone"
        body = blk.group(1)
        assert "--decision-interval 4" in body
        assert "--tag makefile_fast_$$(date" in body
        assert "--tag makefile_fast " not in body and "--tag makefile_fast |" not in body
