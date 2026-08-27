"""[P420] Deploy + roster half of the ops fix pass.

  * scripts/hetzner_deploy.sh CI-checked origin/main at ls-remote time but
    built whatever the server's pull produced; a push landing in between
    deployed an UNVERIFIED commit (2026-08-27 10:32: the P417 PARENT ran for
    15 minutes). After the pull the script now asserts server HEAD ==
    DEPLOY_SHA and a clean server tree, else refuses. Pinned with
    tests/_guard_pins.assert_live_line the way P328 pins the cleanup trap.
  * scripts/maker_fill_review.py and scripts/coinbase_probe_stop_support.py
    looped a trio literal, so XRP/BNB fills could never feed the P315
    fee-revision rule; rosters are DERIVED from core.cde_fees / the symbol map.
  * scripts/conviction_sizing_review.py carried a hand-copied cost dict; it is
    DERIVED (2 x taker fee + measured spread + P374 impact) and pinned to the
    lab it forward-tests.
  * docs/ops/SERVER_CRONS.md records the crons and the no-docker-cp rule.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from _guard_pins import assert_live_line  # noqa: E402
from _source_scan import code_only  # noqa: E402


# ── 1. deploy: server HEAD == verified sha, tree clean ───────────────────────

class TestDeployRefusesAnUnverifiedServerHead:

    def _src(self):
        return io.open(REPO / "scripts" / "hetzner_deploy.sh", encoding="utf-8").read()

    def test_the_post_pull_assertions_are_live(self):
        src = self._src()
        assert_live_line(src, 'REMOTE_HEAD="$(ssh',
                         why="the server HEAD must be read after the pull")
        assert_live_line(src, 'if [ "${REMOTE_HEAD}" != "${DEPLOY_SHA}" ]; then',
                         why="a server HEAD that is not the CI-verified sha must refuse")
        assert_live_line(src, 'REMOTE_DIRTY="$(ssh',
                         why="the server tree must be checked for uncommitted edits")
        assert_live_line(src, 'if [ -n "${REMOTE_DIRTY}" ]; then',
                         why="a dirty server tree must refuse")

    def test_both_refusals_exit_nonzero_before_the_build(self):
        src = self._src()
        head = src.index('if [ "${REMOTE_HEAD}" != "${DEPLOY_SHA}" ]; then')
        dirty = src.index('if [ -n "${REMOTE_DIRTY}" ]; then')
        build = src.index("[3/5] Building Docker images")
        assert head < build and dirty < build
        for i in (head, dirty):
            blk = src[i:src.index("\nfi\n", i)]   # the closing `fi`, not "veriFIed"
            assert "exit 1" in blk, "the refusal must exit, not merely warn"

    def test_the_check_runs_after_the_pull(self):
        src = self._src()
        pull = src.index("git pull origin main")
        assert pull < src.index('REMOTE_HEAD="$(ssh')

    def test_a_commented_check_would_fail_the_pin(self):
        """Anti-vacuity for the pin itself (P330): comment the line out and
        assert_live_line must go red."""
        src = self._src().replace('REMOTE_HEAD="$(ssh', '# REMOTE_HEAD="$(ssh', 1)
        with pytest.raises(AssertionError, match="COMMENTED"):
            assert_live_line(src, 'REMOTE_HEAD="$(ssh')


# ── 2. rosters derived, not restated ─────────────────────────────────────────

class TestMakerFillReviewRoster:

    def test_roster_includes_every_priced_asset(self):
        import scripts.maker_fill_review as m
        from core.cde_fees import CDE_FEE_BPS, CDE_FEE_ASSUMED, CDE_FEE_PREVIEW
        assert set(CDE_FEE_BPS) <= set(m.CONTRACT_SIZE)
        assert m.CONTRACT_SIZE["XRP"] == 500.0 and m.CONTRACT_SIZE["BNB"] == 1.0
        assert m.FEE_NOT_FILL_MEASURED == set(CDE_FEE_ASSUMED) | set(CDE_FEE_PREVIEW)
        assert {"XRP", "BNB", "SOL"} <= m.FEE_NOT_FILL_MEASURED

    def test_edges_come_from_core_seat_alpha(self):
        import scripts.maker_fill_review as m
        from core.seat_alpha import REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP as RT
        assert m.EDGE_RT_BPS == dict(RT)

    def test_no_trio_loop_survives_in_the_ledger_report(self):
        src = code_only(REPO / "scripts" / "maker_fill_review.py")
        i = src.index("def ledger_report(")
        blk = src[i:]
        assert 'for a in ("BTC", "ETH", "SOL")' not in blk

    def test_an_xrp_fill_feeds_the_reprice_progress(self, capsys):
        import scripts.maker_fill_review as m
        rows = [{"asset": "XRP", "liquidity": "maker", "urgent": False,
                 "realized_slippage_bps": -1.0, "fees_usd": 0.64,
                 "fill_avg_price": 1.41, "contracts": 1}]
        assert m.ledger_report(rows) == 0
        out = capsys.readouterr().out
        assert "XRP: " in out and "XRP fee is NOT fill-measured" in out
        assert "1/20 fills toward re-pricing" in out

    def test_report_assets_includes_ledger_assets_the_roster_lacks(self):
        import scripts.maker_fill_review as m
        assets = m.report_assets([{"asset": "ZZZ"}])
        assert "ZZZ" in assets and assets.index("BTC") < assets.index("ZZZ")


class TestStopProbeRoster:

    def test_roster_covers_every_priced_asset_with_its_venue_pid(self):
        import scripts.coinbase_probe_stop_support as p
        from core.cde_fees import CDE_FEE_BPS
        from exchange.symbol_mapping import SYMBOL_MAP
        pids = SYMBOL_MAP["coinbase"]["perp"]
        assert set(p.ASSETS) == set(CDE_FEE_BPS)
        for a in p.ASSETS:
            assert p.EXPECTED_PRODUCTS[a] == pids[a]
        assert p.EXPECTED_PRODUCTS["XRP"] == "XPP-20DEC30-CDE"

    def test_no_trio_literal_is_the_roster(self):
        src = code_only(REPO / "scripts" / "coinbase_probe_stop_support.py")
        assert 'ASSETS = ("BTC", "ETH", "SOL")' not in src


class TestConvictionSizingReviewCost:

    def test_cost_is_derived_and_matches_the_lab_within_half_a_bp(self):
        import scripts.conviction_sizing_review as r
        from training.conviction_sizing_lab import COST_BPS
        from core.cde_fees import CDE_FEE_BPS
        from defense.constitution import CDE_SPREAD_BPS_MEASURED
        for a in ("BTC", "ETH"):
            expect = 2.0 * CDE_FEE_BPS[a]["taker"] + CDE_SPREAD_BPS_MEASURED[a] + r.IMPACT_BPS[a]
            assert r.COST[a] == pytest.approx(expect, abs=0.05)
            assert abs(r.COST[a] - COST_BPS[a]) <= 0.5, (
                f"{a}: the forward reader ({r.COST[a]}) and the lab it forward-"
                f"tests ({COST_BPS[a]}) disagree on cost by >0.5bps")

    def test_no_hand_copied_cost_dict_remains(self):
        src = code_only(REPO / "scripts" / "conviction_sizing_review.py")
        assert 'COST = {"BTC": 27.7, "ETH": 44.0}' not in src
        assert "from core.cde_fees import CDE_FEE_BPS" in src


# ── 3. the crons doc ─────────────────────────────────────────────────────────

class TestServerCronsDoc:

    def test_exists_and_states_the_rule(self):
        doc = (REPO / "docs" / "ops" / "SERVER_CRONS.md").read_text(encoding="utf-8")
        assert "docker cp" in doc and "never" in doc.lower()
        for job in ("agent_ic_review.py", "slope_calibrator.py", "tripwire_check.py",
                    "sleeve_beta_review.py", "trend_regime_review.py",
                    "seat_check_weekly.sh", "calibration_check.py",
                    "september_check.py --countdown-only",
                    "fetch_coinglass_history.py",
                    "accumulate_newdata_snapshots.py", "newdata_gated_probe.py",
                    "etfflow_timing_check.py"):
            assert job in doc, f"crons doc lost {job}"

    def test_every_named_script_exists(self):
        doc = (REPO / "docs" / "ops" / "SERVER_CRONS.md").read_text(encoding="utf-8")
        import re
        for rel in set(re.findall(r"`((?:analytics|scripts|training)/[\w/]+\.py)", doc)):
            assert (REPO / rel).exists(), f"crons doc names {rel}, which is not in the tree"
