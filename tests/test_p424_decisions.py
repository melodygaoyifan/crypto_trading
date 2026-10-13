"""[P424] The six standing decisions, judged by delegation ("from the
profitability perspective, make all judgment on my behalf", 2026-10-13) and
their mechanisms:

  1. home_trio_posture  -> A keep enforce (recorded; no config change) + a
                           dated re-read entry (home_trio_reread, 2026-12-01)
  2. skew_rearm         -> A BTC ONLY on the replica-bank feed
                           (SkewFlowSignal feed="bank"; skew_seat_feed config)
  3. calbasis_reread    -> MECHANIZED (in-container Monday cron + image allowlist)
  4. jump_regime_cutover-> B keep shadow (recorded)
  5. bitbeast_shadow    -> keep shadow, pre-committed EARNS rule (recorded)
  6. capital_venue      -> A (operator probe owed) then C (recorded)

Every pin here is a DECIDED-VALUE pin (P237): a silent revert and a silent
widening both fail.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LIVE = json.loads((REPO / "configs" / "live_high_risk.json").read_text(encoding="utf-8-sig"))


# ---------------------------------------------------------------- bank feed
def _write_bank(tmp_path, asset, n=40, tenor=30, days_old_latest=0, skew10=True):
    p = tmp_path / f"laevitas_apiv2_skew_{asset}.jsonl"
    now = time.time()
    with open(p, "w", encoding="utf-8") as fh:
        for i in range(n):
            d = time.strftime("%Y-%m-%d", time.gmtime(now - (n - 1 - i + days_old_latest) * 86400))
            row = {"date": d, "tenor": tenor, "skew_25d": float(i % 7) - 3.0,
                   "skew_10d": (float(i % 5) - 2.0) if skew10 else None,
                   "call_25d_iv": 30.0, "put_25d_iv": 31.0, "atm_iv": 30.5,
                   "source": "deribit_replica_v1"}
            fh.write(json.dumps(row) + "\n")
    return p


def _bank_sig(tmp_path, monkeypatch=None):
    from defense.skew_flow_signal import SkewFlowSignal
    s = SkewFlowSignal(data_dir=str(tmp_path), feed="bank")
    s._key = ""  # the dead vendor key: bank mode must not need one
    return s


class TestBankFeed:
    def test_bank_mode_reads_the_archive_without_a_key_and_is_fresh(self, tmp_path):
        _write_bank(tmp_path, "BTC")
        s = _bank_sig(tmp_path)
        d, fresh = s.seat_direction("BTC")
        assert fresh is True
        assert d in (-1.0, 0.0, 1.0)
        from defense.skew_flow_signal import _LAST_DIAG
        assert _LAST_DIAG["BTC"]["feed"] == "bank"
        assert _LAST_DIAG["BTC"]["n_rows"] >= 15

    def test_bank_mode_never_rewrites_the_archive(self, tmp_path):
        p = _write_bank(tmp_path, "BTC")
        before = p.read_bytes()
        _bank_sig(tmp_path).seat_direction("BTC")
        assert p.read_bytes() == before

    def test_bank_mode_never_calls_the_vendor(self, tmp_path, monkeypatch):
        _write_bank(tmp_path, "BTC")
        s = _bank_sig(tmp_path)
        s._key = "would-be-a-live-key"
        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("vendor called")))
        d, fresh = s.seat_direction("BTC")
        assert fresh is True

    def test_a_stale_archive_is_not_fresh_so_the_incumbent_stands(self, tmp_path):
        """The replica cron dying must degrade to 'seat skipped', never to a
        held stale direction (the same _STALE_DAYS bound the API feed has)."""
        _write_bank(tmp_path, "BTC", days_old_latest=5)
        d, fresh = _bank_sig(tmp_path).seat_direction("BTC")
        assert fresh is False

    def test_a_missing_archive_is_not_fresh_and_nothing_is_fabricated(self, tmp_path):
        d, fresh = _bank_sig(tmp_path).seat_direction("ETH")
        assert (d, fresh) == (0.0, False)

    def test_only_tenor_30_rows_feed_the_z(self, tmp_path):
        _write_bank(tmp_path, "BTC", n=40, tenor=7)
        d, fresh = _bank_sig(tmp_path).seat_direction("BTC")
        assert fresh is False  # 0 tenor-30 rows -> warmup, not fresh

    def test_default_feed_is_api_and_an_unknown_value_reads_as_api(self, tmp_path):
        from defense.skew_flow_signal import SkewFlowSignal
        assert SkewFlowSignal(data_dir=str(tmp_path))._feed == "api"
        assert SkewFlowSignal(data_dir=str(tmp_path), feed="replica")._feed == "api"
        assert SkewFlowSignal(data_dir=str(tmp_path), feed="BANK")._feed == "bank"

    def test_bank_and_api_modes_agree_on_identical_rows(self, tmp_path, monkeypatch):
        """The bank reader must hand downstream the SAME row shape the API
        fetch does: fed identical rows, the two modes decide identically."""
        from defense.skew_flow_signal import SkewFlowSignal
        p = _write_bank(tmp_path, "BTC")
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        api_dir = tmp_path / "api"
        api_dir.mkdir()
        a = SkewFlowSignal(data_dir=str(api_dir), feed="api")
        a._key = "test"
        monkeypatch.setattr(a, "_fetch_trailing", lambda asset: rows)
        b = _bank_sig(tmp_path)
        assert a.seat_direction("BTC") == b.seat_direction("BTC")


# ---------------------------------------------------------------- config trio
class TestConfigPlumbing:
    def test_skew_seat_feed_declared_parsed_and_passed(self):
        src = (REPO / "main.py").read_text(encoding="utf-8-sig")
        assert 'skew_seat_feed: str = "api"' in src
        assert 'skew_seat_feed=str(data.get("skew_seat_feed", "api")' in src
        assert 'SkewFlowSignal(data_dir="data", feed=_skew_feed)' in src

    def test_live_profile_is_the_decided_state_btc_only_on_the_bank(self):
        assert LIVE["skew_seat_mode"] == "enforce"
        assert LIVE["skew_seat_assets"] == ["BTC"]       # ETH stays PARKED (era-fragile)
        assert LIVE["skew_seat_feed"] == "bank"
        assert "P424" in LIVE["_skew_seat_rearm_note"]

    def test_live_profile_keeps_the_home_trio_posture(self):
        assert LIVE["regimebook_mode"] == "enforce"
        assert LIVE["trend_following_mode"] == "enforce"
        assert LIVE["bitbeast_veto_mode"] == "shadow"


# ---------------------------------------------------------------- ops
class TestOps:
    def test_weekly_seat_read_is_scoped_off_the_skew_asset(self):
        src = (REPO / "scripts/ops/seat_check_weekly.sh").read_text(encoding="utf-8")
        assert 'SCOPE="--assets ETH,SOL"' in src
        assert src.count("$SCOPE") == 2, "both seat_check reads must be scoped"

    def test_calbasis_read_is_cronned_and_in_the_image_both_halves(self):
        cron = (REPO / "scripts/ops/hmats.crontab").read_text(encoding="utf-8")
        row = [l for l in cron.splitlines() if "calbasis_check.py" in l and not l.startswith("#")]
        assert len(row) == 1
        assert "--window-days 110" in row[0] and "--prefixes calbasis" in row[0]
        assert "--allow-kraken-ohlcv" in row[0] and "--pool-assets" in row[0]
        # crontab `%` is a newline unless escaped — an unescaped date would
        # truncate the command at the first %
        assert "+%Y" not in row[0] and "\\%Y\\%m\\%d" in row[0]
        docker = (REPO / "Dockerfile.engine").read_text(encoding="utf-8")
        assert "scripts/calbasis_check.py" in docker
        ign = (REPO / ".dockerignore").read_text(encoding="utf-8")
        assert "!scripts/calbasis_check.py" in ign


# ---------------------------------------------------------------- registry
class TestRegistryRecordsTheJudgments:
    @pytest.fixture(scope="class")
    def reg(self):
        from analytics.decisions import registry as r
        return {d["id"]: d for d in r.load_registry()}

    @pytest.mark.parametrize("item,needle", [
        ("home_trio_posture", "A: KEEP ENFORCE"),
        ("skew_rearm", "A: RE-ARM BTC ONLY"),
        ("jump_regime_cutover", "B: KEEP SHADOW"),
        ("capital_venue", "A then C"),
    ])
    def test_decided_items_carry_date_and_decision(self, reg, item, needle):
        d = reg[item]
        assert d["status"] == "decided"
        assert d["decided_on"] == "2026-10-13"
        assert needle in d["decision"]
        assert "config_change" in d

    @pytest.mark.parametrize("item", ["calbasis_reread", "bitbeast_shadow_review", "home_trio_reread"])
    def test_dated_reads_stay_pending_with_a_pre_committed_rule(self, reg, item):
        d = reg[item]
        assert d["status"] == "pending"
        assert "decision_rule" in d and "2026" in d["decision_rule"]

    def test_the_rearm_is_recorded_beside_its_config_change(self, reg):
        assert "skew_seat_feed bank" in reg["skew_rearm"]["config_change"]

    def test_the_rearm_decision_names_the_flatten_hazard_of_option_b(self, reg):
        assert "Best-of-N" in reg["home_trio_posture"]["decision"]
