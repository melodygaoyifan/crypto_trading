"""[P420] The seat controller described a seat architecture that no longer
exists — its 2026-08-24 cron run prescribed the P299-RETIRED `trend_assets: []`.

  a. SEAT_CONFIG_EDIT[FLAT] never names trend_assets; it vacates the skew and
     ETF seats too;
  b. incumbent precedence is per asset, in seat-run order:
     skew (BTC/ETH) > etf-decide > regimebook > whale > trend;
  c. whale is a candidate only while whale_seat_mode == "enforce";
  d. a live decider the instrument cannot score (skew/etf) is a REFUSAL, not
     a prescription;
  e. the quant series is labelled by the report's primary_strategy census when
     it carries one, else "quant (mixed seats)".
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analytics.seat.seat_controller import (  # noqa: E402
    FLAT, SEAT_CONFIG_EDIT, UNSCOREABLE_SEATS, live_incumbent, live_incumbents)

LIVE_CFG = json.loads((REPO / "configs" / "live_high_risk.json")
                      .read_text(encoding="utf-8-sig"))
REPORT = {
    "generated": "2026-08-24T06:10:02+00:00", "window_days": 30,
    "agents": {
        "quant": {"horizons": {"1": {"n": 713, "ic": -0.0326, "t": -0.87},
                               "4": {"n": 704, "ic": -0.086, "t": -1.14}},
                  "verdict": "HOLD"},
        "whale": {"horizons": {"1": {"n": 226, "ic": 0.05, "t": 1.5},
                               "4": {"n": 225, "ic": 0.04, "t": 1.2}},
                  "verdict": "HOLD"},
    },
}


def _run(*args):
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(REPO / "scripts" / "seat_check.py"),
         *args], capture_output=True, text=True, cwd=str(REPO),
        timeout=120, encoding="utf-8")


# ---------------------------------------------------------------- a
class TestConfigEdits:
    def test_flat_never_names_trend_assets_and_vacates_every_seat(self):
        flat = SEAT_CONFIG_EDIT[FLAT]
        assert "trend_assets" not in flat
        for key in ('skew_seat_mode: "off"', 'etf_seat_mode: "off"',
                    'regimebook_mode: "off"', 'whale_seat_mode: "off"'):
            assert key in flat, key

    def test_no_edit_anywhere_names_the_retired_actuator(self):
        for seat, edit in SEAT_CONFIG_EDIT.items():
            assert "trend_assets" not in edit, seat

    def test_unscoreable_seats_have_edits_that_say_they_are_not_decided_here(self):
        for seat in UNSCOREABLE_SEATS:
            assert "NOT decided by this instrument" in SEAT_CONFIG_EDIT[seat]


# ---------------------------------------------------------------- b
class TestIncumbentPrecedence:
    ALL_ON = {"skew_seat_mode": "enforce", "skew_seat_assets": ["BTC", "ETH"],
              "etf_seat_mode": "enforce", "etf_decide_assets": ["ETH", "SOL"],
              "regimebook_mode": "enforce", "whale_seat_mode": "enforce",
              "trend_following_mode": "enforce"}

    def test_seat_run_order_wins(self):
        assert live_incumbent(self.ALL_ON, "BTC") == "skew_contra"
        assert live_incumbent(self.ALL_ON, "ETH") == "skew_contra"   # skew > etf
        assert live_incumbent(self.ALL_ON, "SOL") == "etf_flow"      # etf > book
        assert live_incumbent(self.ALL_ON, "XRP") == "regimebook"    # book > whale
        cfg = dict(self.ALL_ON, regimebook_mode="off")
        assert live_incumbent(cfg, "XRP") == "whale"
        cfg["whale_seat_mode"] = "off"
        assert live_incumbent(cfg, "XRP") == "trend"
        cfg["trend_following_mode"] = "off"
        assert live_incumbent(cfg, "XRP") == FLAT

    def test_reads_the_live_profile_as_it_stands(self):
        m = live_incumbents(LIVE_CFG, ["BTC", "ETH", "SOL"])
        assert m == {"BTC": "skew_contra", "ETH": "skew_contra", "SOL": "regimebook"}

    def test_asset_membership_is_case_insensitive_and_absent_lists_are_empty(self):
        cfg = {"skew_seat_mode": "enforce", "regimebook_mode": "enforce"}
        assert live_incumbent(cfg, "btc") == "regimebook"     # no skew_seat_assets
        cfg["skew_seat_assets"] = ["btc"]
        assert live_incumbent(cfg, "BTC") == "skew_contra"


# ---------------------------------------------------------------- c, d, e (CLI)
class TestCli:
    def _cfg(self, tmp_path, **over):
        cfg = dict(LIVE_CFG, **over)
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        return p

    def _report(self, tmp_path, extra=None):
        rep = dict(REPORT, **(extra or {}))
        p = tmp_path / "ic.json"
        p.write_text(json.dumps(rep), encoding="utf-8")
        return p

    def test_unscored_decider_refuses_with_the_named_reason(self, tmp_path):
        r = _run("--ic-report", str(self._report(tmp_path)),
                 "--config", str(self._cfg(tmp_path)))
        assert r.returncode == 2, r.stdout + r.stderr
        assert "decider not scoreable by this instrument" in r.stderr
        assert "read skewetf_* via compute_shadow_ic" in r.stderr
        assert re.search(r"BTC\s*:\s*skew_contra", r.stdout)
        assert "CONFIG EDIT IMPLIED" not in r.stdout

    def test_explicit_unscoreable_incumbent_refuses_too(self, tmp_path):
        r = _run("--incumbent", "skew_contra", "--ic-report",
                 str(self._report(tmp_path)))
        assert r.returncode == 2 and "not scoreable" in r.stderr

    def test_whale_is_dropped_unless_its_seat_is_enforced(self, tmp_path):
        cfg_off = self._cfg(tmp_path, whale_seat_mode="off")
        r = _run("--assets", "SOL", "--ic-report", str(self._report(tmp_path)),
                 "--config", str(cfg_off))
        assert r.returncode in (0, 3), r.stdout + r.stderr
        assert "not a seat candidate (P417)" in r.stdout
        assert not re.search(r"^\s+whale\s", r.stdout, re.M), r.stdout
        cfg_on = self._cfg(tmp_path, whale_seat_mode="enforce")
        r2 = _run("--assets", "SOL", "--ic-report", str(self._report(tmp_path)),
                  "--config", str(cfg_on))
        assert r2.returncode in (0, 3), r2.stdout + r2.stderr
        assert re.search(r"^\s+whale\s", r2.stdout, re.M), r2.stdout

    def test_mixed_scoreable_deciders_refuse_rather_than_pick_one(self, tmp_path):
        cfg = self._cfg(tmp_path, skew_seat_mode="off", etf_seat_mode="enforce",
                        etf_decide_assets=[], whale_seat_mode="enforce",
                        regimebook_mode="off")
        # SOL -> whale? no: regimebook off, whale on -> whale for ALL assets;
        # make BTC differ by giving it the ETF decide seat
        cfg2 = json.loads(cfg.read_text(encoding="utf-8"))
        cfg2["etf_decide_assets"] = ["BTC"]
        cfg.write_text(json.dumps(cfg2), encoding="utf-8")
        r = _run("--assets", "BTC,SOL", "--ic-report", str(self._report(tmp_path)),
                 "--config", str(cfg))
        assert r.returncode == 2 and ("differs across" in r.stderr
                                      or "not scoreable" in r.stderr)

    def test_quant_series_label_from_census_or_mixed(self, tmp_path):
        r = _run("--assets", "SOL", "--ic-report", str(self._report(tmp_path)),
                 "--config", str(self._cfg(tmp_path)))
        assert "quant series label: quant (mixed seats)" in r.stdout
        rep = self._report(tmp_path, {"primary_strategy_census":
                                      {"regimebook": 0.6, "skew_contra": 0.4}})
        r2 = _run("--assets", "SOL", "--ic-report", str(rep),
                  "--config", str(self._cfg(tmp_path)))
        assert "quant (census: regimebook 60%, skew_contra 40%)" in r2.stdout

    def test_a_scoped_run_still_reaches_a_measured_verdict(self, tmp_path):
        """The tool must not have become a permanent refusal: scoped to the
        asset whose decider is an attribution series it still decides."""
        r = _run("--assets", "SOL", "--ic-report", str(self._report(tmp_path)),
                 "--config", str(self._cfg(tmp_path)))
        assert r.returncode == 3, r.stdout + r.stderr   # both negative -> flat
        assert "incumbent : regimebook" in r.stdout
        assert "trend_assets" not in r.stdout
