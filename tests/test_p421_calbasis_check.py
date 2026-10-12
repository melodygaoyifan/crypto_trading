"""[P421] The calbasis re-read checker — the pre-committed criterion is a
mechanism, and each clause of it must be able to FIRE and to REFUSE.

Exit-code contract (P185): 0 not-due/keep-accruing, 2 refusal, 3 criterion met.
The date gate, both refusal shapes (missing evidence, too-short window) and
both criteria are each pinned; `decide()` is pure (P206) so the criterion is
testable without a report on disk.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.calbasis_check import decide, horizon_passes  # noqa: E402

AFTER = "2026-11-11"   # past the read date
BEFORE = "2026-10-20"  # before the read date


def _run(*args):
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(REPO / "scripts" / "calbasis_check.py"),
         *args], capture_output=True, text=True, cwd=str(REPO),
        timeout=120, encoding="utf-8")


def _h(ic=0.3, t=2.5, edge=100.0, need=58.0):
    return {"ic": ic, "t_stat": t, "edge_bps": edge, "required_bps": need,
            "n": 500, "n_eff": 60, "fwd_vol_bps": 600.0, "required_ic": 0.1}


def _report(tmp_path, rec, window_days=95):
    rep = {"generated_at": "x", "window_days": window_days,
           "horizons_bars": [4, 12, 24], "n_records": 1,
           "per_strategy": [rec]}
    p = tmp_path / "shadow_ic_test.json"
    p.write_text(json.dumps(rep), encoding="utf-8")
    return p


def _rec(verdict="HOLD", h4=None, h12=None, h24=None):
    ph = {}
    if h4 is not None:
        ph["4"] = h4
    if h12 is not None:
        ph["12"] = h12
    if h24 is not None:
        ph["24"] = h24
    return {"strategy": "calbasis", "asset": "POOLED", "verdict": verdict,
            "per_horizon": ph,
            "n_per_horizon": {k: v.get("n", 0) for k, v in ph.items()}}


class TestDecidePure:
    def test_criterion_a_scorer_promote_wins(self):
        code, action = decide(_rec(verdict="PROMOTE"))
        assert code == 3 and "CRITERION A" in action and "P141" in action

    def test_criterion_b_slow_horizons_clear_while_h4_does_not(self):
        code, action = decide(_rec(
            h4=_h(ic=0.2, t=3.7, edge=53.4, need=58.0),   # the live near-miss shape
            h12=_h(ic=0.27, t=2.8, edge=136.0, need=58.0),
            h24=_h(ic=0.31, t=2.2, edge=214.0, need=58.0)))
        assert code == 3 and "CRITERION B" in action and "SLOW-EXPRESSION" in action

    def test_keep_accruing_when_h24_misses_significance(self):
        code, action = decide(_rec(
            h12=_h(), h24=_h(t=1.4)))
        assert code == 0 and "keep accruing" in action

    def test_keep_accruing_when_h12_edge_sub_bar(self):
        code, _ = decide(_rec(h12=_h(edge=30.0, need=58.0), h24=_h()))
        assert code == 0

    def test_negative_ic_never_fires_even_with_big_t(self):
        # |t| alone must not fire criterion B — a significantly NEGATIVE IC is
        # the regimebook shape, not a promotion shape.
        assert not horizon_passes(_h(ic=-0.3, t=-3.5))
        code, _ = decide(_rec(h12=_h(ic=-0.3, t=-3.5), h24=_h(ic=-0.3, t=-3.5)))
        assert code == 0

    def test_malformed_horizon_fails_closed(self):
        assert not horizon_passes({})
        assert not horizon_passes({"ic": "x", "t_stat": 3, "edge_bps": 99,
                                   "required_bps": 58})


class TestCli:
    def test_date_gate_not_due_exits_0(self, tmp_path):
        r = _run("--today", BEFORE, "--report",
                 str(_report(tmp_path, _rec(verdict="PROMOTE"))))
        assert r.returncode == 0 and "not due" in r.stdout
        # the gate must run BEFORE the report is read: a PROMOTE record must
        # not fire early
        assert "CRITERION" not in r.stdout

    def test_missing_report_refuses_2(self, tmp_path):
        r = _run("--today", AFTER, "--report", str(tmp_path / "nope.json"))
        assert r.returncode == 2 and "REFUSING" in r.stderr
        assert "september_check" in r.stderr

    def test_short_window_refuses_2(self, tmp_path):
        p = _report(tmp_path, _rec(h12=_h(), h24=_h()), window_days=30)
        r = _run("--today", AFTER, "--report", str(p))
        assert r.returncode == 2 and "does not span" in r.stderr

    def test_no_scoreable_record_refuses_2(self, tmp_path):
        p = _report(tmp_path, {"strategy": "calbasis", "asset": "POOLED",
                               "verdict": "INSUFFICIENT_SAMPLES",
                               "per_horizon": {}, "n_per_horizon": {}})
        r = _run("--today", AFTER, "--report", str(p))
        assert r.returncode == 2 and "wiring question" in r.stderr

    def test_criterion_b_end_to_end_exits_3(self, tmp_path):
        p = _report(tmp_path, _rec(h4=_h(edge=40.0), h12=_h(), h24=_h()))
        r = _run("--today", AFTER, "--report", str(p))
        assert r.returncode == 3, r.stdout + r.stderr
        assert "CRITERION B" in r.stdout

    def test_keep_accruing_end_to_end_exits_0(self, tmp_path):
        p = _report(tmp_path, _rec(h12=_h(t=1.0), h24=_h(t=1.0)))
        r = _run("--today", AFTER, "--report", str(p))
        assert r.returncode == 0 and "keep accruing" in r.stdout
