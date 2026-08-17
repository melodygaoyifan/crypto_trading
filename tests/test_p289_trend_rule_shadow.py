"""[P289] Trend-rule challenger forward ledgers — donchian + emaens.

Pins for the P288 dethroning candidates' observation-only harness:
  * labeler PARITY vs training/trend_rule_lab.py (the runtime cannot import
    training/ (P214), so the math is duplicated — this test is the P192
    two-file drift guard; a drift would forward-test a different mechanism
    than the lab measured, the P164/P214 class)
  * causality (violent future perturbation cannot move past labels)
  * warmup honesty (flat-with-reason, never a fabricated direction)
  * P236 confidence = |direction| (the scorer multiplies direction x conf)
  * trend-only expression (no shorts, ever)
  * scorer prefix registration at BOTH default sites (P192)
  * main.py wiring (init + loop-level tick) and behavioral fail-soft
  * september_check countdown rows

Falsification probes run during development (each surgically reverted):
  removing "donchian" from the scorer function default -> both-sites test
  red; forcing confidence=1.0 on flat rows -> P236 test red; changing the
  defense module's DON_WIN to 90 -> parity test red; removing the emaens
  warmup guard -> warmup test red.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from defense.trend_rule_shadow import (  # noqa: E402
    TrendRuleShadow, donchian_labels, ema_ensemble_labels,
    DON_WIN, DON_WARMUP_BARS, EMAENS_WARMUP_BARS, EMA_PAIRS, STRATEGIES)

MAIN_SRC = (REPO / "main.py").read_text(encoding="utf-8-sig", errors="ignore")
SHADOW_IC = (REPO / "analytics" / "shadow_ic" / "compute_shadow_ic.py"
             ).read_text(encoding="utf-8", errors="ignore")
SEPT_SRC = (REPO / "scripts" / "september_check.py"
            ).read_text(encoding="utf-8", errors="ignore")


def _walk(n=800, seed=7, drift=0.0005):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.02, n)))


# ---------------------------------------------------------------------------
# 1. parity vs the lab (the load-bearing guard)
# ---------------------------------------------------------------------------

class TestLabParity:
    @pytest.fixture(scope="class")
    def lab(self):
        pytest.importorskip("sklearn")  # trend_rule_lab's import chain
        pytest.importorskip("pandas")
        import importlib
        return importlib.import_module("training.trend_rule_lab")

    @pytest.mark.parametrize("seed,drift", [(7, 0.0005), (11, -0.0005),
                                            (13, 0.0)])
    def test_labels_identical_on_random_walks(self, lab, seed, drift):
        c = _walk(seed=seed, drift=drift)
        assert np.array_equal(donchian_labels(c), lab.lab_donchian(c)), (
            "donchian labels drifted from the lab's canonical form — the "
            "forward ledger would measure a different mechanism than P288 "
            "certified (P164/P214 class)")
        assert np.array_equal(ema_ensemble_labels(c),
                              lab.lab_ema_ensemble(c)), (
            "emaens labels drifted from the lab's canonical form")

    def test_constants_equal(self, lab):
        assert DON_WIN == lab.DON_WIN
        assert tuple(EMA_PAIRS) == tuple(lab.EMA_PAIRS)


# ---------------------------------------------------------------------------
# 2. causality + expression
# ---------------------------------------------------------------------------

def test_causality_future_perturbation_moves_nothing_past():
    c = _walk()
    t0 = 600
    fut = c.copy()
    fut[t0:] *= np.linspace(3.0, 0.1, len(fut) - t0)
    for fn in (donchian_labels, ema_ensemble_labels):
        a, b = fn(c)[:t0], fn(fut)[:t0]
        assert np.array_equal(a, b), f"{fn.__name__} is not causal"


def test_trend_only_expression_never_short():
    for seed in (7, 11, 13):
        c = _walk(seed=seed, drift=-0.002)   # a bear walk
        assert donchian_labels(c).min() >= 0.0
        assert ema_ensemble_labels(c).min() >= 0.0


# ---------------------------------------------------------------------------
# 3. record shape / warmup / P236
# ---------------------------------------------------------------------------

class TestRecords:
    def _harness(self, tmp_path):
        return TrendRuleShadow(data_dir=str(tmp_path))

    def test_record_shape_and_p236_confidence(self, tmp_path):
        h = self._harness(tmp_path)
        # BOTH directions, deliberately: a bull walk (direction 1.0) AND a
        # bear walk (direction 0.0). The first version used only the bull
        # fixture, and a falsification probe forcing confidence=1.0 stayed
        # GREEN because abs(1.0) == 1.0 — the P238 class (the probe never
        # reached a distinguishing input). The flat row is the one that
        # distinguishes: its confidence MUST be 0.0 (P236/P224 — the scorer
        # multiplies direction x confidence; a saturated confidence on a
        # flat claim is a confident non-signal).
        bull = list(_walk(n=700, drift=0.004))
        bear = list(_walk(n=700, seed=23, drift=-0.004))
        recs_bull = h.record_tick("BTC", bull)
        recs_bear = h.record_tick("ETH", bear)
        saw_flat = False
        for strat in STRATEGIES:
            for recs in (recs_bull, recs_bear):
                rec = recs[strat]
                assert rec is not None
                assert isinstance(rec["ts"], float)
                assert rec["strategy"] == strat
                assert rec["direction"] in (0.0, 1.0)
                assert rec["confidence"] == abs(rec["direction"])
                saw_flat = saw_flat or rec["direction"] == 0.0
        assert saw_flat, (
            "fixture defect: no flat row was produced — the P236 assertion "
            "cannot distinguish a saturated-confidence bug (P238 class)")
        path = tmp_path / "strategy_shadow" / "donchian_BTC.jsonl"
        row = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
        assert row["cert"] == "p288_partial"

    def test_emaens_warmup_is_flat_with_reason(self, tmp_path):
        h = self._harness(tmp_path)
        closes = list(_walk(n=300, drift=0.003))   # < EMAENS_WARMUP_BARS
        recs = h.record_tick("ETH", closes)
        rec = recs["emaens"]
        assert rec["direction"] == 0.0
        assert rec["confidence"] == 0.0
        assert rec["state"].startswith("warmup")
        # donchian has enough bars at 300 — must NOT be forced flat
        assert recs["donchian"]["state"] == "ok"

    def test_donchian_warmup(self, tmp_path):
        h = self._harness(tmp_path)
        closes = list(_walk(n=DON_WARMUP_BARS - 10))
        rec = h.record_tick("SOL", closes)["donchian"]
        assert rec["direction"] == 0.0
        assert rec["state"].startswith("warmup")

    def test_tick_is_fail_soft_when_fetch_raises(self, tmp_path):
        h = self._harness(tmp_path)
        def _boom(asset):
            raise RuntimeError("venue down")
        h.fetch_closes_4h = _boom       # type: ignore[assignment]
        h.tick()                        # must not raise (Iron Law 7)

    def test_tick_uses_injected_closes_and_writes_all_assets(self, tmp_path):
        h = self._harness(tmp_path)
        closes = list(_walk(n=700, drift=0.003))
        h.tick(closes_by_asset={a: closes for a in ("BTC", "ETH", "SOL")})
        for strat in STRATEGIES:
            for a in ("BTC", "ETH", "SOL"):
                assert (tmp_path / "strategy_shadow"
                        / f"{strat}_{a}.jsonl").exists()


# ---------------------------------------------------------------------------
# 4. registration + wiring pins
# ---------------------------------------------------------------------------

def test_scorer_registers_both_prefixes_at_both_sites():
    # the function default AND the argparse default (P192 — one without the
    # other means the ledger accrues forever and is never scored)
    import re
    fn_default = re.search(
        r"prefixes: Tuple\[str, \.\.\.\] = \([^)]*\)", SHADOW_IC)
    assert fn_default, "function default tuple not found"
    for p in ("donchian", "emaens"):
        assert f'"{p}"' in fn_default.group(0), (
            f"{p} missing from the scorer's function default")
    arg_default = re.search(r'default="([a-z_,]+)"', SHADOW_IC)
    assert arg_default, "argparse default not found"
    for p in ("donchian", "emaens"):
        assert p in arg_default.group(1).split(","), (
            f"{p} missing from the scorer's argparse default")


def test_main_wiring_init_and_loop_tick():
    assert "_trend_rule_shadow = TrendRuleShadow(" in MAIN_SRC, (
        "init block missing")
    assert "._trend_rule_shadow.tick()" in MAIN_SRC, (
        "loop-level tick call missing — the ledgers would never accrue")
    # the tick call must be guarded (a harness fault must never touch the
    # order path, Iron Law 7): the None-check + try sit in the same block
    idx = MAIN_SRC.index("._trend_rule_shadow.tick()")
    blk = MAIN_SRC[idx - 400:idx]
    assert '_trend_rule_shadow", None) is not None' in blk
    assert "try:" in blk


def test_september_check_countdown_rows():
    for p in ("donchian", "emaens"):
        assert f'"{p}":' in SEPT_SRC, (
            f"{p} missing from the september_check countdown roster — the "
            f"one command that prints 'days left' would be silent on its "
            f"exam date (the exact P287 gap, reintroduced)")
