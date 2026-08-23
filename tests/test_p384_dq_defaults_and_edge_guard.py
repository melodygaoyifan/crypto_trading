"""[P384] Two dq defaults fail CLOSED, and `has_validated_edge` has no
constant producer.

1. `integration_v36` read `drl_data_quality` / `two_stage_data_quality` with
   an ABSENT default of 1.0 (fresh). Both keys are written by main.py on both
   branches of their blocks (1.0 / 0.0), so absence means the block did not
   run this tick — P170 made the sibling `quant_data_quality` fail closed and
   left these two. Neutral on the live path today (DRL SHADOW; an absent
   direction is 0.0 -> confidence 0.0), pinned so a future producer gap can
   never read as a fresh signal.
2. `has_validated_edge` (the CORRELATION_COLLAPSE exemption conjunct, P383)
   has NO producer by design: a constant `True` producer would vacate the
   trigger, a constant `False` is what absence already means. Guard: no
   production file writes it as a literal True.
"""
from __future__ import annotations

import re
from pathlib import Path

import integration.integration_v36 as iv
import main as m

REPO = Path(m.__file__).resolve().parent
IV_SRC = Path(iv.__file__).read_text(encoding="utf-8")
MAIN_SRC = Path(m.__file__).read_text(encoding="utf-8")


class TestDqDefaultsFailClosed:
    def test_drl_dq_absent_reads_zero(self):
        assert 'agent_signals.get("drl_data_quality", 0.0)' in IV_SRC
        assert 'agent_signals.get("drl_data_quality", 1.0)' not in IV_SRC

    def test_two_stage_dq_absent_reads_zero(self):
        assert 'agent_signals.get("two_stage_data_quality", 0.0)' in IV_SRC
        assert 'agent_signals.get("two_stage_data_quality", 1.0)' not in IV_SRC

    def test_main_still_produces_both_keys_on_both_branches(self):
        # the fail-closed default is only honest while the producers write
        # the key on EVERY reached path; if a writer disappears, absence would
        # exclude a live agent — still the conservative direction, but it
        # should be a decision, not drift.
        for key in ("drl_data_quality", "two_stage_data_quality"):
            assert f"agent_signals['{key}'] = 1.0" in MAIN_SRC, key
            assert f"agent_signals['{key}'] = 0.0" in MAIN_SRC, key


class TestHasValidatedEdgeHasNoConstantProducer:
    _PROD_DIRS = ("main.py", "core", "defense", "integration", "signals",
                  "risk", "execution", "exchange", "data_mgmt", "agents")
    _PAT = re.compile(r"""has_validated_edge["']?\]?\s*[:=]\s*True\b""")

    def test_no_production_file_writes_true(self):
        hits = []
        for d in self._PROD_DIRS:
            p = REPO / d
            files = [p] if p.is_file() else list(p.rglob("*.py"))
            for f in files:
                txt = f.read_text(encoding="utf-8", errors="replace")
                for mm in self._PAT.finditer(txt):
                    hits.append((f.relative_to(REPO).as_posix(),
                                 txt[:mm.start()].count("\n") + 1))
        assert not hits, (
            "a constant-True producer of has_validated_edge would exempt the "
            f"CORRELATION_COLLAPSE trigger on every tick (P383): {hits}")

    def test_the_detector_catches_a_constant_producer(self):
        # anti-vacuity (P174): the regex must see the shapes it guards against
        assert self._PAT.search('market_data["has_validated_edge"] = True')
        assert self._PAT.search("signal_data['has_validated_edge']=True")
        assert self._PAT.search('{"has_validated_edge": True}')
        assert not self._PAT.search('has_validated_edge = bool(x)')

    def test_the_checker_reads_absence_as_no_exemption(self):
        from defense.constitution import NoTradeTriggerChecker
        c = NoTradeTriggerChecker()
        all_same, has_edge, reason = c._correlation_conjuncts(
            {"cross_asset_directions": {"BTC": 0.6, "ETH": 0.5, "SOL": 0.7}}, {})
        assert all_same is True and has_edge is False and not reason


class TestCorrelationCollapseStaysHoldByMeasurement:
    """[P384] `CORRELATION_COLLAPSE` stays a sleeve HOLD — DECIDED on the
    corr_collapse_lab read (training/reports/corr_collapse_lab_p384.json):
    the live trigger fires on 17 of 12,651 bars (0.13%), all in bull
    regimes, fired bars EARN (+8.84 bps/bar vs +5.90 unfired), and the
    FLATTEN counterfactual loses −6.09pp in the design era, below the
    random-mask control's p90. Moving it to the flatten roster is a
    live-money semantics change that the measurement does not support —
    re-run the lab and write a P-entry before touching this."""

    def test_roster_membership_is_the_decided_value(self):
        assert "CORRELATION_COLLAPSE" in m._SLEEVE_HOLD_NO_TRADE_TRIGGERS
        flat = getattr(m, "_SLEEVE_FLATTEN_INTENDED_VETOES", set())
        assert "CORRELATION_COLLAPSE" not in flat

    def test_the_lab_report_carries_the_verdict(self):
        import json
        rp = REPO / "training" / "reports" / "corr_collapse_lab_p384.json"
        if not rp.exists():
            import pytest
            pytest.skip("operator-local report not present")
        d = json.loads(rp.read_text(encoding="utf-8"))
        txt = json.dumps(d)
        assert "HOLD" in txt and "FLATTEN EARNS" in txt
