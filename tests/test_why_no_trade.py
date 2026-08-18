"""[P155] `scripts/why_no_trade.py` must decode the on-disk tick state correctly.

This script is the answer to "I don't want to wait 4 hours": the per-tick
`data/diagnostics/diag_<asset>_<tick>.json` files already contain the three
inputs to `is_actionable`, so the blocker is recoverable from state written
BEFORE the T3 fix was deployed.

The parsing is the fragile part and is what these tests pin: `_diag_record`
stores `repr(output)[:200]`, which is not JSON, embeds enum reprs like
`<SystemMode.NORMAL: 'NORMAL'>` that break ast.literal_eval, and truncates at
200 chars.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "why_no_trade", Path(__file__).resolve().parents[1] / "scripts" / "why_no_trade.py")
wnt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wnt)


class _Mode:
    """Reproduces how a SystemMode enum lands in repr() output."""

    def __init__(self, v):
        self.v = v

    def __repr__(self):
        return f"<SystemMode.{self.v}: '{self.v}'>"


def _diag(direction=0.42, exposure=0.20, veto=False, actionable=False,
          mode="NORMAL", asset="SOL"):
    payload = {
        "system_mode": _Mode(mode),
        "direction": direction,
        "target_exposure": exposure,
        "veto_active": veto,
        "is_actionable": actionable,
        "intent_id": f"2026-08-04T12:00:00_{asset}",
    }
    return {
        "tick_time": "2026-08-04T12:00:00+00:00",
        "asset": asset,
        # exactly how _diag_record stores it
        "components": {"engine_decide": {
            "called": True, "output": repr(payload)[:200],
            "consumed": True, "note": "CORE DECISION", "applicable": True}},
    }


# ---------------------------------------------------------------------------
# clause decomposition
# ---------------------------------------------------------------------------

def test_zero_exposure_is_identified_with_a_clean_veto_chain():
    """The 312-tick SOL shape: strong signal, no veto, sizing collapsed."""
    d = wnt.diagnose(_diag(direction=0.42, exposure=0.0))
    assert d["ok"]
    assert d["blockers"] == ["ZERO_EXPOSURE (target_exposure=0.0000 <= 0.01)"]
    assert d["veto_active"] is False


def test_veto_is_identified():
    d = wnt.diagnose(_diag(veto=True))
    assert d["blockers"] == ["VETO_ACTIVE"]


def test_weak_direction_is_identified():
    d = wnt.diagnose(_diag(direction=0.04))
    assert any(b.startswith("WEAK_DIRECTION") for b in d["blockers"])


def test_multiple_blockers_all_reported():
    d = wnt.diagnose(_diag(direction=0.01, exposure=0.0, veto=True))
    assert len(d["blockers"]) == 3


def test_actionable_tick_reports_no_blocker():
    d = wnt.diagnose(_diag(actionable=True))
    assert d["actionable"] is True
    assert d["blockers"] == []


def test_opportunity_mode_relaxes_the_direction_floor():
    """dir=0.07 blocks at the 0.10 base floor but clears the 0.05 OPPORTUNITY one."""
    assert any(b.startswith("WEAK_DIRECTION")
               for b in wnt.diagnose(_diag(direction=0.07, mode="NORMAL"))["blockers"])
    assert not any(b.startswith("WEAK_DIRECTION")
                   for b in wnt.diagnose(_diag(direction=0.07, mode="OPPORTUNITY"))["blockers"])


# ---------------------------------------------------------------------------
# parsing robustness
# ---------------------------------------------------------------------------

def test_enum_repr_does_not_break_parsing():
    """ast.literal_eval would raise on `<SystemMode.NORMAL: 'NORMAL'>`."""
    d = wnt.diagnose(_diag(mode="OPPORTUNITY"))
    assert d["mode"] == "OPPORTUNITY"
    assert d["direction"] == pytest.approx(0.42)


def test_negative_direction_uses_magnitude():
    d = wnt.diagnose(_diag(direction=-0.42, exposure=0.20))
    assert not any(b.startswith("WEAK_DIRECTION") for b in d["blockers"])


def test_truncation_at_200_chars_does_not_lose_the_clauses():
    """intent_id is last in the dict, so truncation eats it first — the three
    clause fields must still survive."""
    d = _diag(asset="SOLANA_WITH_A_VERY_LONG_IDENTIFIER" * 4)
    out = d["components"]["engine_decide"]["output"]
    assert len(out) == 200  # genuinely truncated
    parsed = wnt.diagnose(d)
    assert parsed["direction"] is not None
    assert parsed["target_exposure"] is not None
    assert parsed["veto_active"] is False


def test_missing_probe_degrades_gracefully():
    assert wnt.diagnose({"components": {}})["ok"] is False


# ---------------------------------------------------------------------------
# file discovery + secondary state
# ---------------------------------------------------------------------------

def test_picks_newest_file_per_asset(tmp_path):
    diag_dir = tmp_path / "diagnostics"
    diag_dir.mkdir()
    for ts, exp in (("2026-08-04T04-00-00", 0.0), ("2026-08-04T12-00-00", 0.25)):
        (diag_dir / f"diag_SOL_{ts}.json").write_text(json.dumps(_diag(exposure=exp)), encoding="utf-8")

    found = wnt.load_latest_diags(str(tmp_path), 1)
    assert list(found) == ["SOL"]
    assert len(found["SOL"]) == 1
    assert wnt.diagnose(found["SOL"][0][1])["target_exposure"] == pytest.approx(0.25)


def test_sticky_halt_is_surfaced(tmp_path):
    (tmp_path / "coinbase_sleeve_state.json").write_text(json.dumps(
        {"sleeve_start_equity": 3997.75, "halted": True, "halt_reason": "dd"}), encoding="utf-8")
    out = "\n".join(wnt.sleeve_and_routing(str(tmp_path)))
    assert "STICKY HALT" in out


def test_absent_routing_state_is_called_out(tmp_path):
    out = "\n".join(wnt.sleeve_and_routing(str(tmp_path)))
    assert "ABSENT" in out
    assert "PRE_PHASE_2" in out


def test_main_exits_nonzero_when_no_diagnostics(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["why_no_trade.py", "--data-dir", str(tmp_path)])
    assert wnt.main() == 2
    assert "No data/diagnostics" in capsys.readouterr().out


def test_main_reports_the_blocker_end_to_end(tmp_path, monkeypatch, capsys):
    diag_dir = tmp_path / "diagnostics"
    diag_dir.mkdir()
    (diag_dir / "diag_SOL_2026-08-04T12-00-00.json").write_text(
        json.dumps(_diag(direction=0.42, exposure=0.0)), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["why_no_trade.py", "--data-dir", str(tmp_path)])
    assert wnt.main() == 0
    out = capsys.readouterr().out
    assert "ZERO_EXPOSURE" in out
    assert "veto chain is INNOCENT" in out
