import json

from main import HMATSProductionRunner


def _make_runner(tmp_path):
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner._KILL_SWITCH_TEST_FILE = tmp_path / "kill_switch_test.json"
    return runner


def test_load_kill_switch_record_strips_bom(tmp_path):
    runner = _make_runner(tmp_path)
    raw = json.dumps(
        {
            "triggered_at": "2026-03-12T05:00:00Z",
            "source": "\ufeffFORCE_FLAT",
            "reason": "\ufeffmanual acceptance test",
            "mode": "\ufeffpaper",
        }
    )
    runner._KILL_SWITCH_TEST_FILE.write_text(raw, encoding="utf-8-sig")

    record = runner._load_kill_switch_test_record()

    assert record["source"] == "FORCE_FLAT"
    assert record["reason"] == "manual acceptance test"
    assert record["mode"] == "paper"


def test_write_kill_switch_record_strips_bom(tmp_path):
    runner = _make_runner(tmp_path)
    runner.config = type("Cfg", (), {"mode": type("Mode", (), {"value": "paper"})()})()

    runner._write_kill_switch_test_record("\ufeffFORCE_FLAT", "\ufeffmanual acceptance test")

    payload = json.loads(runner._KILL_SWITCH_TEST_FILE.read_text(encoding="utf-8"))

    assert payload["source"] == "FORCE_FLAT"
    assert payload["reason"] == "manual acceptance test"
