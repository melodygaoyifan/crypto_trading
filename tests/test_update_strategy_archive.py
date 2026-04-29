"""Operator-CLI: update_strategy_archive.py — unit tests.

Verifies Iron Law 6 enforcement on archive flips + atomic write.
"""

import json
import sys
from pathlib import Path

import pytest

# Add scripts/ to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from update_strategy_archive import (
    MIN_ACTIVE,
    apply_flip,
    atomic_write,
    count_active,
    list_strategies,
    main as cli_main,
)


def _decisions(active=("A", "B", "C", "D"), archived=("E",)) -> dict:
    return {
        "strategies": {
            **{n: {"archived": False, "bucket": "X", "decision": "KEEP"} for n in active},
            **{n: {"archived": True,  "bucket": "X", "decision": "ARCHIVE"} for n in archived},
        }
    }


# ---------- count_active ---------------------------------------------------

def test_count_active_basic():
    d = _decisions(active=("A", "B", "C"), archived=("D", "E"))
    assert count_active(d) == 3


def test_count_active_empty():
    assert count_active({}) == 0
    assert count_active({"strategies": {}}) == 0


# ---------- apply_flip -----------------------------------------------------

def test_apply_flip_archive_pass():
    d = _decisions(active=("A", "B", "C", "D"), archived=("E",))
    changed, msg = apply_flip(d, "A", archive=True)
    assert changed is True
    assert "archived: False -> True" in msg
    assert d["strategies"]["A"]["archived"] is True


def test_apply_flip_unarchive():
    d = _decisions(active=("A", "B", "C"), archived=("D", "E"))
    changed, msg = apply_flip(d, "D", archive=False)
    assert changed is True
    assert d["strategies"]["D"]["archived"] is False


def test_apply_flip_iron_law_6_violation():
    """Active=3 (min); archiving any one would leave 2 < 3 -> REFUSED."""
    d = _decisions(active=("A", "B", "C"), archived=("D", "E"))
    changed, msg = apply_flip(d, "A", archive=True)
    assert changed is False
    assert "Iron Law 6" in msg
    assert "REFUSED" in msg
    # State unchanged
    assert d["strategies"]["A"]["archived"] is False


def test_apply_flip_unknown_strategy():
    d = _decisions(active=("A", "B", "C", "D"))
    changed, msg = apply_flip(d, "DOES_NOT_EXIST", archive=True)
    assert changed is False
    assert "not in decisions JSON" in msg


def test_apply_flip_no_op_already_archived():
    d = _decisions(active=("A", "B", "C", "D"), archived=("E",))
    changed, msg = apply_flip(d, "E", archive=True)
    assert changed is False
    assert "no-op" in msg


def test_apply_flip_no_op_already_unarchived():
    d = _decisions(active=("A", "B", "C", "D"))
    changed, msg = apply_flip(d, "A", archive=False)
    assert changed is False
    assert "no-op" in msg


def test_apply_flip_min_threshold_boundary():
    """Active=4 (above min); archive one -> active=3 still meets min."""
    d = _decisions(active=("A", "B", "C", "D"))
    changed, msg = apply_flip(d, "A", archive=True)
    assert changed is True


# ---------- atomic_write ---------------------------------------------------

def test_atomic_write_creates_file(tmp_path: Path):
    path = tmp_path / "out.json"
    atomic_write(path, {"hello": "world"})
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {"hello": "world"}


def test_atomic_write_replaces_existing(tmp_path: Path):
    path = tmp_path / "out.json"
    path.write_text(json.dumps({"old": True}), encoding="utf-8")
    atomic_write(path, {"new": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}


def test_atomic_write_no_tmp_leftover(tmp_path: Path):
    path = tmp_path / "out.json"
    atomic_write(path, {"a": 1})
    # tmp file should not survive
    assert not (path.with_suffix(".json.tmp")).exists()


# ---------- list_strategies ------------------------------------------------

def test_list_strategies_renders_state():
    d = _decisions(active=("A", "B", "C"), archived=("D",))
    text = list_strategies(d)
    assert "A" in text and "D" in text
    assert "active count: 3" in text


# ---------- CLI ------------------------------------------------------------

def test_cli_list_returns_zero(tmp_path: Path):
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(_decisions(active=("A", "B", "C", "D"))), encoding="utf-8")
    rc = cli_main(["--decisions", str(path), "--list"])
    assert rc == 0


def test_cli_archive_writes_file(tmp_path: Path):
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(_decisions(active=("A", "B", "C", "D"))), encoding="utf-8")
    rc = cli_main(["--decisions", str(path), "--strategy", "A", "--archive"])
    assert rc == 0
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["strategies"]["A"]["archived"] is True


def test_cli_archive_iron_law_6_blocks_write(tmp_path: Path):
    path = tmp_path / "decisions.json"
    initial = _decisions(active=("A", "B", "C"))
    path.write_text(json.dumps(initial), encoding="utf-8")
    rc = cli_main(["--decisions", str(path), "--strategy", "A", "--archive"])
    assert rc == 2
    # File state unchanged
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["strategies"]["A"]["archived"] is False


def test_cli_dry_run_no_write(tmp_path: Path):
    path = tmp_path / "decisions.json"
    initial = _decisions(active=("A", "B", "C", "D"))
    path.write_text(json.dumps(initial), encoding="utf-8")
    rc = cli_main(["--decisions", str(path), "--strategy", "A", "--archive", "--dry-run"])
    assert rc == 0
    after = json.loads(path.read_text(encoding="utf-8"))
    # Disk unchanged in dry-run
    assert after["strategies"]["A"]["archived"] is False


def test_cli_missing_strategy_arg_returns_error(tmp_path: Path):
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(_decisions()), encoding="utf-8")
    rc = cli_main(["--decisions", str(path)])
    assert rc == 1


def test_cli_no_action_specified(tmp_path: Path):
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(_decisions()), encoding="utf-8")
    rc = cli_main(["--decisions", str(path), "--strategy", "A"])
    assert rc == 1


def test_cli_unknown_strategy_returns_2(tmp_path: Path):
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(_decisions()), encoding="utf-8")
    rc = cli_main(["--decisions", str(path), "--strategy", "GHOST", "--archive"])
    assert rc == 2


def test_cli_missing_decisions_file_returns_1(tmp_path: Path):
    rc = cli_main(["--decisions", str(tmp_path / "missing.json"), "--list"])
    assert rc == 1
