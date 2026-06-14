"""P148: DRL frame-buffer persistence — no 32h zero-pad warmup after a restart."""

import time
import numpy as np
import pytest

from drl.ensemble import TQCInference, N_STACK, SINGLE_OBS_DIM


def _frames(k):
    return [np.full(SINGLE_OBS_DIM, i + 1, dtype=np.float32) for i in range(k)]


def test_save_then_restore_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    a = TQCInference(asset="TESTROUNDTRIP")
    for f in _frames(N_STACK):
        a._obs_buffer.append(f)
    a._save_buffer()
    # a fresh instance restores in __init__
    b = TQCInference(asset="TESTROUNDTRIP")
    assert len(b._obs_buffer) == N_STACK
    assert np.allclose(list(b._obs_buffer)[-1], list(a._obs_buffer)[-1])
    assert np.allclose(list(b._obs_buffer)[0], list(a._obs_buffer)[0])


def test_partial_buffer_restored(tmp_path, monkeypatch):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    a = TQCInference(asset="TESTPARTIAL")
    for f in _frames(3):  # not full
        a._obs_buffer.append(f)
    a._save_buffer()
    b = TQCInference(asset="TESTPARTIAL")
    assert len(b._obs_buffer) == 3  # partial buffer restored as-is


def test_stale_buffer_discarded(tmp_path, monkeypatch):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    a = TQCInference(asset="TESTSTALE")
    for f in _frames(N_STACK):
        a._obs_buffer.append(f)
    a._save_buffer()
    # rewrite the file with an old timestamp (beyond MAX_AGE)
    p = a._buffer_path()
    z = np.load(p)
    np.savez(p, frames=z["frames"], saved_ts=np.float64(time.time() - 2 * a._BUFFER_MAX_AGE_SEC))
    b = TQCInference(asset="TESTSTALE")
    assert len(b._obs_buffer) == 0  # stale -> warmup fresh


def test_no_file_starts_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    a = TQCInference(asset="TESTNOFILE")
    assert len(a._obs_buffer) == 0  # no file -> empty (current behavior, no crash)


def test_save_is_best_effort_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    a = TQCInference(asset="TESTBESTEFFORT")
    # empty buffer -> save is a no-op, must not raise
    a._save_buffer()
    assert len(a._obs_buffer) == 0
