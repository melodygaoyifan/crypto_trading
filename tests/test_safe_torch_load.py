"""
Tests for infra/safe_torch_load.py path-prefix validation.

P29 follow-up: torch.load(weights_only=False) is RCE-by-pickle if the path
is attacker-controlled. The wrapper rejects any path outside an allowlist
of model directories.
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infra.safe_torch_load import (
    safe_torch_load,
    validate_model_path,
    _resolve_allowed_roots,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


class TestValidateModelPath:
    def test_path_under_repo_models_dir_passes(self, tmp_path, monkeypatch):
        # Use the actual repo models/ dir to avoid env dependence.
        if not MODELS_DIR.exists():
            pytest.skip("repo models/ dir doesn't exist in this checkout")
        target = MODELS_DIR / "_test_marker.pt"
        target.touch()
        try:
            result = validate_model_path(target)
            assert result == target.resolve()
        finally:
            target.unlink(missing_ok=True)

    def test_path_outside_allowlist_rejected(self, tmp_path):
        bad_path = tmp_path / "evil.pt"
        bad_path.touch()
        with pytest.raises(PermissionError, match="not under any allowed"):
            validate_model_path(bad_path)

    def test_traversal_attempt_rejected(self, tmp_path, monkeypatch):
        # A path that resolves outside repo models/ via .. → must reject
        bad = REPO_ROOT / "models" / ".." / ".." / "etc" / "passwd"
        with pytest.raises(PermissionError):
            validate_model_path(bad)

    def test_extra_allowed_roots_kwarg_widens_allowlist(self, tmp_path):
        target = tmp_path / "ok.pt"
        target.touch()
        result = validate_model_path(target, extra_allowed_roots=[str(tmp_path)])
        assert result == target.resolve()

    def test_env_var_widens_allowlist(self, tmp_path, monkeypatch):
        target = tmp_path / "ok.pt"
        target.touch()
        monkeypatch.setenv("HMATS_TORCH_LOAD_ALLOWED_ROOTS", str(tmp_path))
        result = validate_model_path(target)
        assert result == target.resolve()

    def test_no_roots_resolved_refuses(self, monkeypatch, tmp_path):
        # Patch the default roots to all-nonexistent + clear env.
        target = tmp_path / "x.pt"
        target.touch()
        monkeypatch.setattr(
            "infra.safe_torch_load._DEFAULT_ALLOWED_ROOTS",
            ("/nonexistent/abc", "/nope/def"),
        )
        monkeypatch.delenv("HMATS_TORCH_LOAD_ALLOWED_ROOTS", raising=False)
        # Plus we need to defeat the parent.exists() escape — those
        # synthetic roots resolve under the cwd, parent may exist.
        # Use absolute non-existent roots whose parent also doesn't exist.
        monkeypatch.setattr(
            "infra.safe_torch_load._DEFAULT_ALLOWED_ROOTS",
            ("/zzz_no_such_dir_abc/sub",),
        )
        with pytest.raises(PermissionError):
            validate_model_path(target)


class TestSafeTorchLoad:
    def test_calls_torch_load_for_allowed_path(self, tmp_path, monkeypatch):
        target = tmp_path / "ok.pt"
        target.touch()
        monkeypatch.setenv("HMATS_TORCH_LOAD_ALLOWED_ROOTS", str(tmp_path))

        fake_torch = MagicMock()
        fake_torch.load.return_value = {"state": "loaded"}

        result = safe_torch_load(
            target, map_location="cpu", weights_only=False,
            torch_module=fake_torch,
        )
        assert result == {"state": "loaded"}
        fake_torch.load.assert_called_once()
        call_kwargs = fake_torch.load.call_args.kwargs
        assert call_kwargs["map_location"] == "cpu"
        assert call_kwargs["weights_only"] is False

    def test_blocks_torch_load_for_rejected_path(self, tmp_path):
        bad = tmp_path / "evil.pt"
        bad.touch()
        fake_torch = MagicMock()
        with pytest.raises(PermissionError):
            safe_torch_load(bad, torch_module=fake_torch)
        # Critically: torch.load must NEVER be called when path is rejected.
        fake_torch.load.assert_not_called()

    def test_extra_kwargs_forwarded(self, tmp_path, monkeypatch):
        target = tmp_path / "ok.pt"
        target.touch()
        monkeypatch.setenv("HMATS_TORCH_LOAD_ALLOWED_ROOTS", str(tmp_path))

        fake_torch = MagicMock()
        safe_torch_load(
            target, torch_module=fake_torch, pickle_module="custom",
        )
        call_kwargs = fake_torch.load.call_args.kwargs
        assert call_kwargs["pickle_module"] == "custom"
