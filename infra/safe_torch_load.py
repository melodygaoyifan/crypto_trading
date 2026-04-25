"""
Path-validated torch.load wrapper.

torch.load with weights_only=False executes arbitrary pickle code, which is
a known RCE vector if the checkpoint path is attacker-controlled. The HMATS
codebase has multiple sites that load custom-class checkpoints (DT configs,
sentiment configs, sequence-alpha configs) and can't safely flip
weights_only=True without first refactoring training scripts to save
state_dict and config separately.

This wrapper is defense-in-depth: it rejects any path that doesn't resolve
under one of an allowlist of model-root directories. The default allowlist
covers the repo's `models/` dir, the in-container `/opt/hmats/models`, and
the Docker volume bind `/var/lib/docker/volumes/hmats-models/_data`.

Override via:
- `HMATS_TORCH_LOAD_ALLOWED_ROOTS` env var (os.pathsep-separated paths)
- `extra_allowed_roots` keyword arg (per-call list)

Added 2026-04-24 (P22 audit follow-up).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


_DEFAULT_ALLOWED_ROOTS: tuple[str, ...] = (
    "models",  # repo-relative
    "/opt/hmats/models",
    "/var/lib/docker/volumes/hmats-models",
)


def _resolve_allowed_roots(extra: Optional[Iterable[str]] = None) -> list[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    roots: list[Path] = []
    for r in _DEFAULT_ALLOWED_ROOTS:
        p = Path(r)
        if not p.is_absolute():
            p = repo_root / p
        try:
            resolved = p.resolve()
            if resolved.exists() or resolved.parent.exists():
                roots.append(resolved)
        except (OSError, RuntimeError):
            pass
    if extra:
        for r in extra:
            try:
                roots.append(Path(r).resolve())
            except (OSError, RuntimeError):
                pass
    env_extra = os.environ.get("HMATS_TORCH_LOAD_ALLOWED_ROOTS", "")
    if env_extra:
        for r in env_extra.split(os.pathsep):
            r = r.strip()
            if not r:
                continue
            try:
                roots.append(Path(r).resolve())
            except (OSError, RuntimeError):
                pass
    return roots


def _is_under(target: Path, root: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def validate_model_path(
    path,
    *,
    extra_allowed_roots: Optional[Iterable[str]] = None,
) -> Path:
    """Resolve `path` and verify it falls under an allowed model root.

    Returns the resolved Path on success. Raises PermissionError if the
    path escapes all allowed roots.
    """
    target = Path(path).resolve()
    roots = _resolve_allowed_roots(extra_allowed_roots)
    if not roots:
        # No roots configured — refuse rather than allow-anything.
        raise PermissionError(
            f"safe_torch_load: no allowed model roots resolved; refusing to "
            f"load {target} (set HMATS_TORCH_LOAD_ALLOWED_ROOTS to override)"
        )
    for r in roots:
        if _is_under(target, r):
            return target
    raise PermissionError(
        f"safe_torch_load refusing to load from {target}: not under any "
        f"allowed model root (allowed: {[str(r) for r in roots]})"
    )


def safe_torch_load(
    path,
    *,
    map_location=None,
    weights_only: bool = False,
    extra_allowed_roots: Optional[Iterable[str]] = None,
    torch_module=None,
    **kwargs,
):
    """Wrapper around `torch.load` that validates `path` is under an
    allowlist of model directories before unpickling.

    Args:
        path: file path to the checkpoint
        map_location: forwarded to torch.load
        weights_only: forwarded to torch.load (False = unsafe pickle path,
            but acceptable IF the path is allowlisted)
        extra_allowed_roots: extra directories to add to the allowlist for
            this call only
        torch_module: pass an already-imported torch module (avoids
            re-importing in hot paths). Defaults to lazy `import torch`.
        **kwargs: forwarded to torch.load
    """
    target = validate_model_path(path, extra_allowed_roots=extra_allowed_roots)
    if torch_module is None:
        import torch as torch_module  # lazy
    return torch_module.load(
        str(target),
        map_location=map_location,
        weights_only=weights_only,
        **kwargs,
    )
