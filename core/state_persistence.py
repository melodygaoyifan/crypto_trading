"""[F7-PERSIST 2026-04-15] Unified state persistence helpers for governors.

Each governor has to_dict/from_dict but no file I/O. This module provides
generic save/load that uses to_dict + atomic write + JSON.

Usage:
    from core.state_persistence import save_state, load_state
    save_state("data/cascade_governor.json", governor.to_dict())
    if data := load_state("data/cascade_governor.json"):
        governor.from_dict(data)
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("HMATS.StatePersistence")


def save_state(path: str | Path, data: dict) -> bool:
    """Atomically write JSON state to disk. Returns True on success."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: write to temp, then rename
        fd, tmp = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, path)
            return True
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
    except Exception as e:
        logger.warning(f"[STATE_PERSIST] save_state({path}) failed: {e}")
        return False


def load_state(path: str | Path) -> Optional[dict]:
    """Read JSON state from disk. Returns None if missing or unreadable."""
    try:
        path = Path(path)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[STATE_PERSIST] load_state({path}) failed: {e}")
        return None
