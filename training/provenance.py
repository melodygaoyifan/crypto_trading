"""[GP0] Provenance stamping — every artifact records what produced it.

P200's forensics found the deployed models' training parquets had been
overwritten: the runs were unreproducible and nobody could say which data
made which number. The fix is structural: every results artifact carries
(git commit, dirty flag, content hash of each input file, config) so a
number can always be traced to the exact code + data that produced it.
"""
from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO), *args],
                              capture_output=True, text=True, timeout=15
                              , encoding="utf-8").stdout.strip()
    except Exception:
        return ""


def file_hash(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def provenance_stamp(data_files=(), config=None):
    """Returns the provenance dict to embed in a results artifact."""
    dirty = bool(_git("status", "--porcelain"))
    return {
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": dirty,
        "data_hashes": {Path(p).name: file_hash(p) for p in data_files
                        if Path(p).exists()},
        "config": config or {},
        "stamped_at": datetime.now(timezone.utc).isoformat(),
    }
