"""[P421d] coinbase_set_assets.sh must never print "done." over a failed write.

2026-10-12 the script was run ON the trading server. Its state write (`scp
... hmats:`) and its restart (`ssh hmats ...`) both go through the host alias
`hmats`, which does not resolve from the host itself, so both failed silently,
nothing was written, the manual restart re-read the OLD five-asset routing
state, and the operator-instructed XRP/BNB de-route never landed -- while the
run printed its "done." line. These tests drive the REAL script with a fake
`scp`/`ssh` on PATH (never the dry-run path, which stops before the write) and
pin that a failed write exits 4 with routing unchanged and a failed restart
exits 5. Nothing here touches the network: the fakes are the only scp/ssh the
script can find.
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SET_ASSETS = REPO / "scripts" / "coinbase_set_assets.sh"


def _fake(dirpath: Path, name: str, body: str) -> None:
    f = dirpath / name
    f.write_text("#!/bin/bash\n" + body + "\n", encoding="utf-8", newline="\n")
    f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run(tmp_path: Path, scp_rc: int, ssh_body: str) -> subprocess.CompletedProcess:
    fakes = tmp_path / "fakebin"
    fakes.mkdir()
    _fake(fakes, "scp", f"exit {scp_rc}")
    _fake(fakes, "ssh", ssh_body)
    env = dict(os.environ)
    env.pop("HMATS_SET_ASSETS_DRY_RUN", None)  # the write path, not the dry-run stop
    env["PATH"] = str(fakes) + os.pathsep + env.get("PATH", "")
    return subprocess.run(
        ["bash", str(SET_ASSETS), "BTC,ETH,SOL"],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=60,
    )


def test_failed_state_write_is_a_loud_refusal_not_done(tmp_path):
    r = _run(tmp_path, scp_rc=1, ssh_body="echo SHOULD_NOT_RUN; exit 0")
    assert r.returncode == 4, (r.stdout, r.stderr)
    assert "Routing is UNCHANGED" in r.stderr
    assert "done." not in r.stdout
    # the restart must not even be attempted on top of a failed write
    assert "SHOULD_NOT_RUN" not in r.stdout + r.stderr


def test_failed_restart_after_a_successful_write_exits_5(tmp_path):
    r = _run(tmp_path, scp_rc=0, ssh_body="exit 1")
    assert r.returncode == 5, (r.stdout, r.stderr)
    assert "WAS written" in r.stderr
    assert "done." not in r.stdout


def test_restart_that_cannot_verify_routing_exits_5(tmp_path):
    # The `| grep ROUTED` runs on the REMOTE side (it is inside the ssh command
    # string), so ssh's exit status IS the remote verify grep's: the engine came
    # up (STATUS printed) but the in-container ROUTED line never appeared ->
    # remote grep 1 -> ssh 1. An unverified restart must not read as done.
    r = _run(tmp_path, scp_rc=0, ssh_body="echo 'STATUS: Up 16 seconds'; exit 1")
    assert r.returncode == 5, (r.stdout, r.stderr)
    assert "done." not in r.stdout


def test_happy_path_still_prints_done(tmp_path):
    r = _run(tmp_path, scp_rc=0,
             ssh_body="echo 'STATUS: Up 16 seconds (healthy)'; echo \"ROUTED: {'BTC': True}\"; exit 0")
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "done." in r.stdout


def test_source_pins_the_two_exit_codes():
    src = SET_ASSETS.read_text(encoding="utf-8")
    assert "if ! scp -o BatchMode=yes" in src
    assert "exit 4" in src and "exit 5" in src
