"""[P366] Falsification probes — reintroduce each defect, require red.

A green suite proves nothing until every guard has been shown to fail when
the thing it guards is broken (P328). Each probe below is one of the three
defects put back.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.falsify import Probe, run_probes  # noqa: E402

T = "tests/test_p366_readthrough_fixes.py"

PROBES = [
    # ---- fix 1: the status collapse -------------------------------------
    Probe(
        name="EXIT_ONLY collapses every outcome back into 'EXITED'",
        path="main.py",
        old=('            return (sleeve_exit_status(_vs, "EXITED"),\n'
             '                    f"execute_target(0) -> {_vs}")'),
        new='            return "EXITED", f"execute_target(0) -> {_vs}"',
        expect_red=[T],
    ),
    Probe(
        name="REDUCE_50 collapses every outcome back into 'REDUCED'",
        path="main.py",
        old=('            return (sleeve_exit_status(_vs, "REDUCED"),\n'
             '                    f"execute_target({_tgt}) -> {_vs}")'),
        new='            return "REDUCED", f"execute_target({_tgt}) -> {_vs}"',
        expect_red=[T],
    ),
    Probe(
        name="an unrecognised venue status reads as a silent success",
        path="main.py",
        old='    return "EXIT_FAILED"\n\n\nasync def sleeve_fast_risk_action',
        new='    return ok_name\n\n\nasync def sleeve_fast_risk_action',
        expect_red=[T],
    ),
    Probe(
        name="BLOCKED is no longer distinguishable from a venue rejection",
        path="main.py",
        old='    if vs == "BLOCKED":\n        return "EXIT_BLOCKED"',
        new='    if vs == "BLOCKED":\n        return "EXIT_FAILED"',
        expect_red=[T],
    ),
    Probe(
        name="a structural failure clears the P329 unreadable streak again",
        path="main.py",
        old='("SKIPPED_STALE", "ERROR", "EXIT_FAILED",\n                                "DISABLED", "NO_SLEEVE")',
        new='("SKIPPED_STALE", "ERROR",\n                                "DISABLED", "NO_SLEEVE")',
        expect_red=[T],
    ),
    Probe(
        name="the P110/P329 failure detector goes unreachable again",
        path="main.py",
        old='elif (_frs_st in ("ERROR", "SKIPPED_STALE", "EXIT_FAILED")',
        new='elif (_frs_st in ("ERROR", "SKIPPED_STALE")',
        expect_red=[T],
    ),

    # ---- fix 2: wavelet persistence -------------------------------------
    Probe(
        name="the wavelet buffer stops being restored at construction",
        path="data_mgmt/market_data_pipeline.py",
        old=('        self._restore_rolling_buffer("wavelet_buffers",\n'
             '                                     self._wavelet_flat_view())'),
        new="        pass  # probe: no wavelet restore",
        expect_red=[T],
    ),
    Probe(
        name="the wavelet buffer drops out of the persist roster",
        path="data_mgmt/market_data_pipeline.py",
        old='                _targets.append(("wavelet_buffers", self._wavelet_flat_view()))',
        new="                pass  # probe: never persisted",
        expect_red=[T],
    ),
    Probe(
        name="the flat view COPIES, so restores land in throwaway deques",
        path="data_mgmt/market_data_pipeline.py",
        old="        return {f\"{a}::{feat}\": dq",
        new="        return {f\"{a}::{feat}\": __import__('collections').deque(dq, maxlen=dq.maxlen)",
        expect_red=[T],
    ),
    Probe(
        name="a failed save still clears the dirty flag (window loses a bar)",
        path="data_mgmt/market_data_pipeline.py",
        old='                    if name == "wavelet_buffers" and _ok:',
        new='                    if name == "wavelet_buffers":',
        expect_red=[T],
    ),
    Probe(
        name="the cold-start message asserts a warmup length again",
        path="data_mgmt/market_data_pipeline.py",
        old='                logger.info("[BUFFER] %s: no saved state — warming up from "\n                            "scratch", name)',
        new='                logger.info("[BUFFER] %s: no saved state — warming up from "\n                            "scratch (needs ~20h of uptime to emit)", name)',
        expect_red=[T],
    ),

    # ---- fix 3: sustained cancel-refusal escalation ----------------------
    Probe(
        name="a sustained refusal streak stops escalating",
        path="exchange/coinbase_sleeve.py",
        old="            if n == self._CANCEL_REFUSE_SUSTAINED:",
        new="            if False and n == self._CANCEL_REFUSE_SUSTAINED:",
        expect_red=[T],
    ),
    Probe(
        name="the streak never resets, so blips accumulate forever",
        path="exchange/coinbase_sleeve.py",
        old='        # [P366] the book is verified clear — this asset can trade again.\n        self._note_cancel_ok(asset)',
        new="        pass  # probe: no reset",
        expect_red=[T],
    ),
    Probe(
        name="the listing-failure path stops feeding the counter",
        path="exchange/coinbase_sleeve.py",
        old=('            self._note_cancel_refusal(\n'
             '                asset, f"could not list resting orders "\n'
             '                       f"({type(e).__name__}: {e})")'),
        new="            pass  # probe: listing failure uncounted",
        expect_red=[T],
    ),
]

if __name__ == "__main__":
    sys.exit(0 if run_probes(PROBES) else 1)
