"""[P371] Falsification probes — reintroduce each defect, require red.

A green suite proves nothing until every guard has been shown to fail when
the thing it guards is broken (P328). Each probe below puts one half of P371
back: the IC tool promoting a one-flip signal, or the micro agent's deques
forgetting everything at a restart.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.falsify import Probe, run_probes  # noqa: E402

T_IC = "tests/test_p371_ic_degenerate_signal_guard.py"
T_MICRO = "tests/test_p371_micro_deque_persistence.py"
IC = "analytics/ic/agent_ic_review.py"
MICRO = "agents/microstructure_agent.py"

PROBES = [
    # ---- defect 1: the degenerate-signal guard --------------------------
    Probe(
        name="REFUSED-DEGENERATE no longer outranks PROMOTE-CANDIDATE",
        path=IC,
        old='    if "DEGENERATE" in verdict_bits:  # [P371]',
        new='    if False and "DEGENERATE" in verdict_bits:  # [P371]',
        expect_red=[T_IC],
    ),
    Probe(
        name="the degeneracy measurement is computed and then ignored",
        path=IC,
        old='        degenerate = bool(dg["degenerate"])  # [P371]',
        new='        degenerate = False  # [P371]',
        expect_red=[T_IC],
    ),
    Probe(
        name="sign changes counted as ONE interleaved list, not per asset",
        path=IC,
        old='            xs, (assets_by_h or {}).get(h) if assets_by_h else None)  # [P371]',
        new='            xs, None)  # [P371]',
        expect_red=[T_IC],
    ),
    Probe(
        name="the dominant-share branch is removed (perma-bias passes)",
        path=IC,
        old='    if dominant > DEGENERATE_DOMINANT_SHARE:  # [P371]',
        new='    if False:  # [P371]',
        expect_red=[T_IC],
    ),
    Probe(
        name="the n_eff-signal re-pricing of t is removed (live shape passes)",
        path=IC,
        old='        if abs(t) >= 2.0 and abs(t_signal) < DEGENERATE_MIN_T_AT_SIGNAL_N_EFF:  # [P371]',
        new='        if False:  # probe: no re-pricing  # [P371]',
        expect_red=[T_IC],
    ),
    Probe(
        name="the sign-change bar is loosened to 0 (one flip is 'varying')",
        path=IC,
        old='DEGENERATE_MAX_SIGN_CHANGES = 2        # [P371] <= this many -> degenerate',
        # NOTE the replacement changes the file SIZE on purpose: a same-size
        # edit landing in the same integer second as the previous probe's
        # restore is invisible to Python's pyc cache (mtime+size check) and
        # read as VACUOUS — observed on this probe's first run.
        new='DEGENERATE_MAX_SIGN_CHANGES = 0  # probe-loosened  # [P371] <= this many -> degenerate',
        expect_red=[T_IC],
    ),

    # ---- defect 2: micro deque persistence ------------------------------
    Probe(
        name="the micro agent stops restoring its samples at construction",
        path=MICRO,
        old='        self._restore_warmup_samples()  # [P371] before the first tick; fail-soft',
        new='        pass  # probe: no restore',
        expect_red=[T_MICRO],
    ),
    Probe(
        name="the stale check can never fire (max_age = inf)",
        path=MICRO,
        old='                           max_age_sec=self._WARMUP_MAX_AGE_SEC)  # [P371]',
        new='                           max_age_sec=float("inf"))  # [P371]',
        expect_red=[T_MICRO],
    ),
    Probe(
        name="the samples are never persisted after an append",
        path=MICRO,
        old='        self._persist_warmup_samples()  # [P371] after each append; fail-soft',
        new='        pass  # probe: never persisted',
        expect_red=[T_MICRO],
    ),
    Probe(
        name="a ts/px length mismatch is PAIRED instead of dropped",
        path=MICRO,
        old='                if px is None or len(px) != len(ts):  # [P371]',
        new='                if px is None:  # [P371]',
        expect_red=[T_MICRO],
    ),
    Probe(
        name="a restore failure propagates into the constructor",
        path=MICRO,
        old='            logger.warning("[MicroV6] warmup samples: restore failed (%s: %s) — "  # [P371]',
        new='            raise  # probe\n            logger.warning("[MicroV6] warmup samples: restore failed (%s: %s) — "  # [P371]',
        expect_red=[T_MICRO],
    ),
    Probe(
        name="a persist failure propagates into the tick",
        path=MICRO,
        old='            logger.debug("[MicroV6] warmup samples: persist skipped (%s: %s)",  # [P371]',
        new='            raise  # probe\n            logger.debug("[MicroV6] warmup samples: persist skipped (%s: %s)",  # [P371]',
        expect_red=[T_MICRO],
    ),
]

if __name__ == "__main__":
    sys.exit(0 if run_probes(PROBES) else 1)
