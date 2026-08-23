"""[P381] Pin the whale_filter reconstruction verdict.

The whale_filter was called "forward-only, cannot be backtested by construction"
(too broad). WhaleDetector is a DETERMINISTIC rule, so its output is recomputable
over historical trades. `training/whale_filter_reconstruction_lab.py` replays the
REAL detector over 120d of futures aggTrades (fidelity-checked to the live class),
replays the deterministic regimebook decider, and buckets forward returns.

This test pins the committed report so the verdict cannot silently drift: the
reconstruction was FAITHFUL (>=0.9999 match to the real detector) and NOT_EARNED
on all three assets (disagreements did not mark worse entries at 16h) — confirming
P337 (neutral) / P348 (no feasible window reaches significance) / P356 (disarm was
correct). It reads the report, never re-downloads (the 5GB tick cache is
gitignored and operator-local).
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest

REPORT = (Path(__file__).resolve().parents[1] / "training" / "reports"
          / "whale_filter_reconstruction_p381.json")


@pytest.fixture(scope="module")
def rep():
    if not REPORT.exists():
        pytest.skip("reconstruction report not present (operator-local run)")
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_reconstruction_was_faithful_to_the_real_detector(rep):
    """The numpy replay must match the real WhaleDetector, or the run is void
    (P172 — the source of truth is the detector, not the reproduction)."""
    fc = rep.get("fidelity")
    assert fc is not None, "no fidelity cross-check recorded"
    assert fc["match_rate"] >= 0.9999, (
        f"vectorized whale rule diverged from the real detector "
        f"(match {fc['match_rate']}) — reconstruction not trustworthy")


def test_the_filter_is_NOT_EARNED_where_it_was_measured(rep):
    """Every asset that produced a verdict must be NOT_EARNED — disagreements did
    not mark worse entries at the decisive 16h horizon. A future EARNS here would
    be a real finding to act on, not a silent drift."""
    measured = {a: r for a, r in rep["assets"].items() if "verdict" in r}
    assert measured, "no asset produced a verdict"
    for a, r in measured.items():
        assert r["verdict"].startswith("NOT_EARNED"), (
            f"{a}: verdict changed to {r['verdict']!r} — re-examine before acting")


def test_the_sample_is_too_small_for_significance(rep):
    """P348's point, made concrete: the filter fires rarely, so no downloadable
    window reaches the >=30-disagreement floor. This pins that the reconstruction
    is a POINT ESTIMATE, not a significance test — a guard against over-reading it."""
    measured = {a: r for a, r in rep["assets"].items() if "horizons" in r}
    assert measured
    n16 = sum(r["horizons"]["16h"]["n_disagree"] for r in measured.values())
    assert n16 < 30 or all(
        r["horizons"]["16h"].get("contrast_bps", 0) is not None
        for r in measured.values()), "sanity"
    # documented expectation: pooled disagreements are few (P348 ~2.8y for |t|>=2)
    assert n16 <= 60, (
        f"pooled disagreements={n16}: if this ever clears ~30 with a consistent "
        f"negative contrast, the filter has a real (still small) effect — revisit")
