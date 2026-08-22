"""[P368] The last open gap from P325's profitability audit, measured — and
repairing it would trade LESS, not more.

The operator asked what remains FROM A PROFIT PERSPECTIVE, which is the right
question and one this session had stopped asking: P357-P367 were hygiene,
tooling and guards, and the repo's own Trade-Frequency rule names that as the
anti-pattern ("Let me validate one more thing").

--------------------------------------------------------------------------
THE DEFECT (P325 gap 6, never resolved)
--------------------------------------------------------------------------
`check_alpha_gate` multiplies the asserted alpha by

    perf_factor = 0.5 + (_rolling_hit_rate * 0.5)

so the gate self-corrects toward realized performance. `_rolling_hit_rate` is
written by `update_hit_rate()`, whose only three callers live at
`core/execution_service.py:2998/3560/3755` — **past the P152 routed-asset
early return at :648.** On the only venue that trades they can never run, so
the value is frozen at its `0.5` initialiser and **every asserted alpha has
been cut 25% on a number nobody ever measured.**

The comment at the site says "the system ALREADY tracks realized wins via
update_hit_rate()". That is false for the sleeve, and has been since the
2026-06-13 cutover.

--------------------------------------------------------------------------
THE MEASUREMENT, AND IT INVERTS THE OBVIOUS FIX
--------------------------------------------------------------------------
Reconstructed from 552 sleeve-PnL rows over 69 days (flow-adjusted, so a
deposit is never counted as PnL — P293h). An EPISODE is a contiguous span
where an asset's contract count is non-zero:

    all episodes           n=16  hit_rate 0.562  perf_factor 0.781   (+4%)
    SOLO episodes (clean)  n= 8  hit_rate 0.250  perf_factor 0.625  (-17%)

Equity is aggregate, so an episode overlapping another asset's winning
position is credited that asset's gain — which is exactly why ETH and SOL
show 3 wins from 3 while BTC, usually held alone, shows 3 from 10. **The
SOLO subset is the honest sample**, and it says the true factor is BELOW the
frozen 0.75.

So the naive reading — "the haircut is fictional, remove it, trade more" — is
wrong on this evidence. Repairing the loop would TIGHTEN the gate. That makes
it a correctness item with negative expected profit, not a profit lever.

--------------------------------------------------------------------------
WHAT IS THEREFORE **NOT** DONE, AND WHY
--------------------------------------------------------------------------
No repair. An honest fix needs per-asset realized PnL, which the sleeve does
not record; building that is real work whose measured expected effect is to
trade less. n=8 cannot justify a live gate change in either direction, and
choosing the 0.562 reading over the 0.250 one because it is friendlier would
be exactly the selection this file exists to prevent.

What ships is the measurement, the pin, and the honest statement that P325's
audit is now fully closed.
"""

import inspect
import pathlib

import main
from defense.constitution import AlphaThresholdCalculator

REPO = pathlib.Path(main.__file__).parent


def test_the_hit_rate_is_structurally_frozen():
    """The premise: every writer is past the P152 early return, so on a
    routed asset the value can never move off its initialiser."""
    src = (REPO / "core" / "execution_service.py").read_text(
        encoding="utf-8", errors="replace")
    skip = src.index("coinbase_routed_no_kraken_entry")
    writers = [i for i in range(len(src))
               if src.startswith("update_hit_rate(", i)]
    assert writers, "no update_hit_rate callers found — premise broken"
    assert all(i > skip for i in writers), (
        "an update_hit_rate caller now sits BEFORE the routed-asset skip — "
        "the hit rate may be live, and P368's measurement (which assumes it "
        "is frozen at 0.5) must be re-derived, not inherited"
    )


def test_the_frozen_value_and_the_haircut_it_produces():
    """0.5 -> perf_factor 0.75. If either moves, the 25% figure in P368 and
    the measured comparison against it are stale."""
    calc = AlphaThresholdCalculator()
    assert calc._rolling_hit_rate == 0.5
    assert 0.5 + calc._rolling_hit_rate * 0.5 == 0.75


def test_the_haircut_is_applied_on_the_OVERRIDE_path():
    """It is the override path (signal_edge_bps) that the live seat uses, so
    that is where the 25% actually binds."""
    src = inspect.getsource(AlphaThresholdCalculator.check_alpha_gate)
    i = src.index("[ALPHA-FEEDBACK 2026-06-13]")   # the assignment site
    window = src[i:i + 1600]
    assert "_perf_factor = 0.5 + (self._rolling_hit_rate * 0.5)" in window
    assert "estimated_alpha = estimated_alpha * _perf_factor" in window


def test_repairing_it_would_TIGHTEN_on_the_clean_sample():
    """The finding that stops this being a profit lever. Pinned as
    arithmetic so nobody re-derives it in the friendlier direction."""
    clean_hit_rate = 0.250          # n=8 solo episodes, 69 days
    contaminated = 0.562            # n=16, overlapping positions double-count
    frozen = 0.75
    assert 0.5 + clean_hit_rate * 0.5 < frozen, (
        "on the clean sample the repaired factor is BELOW the frozen one, "
        "i.e. the gate tightens and the system trades less"
    )
    assert 0.5 + contaminated * 0.5 > frozen
    # ...and the two disagree in DIRECTION, which is why n=8 cannot decide it.
    assert (0.5 + clean_hit_rate * 0.5 < frozen
            < 0.5 + contaminated * 0.5)


def test_no_repair_shipped():
    """P368 is measurement only. A repair is a live gate change on n=8."""
    src = (REPO / "defense" / "constitution.py").read_text(
        encoding="utf-8", errors="replace")
    assert "sleeve_hit_rate" not in src, (
        "a hit-rate source was wired without the sample to justify it — "
        "P368 measured n=8 with the two readings disagreeing in direction"
    )
