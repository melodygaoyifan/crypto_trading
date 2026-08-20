"""
[P334] The T1 volume-collapse guard was INVERTED at tick time, because it was
the one call site in the tree that discarded the partial-bar correction.

FOUND FROM DATA, not from reading. Live 2026-08-20 04:11:44:

    [REGIMEBOOK-SEAT] ETH: dir=+1.0 leg=trend_hold edge=88bps - book holds the seat
    ALPHA_GATE: PASS (66bps > 55bps, gate=ALLOW)
    [TRANCHE] ETH: refusing T1 entry in volume collapse
                   (effective=0.021, raw=0.021, bar_progress=0.04)

`effective == raw` is the tell: the producer computes
`volume_ratio_effective = max(raw, raw / max(bar_progress, 0.20))` = 0.105 for
that tick, comfortably above the 0.05 threshold, and the guard used 0.021.

THE MECHANISM. `effective_volume_ratio(md, is_4h_bar_close=True)` returns the
RAW ratio by its first branch. `is_4h_bar_close` has NO PRODUCER (P173), so
main.py's `market_data.get("is_4h_bar_close", True)` is always True. Ten other
call sites pass nothing (default False) and get the correction; only this one
passed the flag.

WHY IT INVERTS. `volume_ratio` is current-bar-volume / SMA(FULL-bar volumes),
and the 4H tick fires ~10 min into a new bar, so bar_progress ~= 0.04 on every
tick. With true volume pace k, raw ~= k * 0.04:

    k=1.0  normal           raw 0.040  -> REFUSED   (false positive)
    k=0.5                   raw 0.020  -> REFUSED
    k=0.2  severe collapse  raw 0.008  -> ALLOWED   (< the 0.01 artifact clause)

So normal volume was blocked and the worse the collapse the more likely it
passed. The correction fixes both directions and is self-disabling on a
complete bar, so the flag was redundant with data market_data already carries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.market_data_helpers import effective_volume_ratio  # noqa: E402


def _md(*, k: float, bar_progress: float = 0.04) -> dict:
    """market_data as the pipeline builds it, for volume pace `k`.

    Mirrors data_mgmt/market_data_pipeline.py: raw = k * bar_progress, and
    volume_ratio_effective = max(raw, min(raw / max(progress, 0.20), 5.0)).
    """
    raw = k * bar_progress
    if 0.0 < bar_progress < 1.0:
        pace = min(raw / max(bar_progress, 0.20), 5.0)
        eff = max(raw, pace)
    else:
        eff = raw
    return {
        "volume_ratio": raw,
        "volume_ratio_effective": eff,
        "bar_progress_4h": bar_progress,
        "current_price": 2255.0,
        "data_valid": True,
    }


THRESHOLD = 0.05


def _refuses(ratio: float) -> bool:
    """The guard's own predicate."""
    return ratio < THRESHOLD and ratio >= 0.01


class TestTheInversion:
    """Characterisation of the bug, so the fix is judged against arithmetic
    rather than against a hunch about what 'volume collapse' should mean."""

    @pytest.mark.parametrize("k,label", [(1.0, "normal"), (0.5, "half")])
    def test_raw_ratio_refuses_normal_volume(self, k, label):
        raw = _md(k=k)["volume_ratio"]
        assert _refuses(raw), (
            f"{label} volume gives raw={raw:.3f}, which the guard refuses — "
            f"this is the false positive that blocked ETH")

    def test_raw_ratio_ALLOWS_a_severe_collapse(self):
        """The inversion. A severe collapse falls BELOW the 0.01 artifact
        clause and sails through the guard meant to catch it."""
        raw = _md(k=0.2)["volume_ratio"]
        assert not _refuses(raw), f"raw={raw:.3f} passes the guard"

    def test_the_correction_restores_the_intended_direction(self):
        normal = _md(k=1.0)["volume_ratio_effective"]
        severe = _md(k=0.2)["volume_ratio_effective"]
        assert not _refuses(normal), f"normal must pass (got {normal:.3f})"
        assert _refuses(severe), f"severe collapse must be caught (got {severe:.3f})"


class TestTheHelperIsWhatDiscardedIt:

    def test_true_returns_raw_and_discards_the_correction(self):
        md = _md(k=1.0)
        assert effective_volume_ratio(md, is_4h_bar_close=True) == md["volume_ratio"]

    def test_default_uses_the_correction(self):
        md = _md(k=1.0)
        assert effective_volume_ratio(md) == md["volume_ratio_effective"]

    def test_the_correction_is_self_disabling_on_a_complete_bar(self):
        """Why the flag was redundant: when the bar really is complete the
        producer already sets effective == raw, so passing the flag buys
        nothing and only ever destroys information."""
        md = _md(k=1.0, bar_progress=1.0)
        assert md["volume_ratio_effective"] == md["volume_ratio"]
        assert effective_volume_ratio(md) == effective_volume_ratio(
            md, is_4h_bar_close=True)


class TestTheCallSiteIsFixed:

    def _t1_block(self) -> str:
        from tests._source_scan import code_only
        src = code_only(REPO / "defense" / "constitution.py",
                        strip_docstrings=True)
        i = src.index("refusing T1 entry in volume collapse")
        return src[max(0, i - 1500):i]

    def test_the_t1_guard_no_longer_passes_the_flag(self):
        blk = self._t1_block()
        assert "is_4h_bar_close=is_4h_bar_close" not in blk, (
            "passing the flag reinstates the inversion: it has no producer, "
            "so it is always True and the helper returns the raw ratio")

    def test_the_t1_guard_still_calls_the_helper(self):
        """It must not have been replaced by a raw market_data read — that
        would discard the correction by a different route."""
        blk = self._t1_block()
        assert "_effective_volume_ratio(market_data)" in blk

    def test_the_flag_still_has_no_producer(self):
        """The premise. If someone ever writes this key with a real value the
        whole analysis has to be redone — so fail loudly rather than let the
        reasoning rot (P173)."""
        import re
        hits = []
        for p in REPO.rglob("*.py"):
            s = str(p)
            if any(x in s for x in ("venv", "archive", "tests", "site-packages")):
                continue
            txt = p.read_text(encoding="utf-8-sig", errors="replace")
            if re.search(r"""\[["']is_4h_bar_close["']\]\s*=""", txt):
                hits.append(str(p.relative_to(REPO)))
        assert not hits, (
            f"is_4h_bar_close now has a producer ({hits}) — re-derive P334: "
            f"the guard's behaviour depended on the key being absent")


class TestTheGuardStillGuards:
    """The fix must not become a blanket disable — it changes WHICH readings
    are refused, not whether refusal is possible."""

    def test_a_genuine_collapse_on_a_complete_bar_is_still_refused(self):
        md = _md(k=0.03, bar_progress=1.0)   # 3% of normal volume, full bar
        assert _refuses(effective_volume_ratio(md))

    def test_a_healthy_complete_bar_is_allowed(self):
        md = _md(k=1.0, bar_progress=1.0)
        assert not _refuses(effective_volume_ratio(md))
