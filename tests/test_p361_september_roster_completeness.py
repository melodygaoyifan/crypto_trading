"""[P361] An accruing ledger with no read date, and a decision tree overtaken
by four later entries.

Found by running the repo's own standing instruments instead of guessing what
was outstanding — which is what P230/P333 built them for.

--------------------------------------------------------------------------
1. `whale_filter` WAS ACCRUING WITH NO SCHEDULED READ
--------------------------------------------------------------------------
`whale_filter_*.jsonl` has been written every tick since 2026-08-18 and
`whale_filtered` is a registered, POOLABLE scorer family — yet it appeared
ZERO times in `docs/SEPTEMBER_DECISION_TREE.md` and had no row in the
countdown. Its sibling `ma_filter` had both.

The roster's own comment (added by P287, which widened it for exactly this
reason) says it "must cover EVERY accruing candidate the scorer reads". It
did not. **An evidence stream with no read date is the P199/P230 shape — a
decision procedure that depends on someone remembering quietly becomes
never** — and it is the shape this script was written to close, one family
over.

So the durable fix is not the row: it is that the roster is now held to the
SCORER's own default prefix list, in both directions.

--------------------------------------------------------------------------
2. THE ma_filter SECTION DESCRIBED A WORLD FOUR ENTRIES OLD
--------------------------------------------------------------------------
The tree calls it "the single most likely promotion" and prescribes
`coinbase_ma_filter_enforce: true` on a PASS. Since it was written:

  P324  measured it NOT EARNED at the pre-committed bar
  P337  measured against the decider it actually filters, and its
        disagreements marked entries that did BETTER (contrast -10.0)
  P348  showed no obtainable sample makes it significant — the ~09-07 read
        moves model_alpha's t from 0.73 to 0.79
  P356  DISARMED it, by operator instruction, on that arithmetic

A stale prescription is worse than none: this is the document a future
session executes from, and it would have sent that session to arm something
the operator deliberately turned off. The read still happens — the evidence
is free either way (P340) — but a PASS now has to argue with four entries,
and the roster says so at the point of action.
"""

import datetime as dt
import importlib.util
import pathlib
import re

import main

REPO = pathlib.Path(main.__file__).parent
TREE = REPO / "docs" / "SEPTEMBER_DECISION_TREE.md"


def _candidates():
    spec = importlib.util.spec_from_file_location(
        "_sep_check", REPO / "scripts" / "september_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CANDIDATES


def _scorer_prefixes():
    """Read the SCORER's own default list — a hardcoded mirror here would be
    the drift this test exists to catch (P310/P172)."""
    src = (REPO / "analytics" / "shadow_ic"
           / "compute_shadow_ic.py").read_text(encoding="utf-8")
    i = src.index("prefixes: Tuple[str, ...] = (")
    block = src[i:src.index("since:", i)]
    return set(re.findall(r'"([a-z0-9_]+)"', block))


# Families the scorer reads that deliberately have NO September read date,
# each with the entry that decided it. An exemption must be a DECISION, not a
# gap — that distinction is the whole point of the roster.
_NO_READ_BY_DECISION = {
    "microstructure": "P299 ARCHIVED (0/2082 directional)",
    "cascade": "P299 ARCHIVED (2/2082)",
    "funding": "P299 — funding_extreme archived; the two live siblings are "
               "not September candidates",
    "ml_factor": "P199 PROMOTE was plausibly in-sample; no scheduled re-read",
    "sentvariant": "P296 settled offline on 3,116 days of history",
}


def test_every_family_the_scorer_reads_has_a_read_date_or_a_reason():
    """[P361] The direction that failed: `whale_filter` accrued for days with
    no read date because nothing checked the roster against the scorer."""
    covered = {prefix for prefix, _d, _a in _candidates().values()}
    unscheduled = sorted(_scorer_prefixes() - covered - set(_NO_READ_BY_DECISION))
    assert not unscheduled, (
        f"the scorer reads {unscheduled} but nothing schedules a read — an "
        f"accruing ledger nobody looks at is the P199/P230 gap this roster "
        f"exists to close. Add a CANDIDATES row, or an entry in "
        f"_NO_READ_BY_DECISION naming the P-entry that decided against it."
    )


def test_no_exemption_is_a_parking_spot():
    """The other direction (P310): an exemption must name the decision, and
    must still refer to something the scorer actually reads — otherwise the
    list silently becomes a place to hide new gaps."""
    prefixes = _scorer_prefixes()
    for name, reason in _NO_READ_BY_DECISION.items():
        assert name in prefixes, (
            f"{name} is exempted but the scorer no longer reads it — delete "
            f"the exemption rather than leave it as coverage that is not"
        )
        assert re.search(r"P\d{2,4}", reason), (
            f"{name}: exemption cites no P-entry, so it is an opinion"
        )


def test_whale_filter_specifically_has_a_read_date():
    """The instance, pinned alongside the class — it is the one that was
    missing, and its ledger is live right now."""
    c = _candidates()
    assert "whale_filter" in c, "whale_filter has no September read"
    prefix, since, action = c["whale_filter"]
    assert prefix == "whale_filter"
    assert dt.date.fromisoformat(since) == dt.date(2026, 8, 18), (
        "the ledger's first file is whale_filter_20260818.jsonl"
    )


def test_both_disarmed_filters_state_what_a_PASS_must_argue_with():
    """[P361] A stale prescription is worse than none: this roster is what a
    future session acts on, and 'arm it on PASS' would send them to turn on
    something the operator deliberately disarmed (P356)."""
    c = _candidates()
    for name, must_cite in (("ma_filter", ("P324", "P337", "P348", "P356")),
                            ("whale_filter", ("P324", "P348", "P356"))):
        action = c[name][2]
        for p in must_cite:
            assert p in action, (
                f"{name}'s action does not cite {p} — a PASS would read as a "
                f"plain arming despite four entries against it"
            )


def _section(text: str, heading: str) -> str:
    """Text from a heading to the next one — a byte window would match a
    neighbouring section and pin nothing (P320's lesson)."""
    i = text.index(heading)
    j = text.find(chr(10) + "## ", i + 1)
    return text[i:j if j != -1 else len(text)]


def test_the_decision_tree_covers_both_filters_and_is_not_stale():
    """[P361] Both guards here were VACUOUS on their first probe and the
    probes were right: `"P356" in section` still matched a different mention
    of P356 in the same section, and checking for the substring
    "whale_filter" still matched the section's body after its heading was
    removed. Pinned on the specific CLAIMS instead."""
    text = TREE.read_text(encoding="utf-8")

    assert "## ~Sep 17 — whale_filter" in text, (
        "the decision tree has no whale_filter SECTION, so one of the two "
        "entry filters has no written read procedure (it had none at all "
        "before P361, while its sibling had a full one)"
    )

    ma = _section(text, "## ~Sep 7 — ma_filter")
    assert "coinbase_ma_filter_enforce: false" in ma, (
        "the ma_filter section no longer records that the flag is DISARMED — "
        "it would read as the most likely promotion again, and a future "
        "session would arm what the operator deliberately turned off (P356)"
    )
    for p in ("P324", "P337", "P348", "P356"):
        assert p in ma, f"the ma_filter section does not cite {p}"

    wh = _section(text, "## ~Sep 17 — whale_filter")
    assert "coinbase_whale_filter_enforce" in wh, (
        "the whale section does not name the flag its read decides"
    )
    for p in ("P324", "P348", "P356"):
        assert p in wh, f"the whale_filter section does not cite {p}"
