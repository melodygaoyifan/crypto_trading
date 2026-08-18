"""
[P309] Running P299's own features against REAL producer output found both of
them keyed on the wrong identifier.

`compute_per_strategy_ic` groups by the record's `strategy` field. P299 keyed
`POOLABLE_FAMILIES` and `ARCHIVED_FAMILIES` on the LEDGER-FILE PREFIX instead,
so:

  * `ma_filter` / `whale_filter` never matched `ma_filtered` / `whale_filtered`
    — those two families were silently left un-pooled while the feature
    reported success (ma_filtered pools to n=225 vs 75 per asset);
  * the archive section never rendered at all;
  * and the archive list was worse than merely inert — `"funding"` is a FILE
    holding THREE strategies, so had the key matched it would have buried
    `funding_mean_reversion` (50 directional) and `funding_post_etf_regime`
    (93) alongside the dead `funding_extreme` (0).

The P294 lesson, committed by the author who had just quoted it. The durable
fix is not the corrected spellings — it is that an allowlist entry matching
NOTHING is now reported, so the next wrong name costs one run instead of a
month (P264: registered-but-unmatched).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analytics.shadow_ic.compute_shadow_ic import (  # noqa: E402
    ARCHIVED_FAMILIES, POOLABLE_FAMILIES)

# Measured on the pulled ledgers, 2026-04-30 -> 2026-08-18.
# name -> (total records, records with a non-zero direction)
MEASURED = {
    "cascade_anticipation": (2082, 2),
    "funding_extreme": (2082, 0),
    "kyle_lambda": (2082, 0),
    "ofi": (2082, 1),
    "stop_hunt_defense": (2082, 0),
    "vpin_spike": (2082, 0),
    # alive — must never be archived
    "ml_factor": (2082, 924),
    "funding_mean_reversion": (2082, 50),
    "funding_post_etf_regime": (2082, 93),
}

# Ledger-file prefixes that are NOT strategy names. Putting one of these in
# either allowlist is the P299 bug.
PREFIX_NOT_A_NAME = {
    "ma_filter": "ma_filtered",
    "whale_filter": "whale_filtered",
    "cascade": "cascade_anticipation",
    "microstructure": "ofi (and kyle_lambda / stop_hunt_defense / vpin_spike)",
    "funding": "funding_extreme (and TWO LIVE siblings)",
    "derivflow": "liquidation_squeeze / liquidation_exhaustion",
    "sentvariant": "sent_momentum_linear / sent_momentum_hist / sent_contrarian",
}


class TestNeitherListHoldsALedgerPrefix:

    @pytest.mark.parametrize("prefix", sorted(PREFIX_NOT_A_NAME))
    def test_poolable_holds_no_prefix(self, prefix):
        assert prefix not in POOLABLE_FAMILIES, (
            f"{prefix!r} is a ledger-file prefix; the scorer groups by the "
            f"record's `strategy` field, which is {PREFIX_NOT_A_NAME[prefix]}")

    @pytest.mark.parametrize("prefix", sorted(PREFIX_NOT_A_NAME))
    def test_archived_holds_no_prefix(self, prefix):
        assert prefix not in ARCHIVED_FAMILIES, (
            f"{prefix!r} is a ledger-file prefix, not a strategy name")

    def test_the_two_that_p299_lost_are_now_poolable(self):
        for name in ("ma_filtered", "whale_filtered"):
            assert name in POOLABLE_FAMILIES, name


class TestTheArchiveNamesWhatItArchives:

    def test_only_measured_dead_strategies_are_archived(self):
        for name in ARCHIVED_FAMILIES:
            assert name in MEASURED, f"{name} archived without a measurement"
            total, directional = MEASURED[name]
            assert directional <= 2, (
                f"{name} emits {directional}/{total} directional records — "
                f"that is not dead")

    @pytest.mark.parametrize("alive", [
        "ml_factor", "funding_mean_reversion", "funding_post_etf_regime"])
    def test_live_siblings_are_not_swept_up(self, alive):
        """The dangerous half of the P299 bug: a FILE-keyed archive buries
        every strategy that happens to share the file."""
        assert alive not in ARCHIVED_FAMILIES

    def test_each_reason_carries_its_measurement(self):
        for name, reason in ARCHIVED_FAMILIES.items():
            assert any(ch.isdigit() for ch in reason), (
                f"{name}'s archive reason is an assertion, not a measurement")


class TestUnmatchedEntriesAreReported:
    """The durable guard. Spellings get fixed; the reporter stops the NEXT
    wrong name from being invisible."""

    def _run(self, tmp_path, monkeypatch, records):
        import json
        import analytics.shadow_ic.compute_shadow_ic as mod
        import pandas as pd

        d = tmp_path / "led"
        d.mkdir()
        with open(d / "regimebook_BTC.jsonl", "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

        idx = pd.date_range("2026-08-01", periods=400, freq="4h", tz="UTC")
        monkeypatch.setattr(mod, "load_ohlcv", lambda a: pd.DataFrame(
            {"timestamp": idx,
             "close": [100.0 + i % 5 for i in range(len(idx))]}))
        return mod

    def test_a_name_that_matches_nothing_is_named_in_the_output(
            self, tmp_path, monkeypatch, capsys):
        import time
        mod = self._run(tmp_path, monkeypatch, [{
            "strategy": "regimebook", "asset": "BTC", "direction": 1.0,
            "confidence": 1.0,
            "ts": time.time() - 3600,
        }])
        monkeypatch.setattr(mod, "POOLABLE_FAMILIES",
                            frozenset({"regimebook", "a_name_nothing_emits"}))
        mod.main(["--ledger-dir", str(tmp_path / "led"), "--window-days", "30",
                  "--pool-assets"])
        out = capsys.readouterr().out
        assert "matched NO strategy" in out
        assert "a_name_nothing_emits" in out
        assert "regimebook" not in out.split("matched NO strategy")[1][:200], (
            "a name that DID match must not be listed as unmatched")

    def test_it_stays_silent_when_every_name_matches(
            self, tmp_path, monkeypatch, capsys):
        """A note that always prints is wallpaper (P202)."""
        import time
        mod = self._run(tmp_path, monkeypatch, [{
            "strategy": "regimebook", "asset": "BTC", "direction": 1.0,
            "confidence": 1.0, "ts": time.time() - 3600,
        }])
        monkeypatch.setattr(mod, "POOLABLE_FAMILIES", frozenset({"regimebook"}))
        monkeypatch.setattr(mod, "ARCHIVED_FAMILIES", {})
        mod.main(["--ledger-dir", str(tmp_path / "led"), "--window-days", "30",
                  "--pool-assets"])
        assert "matched NO strategy" not in capsys.readouterr().out

    def test_the_note_explains_the_prefix_vs_name_distinction(self):
        """Whoever reads it must be told the actual cause, not just the fact."""
        src = (REPO / "analytics" / "shadow_ic"
               / "compute_shadow_ic.py").read_text(encoding="utf-8-sig")
        i = src.index("matched NO strategy")
        assert "ledger-file prefix" in src[i:i + 600]


class TestArchivingCannotDisableTheP213Refusal:
    """[P309] The archive block POPS its rows out of `per_strategy`. Placed
    before the P213 check, an all-archived window emptied the dict, `if
    per_strategy` went False, and the refusal silently returned 0 — disabling
    the guard that exists so "no prices" can never read as "no signal" (P199).
    """

    def test_an_all_archived_priceless_window_still_refuses(
            self, tmp_path, monkeypatch):
        import json
        import subprocess
        import time

        d = tmp_path / "led"
        d.mkdir()
        # An ARCHIVED family, on an asset that can have no parquet.
        with open(d / "microstructure_ZZZ.jsonl", "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "strategy": "ofi", "asset": "ZZZ", "direction": 1.0,
                "confidence": 1.0, "ts": time.time() - 3600}) + "\n")

        r = subprocess.run(
            [sys.executable, "-X", "utf8", "-m",
             "analytics.shadow_ic.compute_shadow_ic",
             "--ledger-dir", str(d), "--window-days", "30",
             "--prefixes", "microstructure"],
            capture_output=True, text=True, encoding="utf-8", cwd=str(REPO))
        assert r.returncode == 2, (
            f"an all-archived, price-less window must still REFUSE, not "
            f"return 0\nrc={r.returncode}\n{r.stdout[-600:]}\n{r.stderr[-400:]}")
        assert "REFUSING TO REPORT" in r.stderr

    def test_the_refusal_is_computed_before_the_archive_filter(self):
        """Source-order pin with a reason: both blocks are in main(), and the
        archive one mutates the dict the refusal reads."""
        src = (REPO / "analytics" / "shadow_ic"
               / "compute_shadow_ic.py").read_text(encoding="utf-8-sig")
        i_ref = src.index("REFUSING TO REPORT")
        i_arch = src.index("_arch = {k: v for k, v in per_strategy.items()")
        assert i_ref < i_arch, (
            "the archive filter pops rows out of per_strategy; running it "
            "first can empty the dict and silently skip the refusal")
