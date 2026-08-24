"""[P390] FOMC-minutes + NFP windows in the static 2026 event calendar.

Observation-only, like the P277 windows they join: `in_event_window` feeds
the `eventfilter` shadow LEDGER; no enforce flag exists by design (P277) and
this change adds none. Unscheduled speech is deliberately NOT here (a static
calendar cannot carry it, P2) — that coverage is the keyfig headline tag.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from defense import enhancement_shadows as es


def _t(iso_day, hour, minute=0):
    y, m, d = (int(x) for x in iso_day.split("-"))
    return datetime(y, m, d, hour, minute, tzinfo=timezone.utc)


class TestNewWindows:
    @pytest.mark.parametrize("day", es.FOMC_MINUTES_DAYS_2026)
    def test_minutes_window_fires_inside_and_not_outside(self, day):
        assert es.in_event_window(_t(day, 17)) == "fomc_minutes"
        assert es.in_event_window(_t(day, 20)) == "fomc_minutes"
        assert es.in_event_window(_t(day, 16)) in (None, "nfp", "cpi")
        assert es.in_event_window(_t(day, 21)) is None

    @pytest.mark.parametrize("day", es.NFP_RELEASE_DAYS_2026)
    def test_nfp_window_fires_inside_and_not_outside(self, day):
        assert es.in_event_window(_t(day, 12, 30)) == "nfp"
        assert es.in_event_window(_t(day, 15)) == "nfp"
        assert es.in_event_window(_t(day, 11)) is None
        assert es.in_event_window(_t(day, 16)) is None

    def test_dates_are_wellformed_2026_and_disjoint_from_decision_days(self):
        for day in es.FOMC_MINUTES_DAYS_2026 + es.NFP_RELEASE_DAYS_2026:
            dt = datetime.fromisoformat(day)
            assert dt.year == 2026
            assert day not in es.FOMC_DECISION_DAYS_2026
            assert day not in es.CPI_RELEASE_DAYS_2026

    def test_nfp_days_are_fridays_minutes_are_wednesdays(self):
        # the schedules' own structure: NFP = first Friday; minutes land on
        # the Wednesday three weeks after the decision Wednesday
        for day in es.NFP_RELEASE_DAYS_2026:
            assert datetime.fromisoformat(day).weekday() == 4, day
        for day in es.FOMC_MINUTES_DAYS_2026:
            assert datetime.fromisoformat(day).weekday() == 2, day


class TestExistingWindowsUnchanged:
    def test_decision_cpi_sunday_still_fire(self):
        assert es.in_event_window(_t("2026-09-16", 14)) == "fomc"
        assert es.in_event_window(_t("2026-09-11", 11)) == "cpi"
        # Sunday 2026-08-23 22:30 UTC
        assert es.in_event_window(datetime(2026, 8, 23, 22, 30,
                                           tzinfo=timezone.utc)) == "sunday_thin"
        assert es.in_event_window(_t("2026-08-19", 14)) is None

    def test_decision_day_outranks_minutes_shape(self):
        # precedence is by clause order; a decision day inside 12-22 UTC must
        # read "fomc" even if a minutes/nfp date ever collided with it
        assert es.in_event_window(_t("2026-12-09", 18)) == "fomc"

    def test_still_observation_only_no_enforce_flag(self):
        # P277: enforcement is its own P-entry; this file must not grow one
        import io
        src = io.open(es.__file__, encoding="utf-8").read()
        assert "eventfilter_enforce" not in src
