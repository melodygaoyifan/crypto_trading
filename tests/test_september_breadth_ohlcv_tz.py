"""[P414b] build_breadth_ohlcv merged tz-aware Kraken rows into the tz-naive
history and raised 'Cannot compare tz-naive and tz-aware timestamps', so every
breadth-asset OHLCV FAILED and the ~09-15 breadth reads would have scored blind.
_naive_utc coerces both sides to the canonical tz-naive-UTC convention (what the
home OHLCV builder and the scorer use)."""
import pandas as pd
import pytest

from scripts.september_check import _naive_utc


def test_naive_stays_naive_utc():
    s = pd.Series(pd.to_datetime(["2026-08-01 00:00", "2026-08-01 04:00"]))
    out = _naive_utc(s)
    assert out.dt.tz is None
    assert list(out) == list(s)   # naive assumed-UTC is unchanged in value


def test_aware_becomes_naive_same_instant():
    s = pd.Series(pd.to_datetime(["2026-08-01 00:00", "2026-08-01 04:00"]).tz_localize("UTC"))
    out = _naive_utc(s)
    assert out.dt.tz is None
    # same wall-clock instant, tz dropped
    assert list(out) == list(pd.to_datetime(["2026-08-01 00:00", "2026-08-01 04:00"]))


def test_mixed_merge_sorts_without_error():
    """The exact failure: an existing NAIVE history + new AWARE Kraken rows."""
    old = pd.DataFrame({"timestamp": pd.to_datetime(["2026-08-01 00:00"]), "close": [1.0]})
    new = pd.DataFrame({"timestamp": pd.to_datetime(["2026-08-01 04:00"]).tz_localize("UTC"),
                        "close": [2.0]})
    old["timestamp"] = _naive_utc(old["timestamp"])
    new["timestamp"] = _naive_utc(new["timestamp"])
    merged = (pd.concat([old, new])
              .drop_duplicates(subset="timestamp", keep="last")
              .sort_values("timestamp").reset_index(drop=True))   # raised pre-fix
    assert len(merged) == 2
    assert merged["timestamp"].dt.tz is None


def test_the_builder_produces_naive_matching_the_home_convention():
    """A source guard: the Kraken row construction must not re-introduce a
    tz-aware timestamp (which is what broke the merge)."""
    import inspect, scripts.september_check as m
    src = inspect.getsource(m.build_breadth_ohlcv)
    assert 'tz="UTC"' not in src, "breadth rows must be built tz-naive (P414b)"
    assert "_naive_utc(df" in src and "_naive_utc(old" in src, "both frames must be normalized"
