from pathlib import Path


FILES = [
    Path("execution/timing_engine.py"),
    Path("scripts/reflect_weekly.py"),
    Path("scripts/verify_suite.py"),
    Path("scripts/verify_sanity_suite.py"),
    Path("scripts/futures_vs_spot_fee_analysis.py"),
    Path("market/multi_asset_intelligence.py"),
    Path("training/scripts/generate_split_manifest.py"),
]


def test_legacy_tooling_has_no_datetime_utcnow():
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        assert "datetime.utcnow(" not in text, f"{path} still uses datetime.utcnow()"
