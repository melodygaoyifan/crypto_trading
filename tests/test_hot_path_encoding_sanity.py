from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Keep the scope small: hot-path runtime files and core operator docs.
HOT_PATH_FILES = [
    "main.py",
    "integration/integration_v36.py",
    "execution/passive_aggressive.py",
    "execution/learned_execution_policy.py",
    "portfolio/portfolio_brain_offensive.py",
    "docs/HMATS_E2E_TRAINING_GUIDE.md",
    "docs/HMATS_PROFITABILITY_DEVELOPER_AUDIT.md",
]

MOJIBAKE_MARKERS = (
    "\ufffd",  # Unicode replacement char
    "���",     # replacement cluster
    "鈺愨",     # mojibake cluster seen in corrupted dividers
    "锟斤拷",   # mojibake cluster seen in corrupted dividers
    "Ã",
    "â€™",
    "â€œ",
    "â€",
    "ðŸ",
)


def test_hot_path_files_decode_as_utf8_without_mojibake_markers():
    for relative_path in HOT_PATH_FILES:
        path = PROJECT_ROOT / relative_path
        text = path.read_bytes().decode("utf-8")
        for marker in MOJIBAKE_MARKERS:
            assert marker not in text, f"{relative_path} contains mojibake marker: {marker!r}"
