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
    # [P165] `docs/HMATS_PROFITABILITY_DEVELOPER_AUDIT.md` was removed from this
    # list: `git log --all -- <path>` is empty, so it has never been committed
    # and this test has raised FileNotFoundError since it was written. Add it
    # back if the doc ever lands.
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


def test_every_listed_hot_path_file_exists():
    """[P165] The list must name real files.

    A path that does not exist makes the encoding test below die with
    FileNotFoundError, which reads as "the encoding check failed" rather than
    "the list is stale" — the same missing-input-masquerading-as-a-result
    shape as P158/P159/P161. Assert the premise separately so the diagnosis
    is in the failure message.
    """
    missing = [p for p in HOT_PATH_FILES if not (PROJECT_ROOT / p).exists()]
    assert not missing, (
        f"HOT_PATH_FILES names {len(missing)} path(s) that do not exist: "
        f"{missing}. Either the file was moved/deleted (update the list) or "
        f"it was never committed (remove it) — until then the encoding check "
        f"below cannot run on the remaining files."
    )


def test_hot_path_files_decode_as_utf8_without_mojibake_markers():
    for relative_path in HOT_PATH_FILES:
        path = PROJECT_ROOT / relative_path
        text = path.read_bytes().decode("utf-8")
        for marker in MOJIBAKE_MARKERS:
            assert marker not in text, f"{relative_path} contains mojibake marker: {marker!r}"
