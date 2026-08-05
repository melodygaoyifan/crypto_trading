# Archived tests

## test_production_patches.py

Tests HMATS **v3.6.1** `integration.core.production_reliability_patches`, which
has lived in `archive/integration/core/` since the initial v6.8.0 commit
(`7b70907`). The test also hardcodes `sys.path.insert(0, '/home/claude/hmats_v36')`
— a path from a machine unrelated to this repo.

It was moved here on 2026-08-04 because a **collection** error aborts the whole
pytest run, not just the offending file: `pytest tests/` returned zero results
for any developer who ran it without `--ignore`. CI never caught this because
`.github/workflows/test-suite.yml` invokes individual test files, never the
`tests/` directory.

Do not move it back without restoring the module it imports.
