# scanner_baselines

Frozen output of `scripts/authority_consistency_audit.py` and
`scripts/silent_failure_audit.py`. The CI gate
(`tools/ci_check_invariants.py`) compares each push's scanner output to
these baselines and refuses the push if findings have INCREASED.

## Why this exists

CLAUDE.md P-entries P15 / P25 / P47 / P48 / P64 are all the same shape:
silent failure pattern. Each was caught manually, weeks-to-months after
the bug landed in production. The static scanners can detect the SHAPE
of these bugs but only if someone runs them. CI gating via these
baselines turns "operator remembered to run scanner" into "PR can't
merge without it".

## Files

- `authority_consistency_baseline.json` — frozen output of the authority
  scanner (Sections A–F: matrix wiring, ENABLE_* flags, constant drift,
  DRL invariants, real-gate audit, multi-site kwarg consistency).
  Comparison is structural per section.
- `silent_failure_baseline.json` — frozen counts of silent_failure_audit
  findings (try-except hits, dict.get misuse, ENABLE_* with no reader).
  Comparison is count-based: counts can DECREASE freely (good — fewer
  findings) but any increase fails the gate.

## Workflow when you make an intentional change

If your change legitimately adds a new finding (e.g., you intentionally
add a new ENABLE_* flag, accept a documented constant drift, or wire a
new agent that touches the matrix), regenerate the baselines:

```bash
python -X utf8 tools/ci_check_invariants.py --update
git add tools/scanner_baselines/
git commit -m "rebaseline: <one-line reason>"
```

The commit message should reference the P-entry or the change that
necessitated the rebaseline so future operators can audit "why did
this number go up".

## Workflow when CI fails

GitHub Actions will print the diff in the "Run codebase invariants
gate" step. The failed run also uploads the baseline files as an
artifact so you can inspect them locally without re-running the
scanner.

If the failure is a real bug:
1. Fix the bug.
2. Re-push. Baseline doesn't change; CI passes naturally.

If the failure is acceptable / known false positive:
1. Run `--update` locally (above).
2. Commit the rebaselined files in the same PR as your intentional
   change.

Never edit the baseline JSON files by hand. Always regenerate via
`--update` so the structure stays consistent with what the CI script
expects.

## Local dry-run

```bash
# Just print the diff; always exit 0 (good for inspecting before commit)
python -X utf8 tools/ci_check_invariants.py --diff-only

# Same as CI — exits 1 on new findings
python -X utf8 tools/ci_check_invariants.py
```
