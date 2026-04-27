"""Golden-trace / replay test corpus.

The .jsonl files in this directory are FROZEN captures from live attribution.
They serve as both INPUT (per-agent signals) and IMPLICIT ground truth
(any change to fusion logic that produces a different output for the same
input is a regression — intentional or not — and must update the snapshot).
"""
