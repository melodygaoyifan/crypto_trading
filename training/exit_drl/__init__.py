"""Exit DRL training package.

Per-spec `HMATS_EXIT_DRL_TRAINING.md` (2026-04-24):
- Trains a separate DRL agent for exit timing only.
- Never decides direction (long/short) — that's the TQC direction DRL in `drl/`.
- Max authority: EXIT_ONLY. Governed by the existing DRLPromotionGate.

Three DRLs in the codebase (don't confuse them — see CLAUDE.md P10):
  1. `drl/ensemble.py` + TQC models (ACTIVE): direction + confidence.
  2. `agents/drl_agent.py` (DISABLED): tranche/exit scaffolding, Phase-2 stub.
  3. `training/exit_drl/` (this package): exit-timing-only discrete SAC, v1 scope.
"""
