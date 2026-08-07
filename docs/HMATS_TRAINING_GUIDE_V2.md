# HMATS Training Guide V2 — the honest pipeline

**Created 2026-08-07 (P200-LADDER).** Replaces `docs/archive/HMATS_E2E_TRAINING_GUIDE_2026-02.md`,
which is retained as a forensic record only: every performance number in it is a
P164 leak artifact at zero fee, and P200's honest rerun of its formulation is
NOT PROMOTABLE on all three BTC folds. Do not copy commands from the archive.

**Read this first:** a training run here is a *measurement*, not a fix. The
expected honest result for a new formulation is near zero — the gates below
exist to say NO cheaply and honestly. A run that fails its gate produced
exactly the information it was run for.

---

## 0. Preconditions (one-time per machine)

- venv with `requirements.txt` + `requirements-train.txt` (torch CUDA for the 5090).
- Run everything **from the repo root** unless stated. Two path traps have each
  caused real incidents: `config/` vs `configs/` (the one-character typo behind
  the P164 GMM leak) and `fetch_binance_full.py`'s formerly cwd-relative output
  (P200 — fixed, now `__file__`-anchored, but stay in the root anyway).

## 1. Gate zero — is there anything to train on?

```bash
python -X utf8 training/scripts/edge_probe.py
# strict falsification variant (predictions fully past the GMM fit boundary):
python -X utf8 training/scripts/edge_probe.py --min-train 7200
```

Walk-forward Ridge/HGB per feature group per horizon, after-cost at 6bps RT,
against the P166-derived required-IC bar. **Exit 1 = NO_EDGE = stop.** RL
cannot conjure edge from features that carry none; this answers in minutes what
a training run answers in days. If only the strict run's numbers hold up,
believe the strict run. (2026-08-07 result: EDGE_CANDIDATE at the **16h**
horizon on all 3 assets; 4h mostly fails — see CLAUDE.md P200-LADDER.)

## 2. Data

```bash
python -X utf8 training/fetch_binance_full.py --years 6
```

6 years is the default and the minimum that matters: the 3-fold walk-forward
needs `n - 3*int(0.15n) - 42 >= 5000` → **~9,200 4H bars (~4.2y)**. The old
3-year default is exactly what produced parquets on which folds 2/3 silently
skip. Coinglass external history (funding/OI/liq, ~180 days of API depth):

```bash
python -X utf8 training/scripts/fetch_coinglass_history.py
# then convert 4h -> the _1d files rebuild expects (see P200 session notes)
```

## 3. Rebuild the feature parquets

```bash
python -X utf8 training/scripts/rebuild_pipeline.py --smooth 2
```

This is the leak-free path: **causal** wavelet denoise (P164) and **split-aware
GMM** (P200 — fit only on rows before the strictest fold boundary;
`--gmm-no-split` is an explicit leaky opt-in for visualization only). Verify
before proceeding — a leaky and a clean GMM are indistinguishable by value:

```bash
python -X utf8 -c "import json; print(json.load(open('training/training_data/gmm_models/BTC/gmm_config.json'))['fit_policy'])"
# MUST print: split_aware
```

Expected scale (2026-08): BTC/ETH 13,095 bars, SOL 13,034, 122 features.
`has_external_data` coverage ~8% is a known limitation (Coinglass depth), not
an error.

## 4. Split manifest

```bash
python -X utf8 training/scripts/generate_split_manifest.py
```

Regenerate whenever the data range changes. Fold_1's validation window should
cover the most recent period (it includes the live era — the first fold that
ever has).

## 5. Train

```bash
python -X utf8 -u training/train_drl_full.py \
    --asset BTC --extractor lstm_film_a \
    --venue coinbase --fee-side taker \
    --decision-interval 4 \
    --tag <FRESH_TAG> --no-progress-bar
```

Every flag is load-bearing:

| flag | why it cannot be omitted |
|---|---|
| `--extractor lstm_film_a` | the runtime hardcodes a 1008-dim (126×8) input; the default ULTIMATE path produces a 126-dim model the runtime cannot consume (P200 blocker 3) |
| `--venue coinbase --fee-side taker` | the default is kraken/26bps — a venue that has been structurally flat since 2026-06-13. The sleeve that actually trades pays Coinbase 3bps |
| `--decision-interval 4` | acts every 4 bars (16h), holds between — the only horizon where the edge probe found after-cost signal, and it structurally caps the churn that killed every prior run (~1,900 trades/fold, friction ≥ the loss) |
| `--tag <FRESH_TAG>` | **without a fresh tag the trainer restores old folds from cache and reports the stale numbers as if it trained** (P200 launch gotcha — the first "run" completed in 9 seconds) |

Do NOT pass `--config config/optuna_winner.json`: it is tainted (BTC-only, 19
trials, zero-fee objective) and two of its values (`net_arch`, `buffer_size`)
never reached the model anyway.

## 6. Judge — the only gate that counts

The run self-reports per fold. A fold PASSES only if it beats **all three**
baselines on after-cost PnL through the same eval env at the same fees, with a
bootstrap Sharpe CI excluding zero (P182):

- `buy_and_hold`
- `sma_200bar`
- `ridge_16h` — the supervised alternative fit on the same train fold with the
  same features at the same cadence. **An RL policy that cannot beat the ridge
  it shares features with is not promotable** (P200-LADDER; the literature and
  our own probe both say the signal is weak-linear).

`NOT PROMOTABLE` in the summary is a verdict, not an error. Never re-run with
looser friction to make a gate pass — that instruction appeared in the old
guide (its L2101) and is the P179 defect written down as procedure.

Reference points for a healthy run: early stopping at 355K–400K steps is
normal (the old guide's "+700 reward by 500K" trajectory was the leak);
eval-reward dispersion should be non-zero (a ±0.00 std means the P181 defect
has returned).

## 7. Promotion — CLAUDE.md P200-LADDER

A passing fold does not deploy anything. The ladder from there:

1. **30-day live shadow**: deploy checkpoints at SHADOW authority (unchanged
   behavior — inference + signal logging only), measure live IC via the
   attribution logs against the cost-aware P166 gate on FORWARD data only.
2. **Three deliberate flips, in order**: gate promote via
   `data/drl_promotion_state.json` (`drl.force_active` stays **false** — the
   persisted level being authoritative is the point of P200); fusion re-admits
   DRL automatically at ACTIVE; the sleeve consumes it only when the P206
   `coinbase_use_gated_intent` flag is flipped — a separate, operator-watched
   step.

Kill criteria at every rung. If the RL candidate dies at rung 2, the
`ridge_16h` signal itself becomes the promotion candidate through the same
remaining rungs — the edge does not need to be an RL policy to be traded.

## 8. Iron laws (amended)

The full list lives in the archive (L229–291) and remains almost entirely
valid. Two amendments (P164/P200):

- **#26**: the wavelet denoise must be **causal** (`wavelet_denoise_causal`,
  trailing 256-bar window) — "applied to both train and runtime" is not
  sufficient; that check passed while every training row saw the future.
- **#35**: the promotion gate is CLAUDE.md **P200-LADDER**, not the old
  three-condition EXIT_ONLY rule.

Unchanged and still binding: `ent_coef=0.1` fixed (never "auto"), obs 126 ×
stack 8 = 1008, DummyVecEnv only, per-asset RobustScaler fit on train only,
GMM/scaler/OOD from the same fold as the policy (P4 mixed-fold incident),
`n_folds=3 / val_ratio=0.15 / gap=42`.
