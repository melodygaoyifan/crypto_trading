#!/usr/bin/env python3
"""[P286] The deep-sequence exam that never actually ran — GRU + TCN under
the pooled/live-expression certification, on the upgraded honest data.

WHY THIS EXISTS: the P281 campaign's "ETH GRU lockbox +0.00" was VACUOUS —
ds_model_pipeline's walk-forward skips GRU ("omitted for brevity"), a
check that couldn't run (P174 class). TCN was never re-run at honest cost
with the real futures columns. With the operator's 5090 free, this closes
the one genuinely unmeasured model-class cell.

PRE-REGISTERED BEFORE RUNNING (no search — the fresh 196-trial searches
just re-demonstrated CV overfit on this exact data):
  * TWO fixed configs, period: gru32 (hidden 32, 1 layer, lr 1e-3,
    40 epochs, batch 256, 8-frame windows) and tcn32 (3 causal conv
    layers, ch 32, kernel 3, dilations 1/2/4, dropout 0.1, lr 1e-3,
    40 epochs). Target ret, feature set top24, refit=180 (the SOL hgb
    precedent for slower refits), assets BTC/ETH/SOL.
  * THREE-SEED PREDICTION AVERAGING (seeds 1,2,3) — an ensemble, not a
    selection: no seed is ever picked, so the P283b single-seed caveat
    never arises for this exam.
  * Decisive cell = the P283b bar verbatim: pooled fold-val windows,
    LIVE sign expression (|dir|>=0.15 bang-bang), carry included,
    block-bootstrap CI must exclude zero. DSR reported at N=10 per asset
    (the zoo's 8 + these 2 — multiplicity is charged, not hidden).
  * A PASS earns exactly one thing: a Rung-3 shadow slot through the
    P166 forward gate (the mlp_small path). Nothing deploys from here.
  * The fold-val windows are multiply read (P260 discount applies) and
    the reads are ledgered under deep_seq:p286.

WALK-FORWARD PARITY: mirrors train_supervised_full.walk_forward_z's
semantics exactly — refit every `refit` bars, train on ALL history before
t0 purged by H, min-train 800, z = preds / std(train preds) — but passes
the full matrix so sequence windows can reach back before each block
(fit_predict's flat interface cannot carry window history; divergence
from the zoo's purge arithmetic here would make the exam measure a
different protocol, so the constants are imported, never restated).
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "training" / "scripts"))

from splits import assert_clean_gmm, record_window_usage  # noqa: E402
import train_supervised_full as T  # noqa: E402
from regime_model_lab import _bar_carry_rate  # noqa: E402
from pooled_certification import live_expression  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 8
SEEDS = (1, 2, 3)
REFIT = 180
EPOCHS = 40
BATCH = 256
LR = 1e-3
N_TRIALS = 10  # zoo's 8 + these 2 — the DSR denominator, charged honestly
ASSETS = ("BTC", "ETH", "SOL")
FAMILIES = ("gru32", "tcn32")
OUT = REPO / "training" / "reports" / "deep_seq_cert_p286.json"


class GRU32(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.gru = nn.GRU(n_feat, 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TCN32(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        layers = []
        ch_in = n_feat
        for d in (1, 2, 4):
            layers += [nn.Conv1d(ch_in, 32, kernel_size=3, dilation=d,
                                 padding=2 * d),
                       nn.ReLU(), nn.Dropout(0.1)]
            ch_in = 32
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(32, 1)

    def forward(self, x):          # x: [n, seq, feat]
        h = self.net(x.transpose(1, 2))       # causal: crop the right pad
        return self.head(h[:, :, :x.shape[1]][:, :, -1]).squeeze(-1)


def _windows(Xs, rows):
    """[n, SEQ_LEN, F] windows ending at each row (rows all >= SEQ_LEN-1,
    window contents pre-validated by the caller)."""
    return np.stack([Xs[r - SEQ_LEN + 1:r + 1] for r in rows])


def _fit_one(family, Wtr, ytr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = (GRU32 if family == "gru32" else TCN32)(Wtr.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    Wt = torch.tensor(Wtr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
    n = len(Wt)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for b in range(0, n, BATCH):
            bi = perm[b:b + BATCH]
            opt.zero_grad()
            loss = lossf(model(Wt[bi]), yt[bi])
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def _predict(model, W):
    out = []
    Wt = torch.tensor(W, dtype=torch.float32, device=DEVICE)
    for b in range(0, len(Wt), 2048):
        out.append(model(Wt[b:b + 2048]).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def deep_walk_forward_z(family, X, y, fidx, start, end):
    """walk_forward_z's semantics for sequence models: refit every REFIT
    bars on ALL history before the block, purged by T.H, min-train 800;
    z = seed-averaged preds / std(seed-averaged train preds)."""
    Xf = X[:, fidx]
    z = np.full(len(y), np.nan)
    for t0 in range(start, end, REFIT):
        pe = t0 - T.H
        # valid train rows: full window + target present, ending before pe
        ok_row = ~np.isnan(Xf).any(axis=1)
        ok_win = np.ones(len(y), dtype=bool)
        for k in range(SEQ_LEN):
            ok_win[SEQ_LEN - 1:] &= ok_row[SEQ_LEN - 1 - k:len(y) - k]
        ok_win[:SEQ_LEN - 1] = False
        rows = np.where(ok_win[:pe] & ~np.isnan(y[:pe]))[0]
        if len(rows) < 800:
            continue
        t1 = min(t0 + REFIT, end)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xf[rows])
        Xs = sc.transform(np.nan_to_num(Xf))
        Wtr = _windows(Xs, rows)
        pred_rows = np.arange(t0, t1)
        Wpr = _windows(Xs, pred_rows)
        tr_preds, pr_preds = [], []
        for seed in SEEDS:
            m = _fit_one(family, Wtr, y[rows], seed)
            tr_preds.append(_predict(m, Wtr))
            pr_preds.append(_predict(m, Wpr))
        tr_avg = np.mean(tr_preds, axis=0)
        pr_avg = np.mean(pr_preds, axis=0)
        sig = float(np.std(tr_avg)) or 1e-9
        z[t0:t1] = pr_avg / sig
    return z


def main() -> int:
    out = {"generated": datetime.now(timezone.utc).isoformat(),
           "device": DEVICE,
           "preregistered": {"families": list(FAMILIES), "seq_len": SEQ_LEN,
                             "seeds": list(SEEDS), "refit": REFIT,
                             "epochs": EPOCHS, "lr": LR, "fset": "top24",
                             "target": "ret", "n_trials_dsr": N_TRIALS},
           "pass_bar": "pooled after-cost CI excludes zero under the LIVE "
                       "expression, carry included (the P283b bar); a PASS "
                       "earns a Rung-3 shadow slot, nothing deploys",
           "assets": {}}
    from pooled_certification import dsr
    for asset in ASSETS:
        assert_clean_gmm(asset)
        X, targets, close, gmm, feats = T.load_asset(asset)
        n = len(close)
        y = targets["ret"]
        carry_rate = np.nan_to_num(
            np.asarray(_bar_carry_rate(asset, n), dtype=float))
        ret = np.zeros(n); ret[1:] = close[1:] / close[:-1] - 1.0
        out["assets"][asset] = {}
        for family in FAMILIES:
            pos_full = np.zeros(n)
            spans = []
            for fold, (train_end, val_start, val_end) in \
                    T.fold_splits(n).items():
                fsets = T.select_features(X, targets["ret"], train_end,
                                          feats)
                z = deep_walk_forward_z(family, X, y, fsets["top24"],
                                        val_start, val_end)
                pos_full[val_start:val_end] = T.positions_from_z(
                    z[val_start:val_end], None)
                spans.append((val_start, val_end))
                print(f"  [{asset}/{family}] {fold} done", flush=True)
            lo = min(s for s, _ in spans); hi = max(e for _, e in spans)
            record_window_usage(f"deep_seq:p286:{family}", asset, lo, hi,
                                "validation")
            mask = np.zeros(n, dtype=bool)
            for s, e in spans:
                mask[s:e] = True
            pos = live_expression(pos_full)
            strat = np.zeros(n)
            strat[1:] = pos[:-1] * ret[1:] - pos[:-1] * carry_rate[1:]
            cost = np.zeros(n)
            cost[1:] = np.abs(np.diff(pos)) * (T.COST_BPS[asset] / 2.0) / 1e4
            seg = (strat - cost)[mask]
            sd = float(np.nanstd(seg))
            sh = float(np.nanmean(seg) / sd * math.sqrt(T.BARS_PER_YEAR)) \
                if sd > 0 else 0.0
            clo, chi = T.block_bootstrap_ci(seg)
            passes = clo is not None and clo > 0
            out["assets"][asset][family] = {
                "pooled_bars": int(mask.sum()),
                "net_pct": round(float(np.nansum(seg)) * 100, 2),
                "sharpe": round(sh, 3),
                "ci": [None if clo is None else round(clo, 3),
                       None if chi is None else round(chi, 3)],
                "dsr": round(dsr(sh, int(mask.sum()), N_TRIALS, seg=seg), 3),
                "passes": passes}
            r = out["assets"][asset][family]
            print(f"[{asset}] {family}: net {r['net_pct']:+7.2f}%  "
                  f"sh {r['sharpe']:+.2f}  CI[{r['ci'][0]},{r['ci'][1]}]  "
                  f"DSR {r['dsr']:.2f}  "
                  f"{'PASS' if passes else 'FAIL'}", flush=True)
            OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nreport: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
