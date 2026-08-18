#!/usr/bin/env python3
"""
training/exit_drl/train_exit_sac.py

Discrete SAC with CQL regularizer for exit-timing actions.

Design choices (per HMATS_EXIT_DRL_TRAINING.md Stage 2):
- Discrete SAC over 4 actions {HOLD, PARTIAL_EXIT, RELEASE_RUNNER, EXIT_ALL}.
- CQL (Kumar et al. 2020) regularizer to prevent OOD Q-value over-estimation
  under offline-only training. Critical — 9 live trades + synthetic oracle data
  means we have no online feedback to correct over-optimistic Q estimates.
- Balanced per-action sampling to counter the ~75% HOLD majority (Stage 4 pitfall #4).
- Time-ordered train/val split (last 20% is val) — shuffling would leak future bars.
- Dual-Q critic + soft target update, standard SAC bits.

Output: models/exit_drl/{ASSET}/exit_sac_best.pt — checkpoint with actor + critic,
plus val metrics (oracle alignment + avg val reward per episode).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("[FATAL] torch not installed. Run `pip install torch`.", file=sys.stderr)
    sys.exit(1)

# [P188] Imported, not restated. The comment below used to read "Must match
# generate_expert_trajectories.py" above two literals that nothing checked —
# the reader/writer-drift shape this codebase keeps paying for (P2, P15, P23,
# P85, P138-P140, P147, P152, P170, P171, P173, P176, P185). One definition,
# in the torch-free module, so a consumer that only needs the dimensions does
# not have to import a file that cannot load without torch.
from training.exit_drl.generate_expert_trajectories import (  # noqa: E402
    STATE_DIM, ACTION_DIM,
)

ACTION_NAMES = ['HOLD', 'PARTIAL_EXIT', 'RELEASE_RUNNER', 'EXIT_ALL']


# ────────────────────────────────────────────────────────────────
# Networks
# ────────────────────────────────────────────────────────────────
class DiscreteActor(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(state)
        probs = F.softmax(logits, dim=-1)
        return probs, logits


class DiscreteQCritic(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, hidden: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state), self.q2(state)


# ────────────────────────────────────────────────────────────────
# Agent
# ────────────────────────────────────────────────────────────────
class ExitSAC:
    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        cql_alpha: float = 1.0,
        device: str = 'cuda',
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.cql_alpha = cql_alpha

        self.actor = DiscreteActor().to(device)
        self.critic = DiscreteQCritic().to(device)
        self.critic_target = DiscreteQCritic().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def update(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        s, a, r, s_next, d = batch

        # ── Critic update (Bellman + CQL) ──
        with torch.no_grad():
            next_probs, next_logits = self.actor(s_next)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            q1_next, q2_next = self.critic_target(s_next)
            q_next = torch.min(q1_next, q2_next)
            # V(s') = E_a[Q(s',a) - alpha*log pi(a|s')]
            v_next = (next_probs * (q_next - self.alpha * next_log_probs)).sum(dim=-1)
            target_q = r + self.gamma * (1.0 - d.float()) * v_next

        q1, q2 = self.critic(s)
        q1_a = q1.gather(1, a.unsqueeze(-1)).squeeze(-1)
        q2_a = q2.gather(1, a.unsqueeze(-1)).squeeze(-1)
        bellman_loss = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)

        # CQL: logsumexp_a Q(s,a) - Q(s,a_dataset) → pulls down Q on OOD actions
        cql_q1 = torch.logsumexp(q1, dim=-1).mean() - q1_a.mean()
        cql_q2 = torch.logsumexp(q2, dim=-1).mean() - q2_a.mean()
        cql_loss = cql_q1 + cql_q2

        critic_loss = bellman_loss + self.cql_alpha * cql_loss
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_opt.step()

        # ── Actor update ──
        probs, logits = self.actor(s)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            q1_curr, q2_curr = self.critic(s)
            q_curr = torch.min(q1_curr, q2_curr)
        # maximize E_a[Q - alpha*log pi]  <=>  minimize E_a[alpha*log pi - Q]
        actor_loss = (probs * (self.alpha * log_probs - q_curr)).sum(dim=-1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()

        # ── Soft target update ──
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        return {
            'critic_loss': float(critic_loss.item()),
            'bellman_loss': float(bellman_loss.item()),
            'cql_loss': float(cql_loss.item()),
            'actor_loss': float(actor_loss.item()),
        }

    @torch.no_grad()
    def greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        probs, _ = self.actor(state)
        return probs.argmax(dim=-1)


# ────────────────────────────────────────────────────────────────
# Balanced per-action sampler
# ────────────────────────────────────────────────────────────────
class BalancedSampler:
    """
    Samples a batch with weighted probability per action, so the ~75% HOLD
    majority doesn't drown the other 3 actions. weight_i = 1 / count_i.
    """
    def __init__(self, actions: np.ndarray, rng: np.random.Generator):
        counts = np.bincount(actions, minlength=ACTION_DIM).astype(np.float64)
        counts = np.maximum(counts, 1.0)  # avoid div-by-zero
        per_action_weight = 1.0 / counts
        self.weights = per_action_weight[actions]
        self.weights = self.weights / self.weights.sum()
        self.n = len(actions)
        self.rng = rng

    def sample(self, batch_size: int) -> np.ndarray:
        return self.rng.choice(self.n, size=batch_size, replace=True, p=self.weights)


# ────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────
def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.require_gpu and device != 'cuda':
        print("[FATAL] --require-gpu set but CUDA unavailable.", file=sys.stderr)
        sys.exit(1)
    print(f"Device: {device}")

    data_path = Path(args.data_path) / f'{args.asset}_exit_trajectories.npz'
    print(f"Loading {data_path}")
    data = np.load(data_path)
    obs = data['observations']
    acts = data['actions']
    rews = data['rewards']
    next_obs = data['next_observations']
    terms = data['terminals']

    if obs.shape[1] != STATE_DIM:
        print(f"[FATAL] state dim mismatch: got {obs.shape[1]}, expected {STATE_DIM}", file=sys.stderr)
        sys.exit(1)

    n = len(obs)
    split = int(n * (1.0 - args.val_split))
    print(
        f"Loaded {n} transitions; train={split}, val={n - split}; "
        f"action dist: {dict(zip(ACTION_NAMES, np.bincount(acts, minlength=ACTION_DIM).tolist()))}"
    )

    train_obs = obs[:split]
    train_acts = acts[:split]
    train_rews = rews[:split]
    train_next = next_obs[:split]
    train_terms = terms[:split]
    val_obs = obs[split:]
    val_acts = acts[split:]
    val_rews = rews[split:]
    val_terms = terms[split:]

    rng = np.random.default_rng(args.seed)
    if args.balanced:
        sampler = BalancedSampler(train_acts, rng)
    else:
        sampler = None

    agent = ExitSAC(
        lr=args.lr, gamma=args.gamma, tau=args.tau,
        alpha=args.alpha, cql_alpha=args.cql_alpha, device=device,
    )

    best_score = -np.inf
    best_payload = None
    updates_per_epoch = max(1, split // args.batch_size)
    val_obs_t = torch.tensor(val_obs, device=device)

    for epoch in range(args.epochs):
        losses = []
        for _ in range(updates_per_epoch):
            if sampler is not None:
                idx = sampler.sample(args.batch_size)
            else:
                idx = rng.integers(0, split, size=args.batch_size)
            batch = (
                torch.tensor(train_obs[idx], device=device),
                torch.tensor(train_acts[idx], device=device, dtype=torch.long),
                torch.tensor(train_rews[idx], device=device, dtype=torch.float32),
                torch.tensor(train_next[idx], device=device),
                torch.tensor(train_terms[idx], device=device, dtype=torch.bool),
            )
            info = agent.update(batch)
            losses.append(info)

        # Validation
        with torch.no_grad():
            greedy = agent.greedy_action(val_obs_t).cpu().numpy()
        alignment = float((greedy == val_acts).mean())
        # Per-episode reward proxy: sum(val_rewards) / num_episodes(val)
        n_episodes = max(1, int(val_terms.sum()))
        mean_reward_per_episode = float(val_rews.sum() / n_episodes)
        action_hist = np.bincount(greedy, minlength=ACTION_DIM).tolist()

        # Validation score: alignment alone is the stable signal early on;
        # mean_reward_per_episode would mostly track the oracle's own reward on val,
        # which doesn't change between epochs (it's static data).
        score = alignment

        mean_losses = {k: float(np.mean([d[k] for d in losses])) for k in losses[0]}
        print(
            f"[{args.asset}] epoch={epoch:3d} "
            f"align={alignment:.3f} reward_ep={mean_reward_per_episode:+.4f} "
            f"greedy={dict(zip(ACTION_NAMES, action_hist))} "
            f"loss[bellman={mean_losses['bellman_loss']:.3f} "
            f"cql={mean_losses['cql_loss']:+.3f} "
            f"actor={mean_losses['actor_loss']:+.3f}]"
        )

        if score > best_score:
            best_score = score
            best_payload = {
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'epoch': epoch,
                'val_alignment': alignment,
                'val_reward_per_episode': mean_reward_per_episode,
                'action_histogram': action_hist,
                'hyperparams': {
                    'lr': args.lr, 'gamma': args.gamma, 'tau': args.tau,
                    'alpha': args.alpha, 'cql_alpha': args.cql_alpha,
                    'batch_size': args.batch_size, 'balanced': args.balanced,
                    'seed': args.seed,
                },
                'state_dim': STATE_DIM,
                'action_dim': ACTION_DIM,
                'action_names': ACTION_NAMES,
            }

    if best_payload is None:
        print("[FATAL] No epochs completed.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_path) / args.asset
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / 'exit_sac_best.pt'
    torch.save(best_payload, ckpt_path)

    meta = {k: v for k, v in best_payload.items()
            if k not in ('actor_state_dict', 'critic_state_dict')}
    (output_dir / 'exit_sac_best_meta.json').write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"\nSaved {ckpt_path}\n"
        f"  best alignment={best_payload['val_alignment']:.3f} @ epoch {best_payload['epoch']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', required=True, choices=['BTC', 'ETH', 'SOL'])
    parser.add_argument('--data-path', default='training/exit_drl/data')
    parser.add_argument('--output-path', default='models/exit_drl')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--cql-alpha', type=float, default=1.0,
                        help='CQL regularizer weight; 0 disables CQL')
    parser.add_argument('--balanced', action='store_true', default=True,
                        help='balanced per-action sampling (default on)')
    parser.add_argument('--no-balanced', dest='balanced', action='store_false')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--require-gpu', action='store_true',
                        help='hard-fail if CUDA is not available')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
