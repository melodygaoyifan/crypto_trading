"""
Smoke tests for training/exit_drl/train_exit_sac.py.

Doesn't train a real model — just verifies:
- Actor/critic shapes align with STATE_DIM / ACTION_DIM.
- One update step runs without NaN on synthetic data.
- Balanced sampler weights inversely by action frequency.
- Checkpoint save/load round-trips the network weights.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.exit_drl.train_exit_sac import (
    ACTION_DIM, ACTION_NAMES, STATE_DIM,
    BalancedSampler, DiscreteActor, DiscreteQCritic, ExitSAC,
)


def test_actor_output_shapes():
    actor = DiscreteActor()
    s = torch.randn(4, STATE_DIM)
    probs, logits = actor(s)
    assert probs.shape == (4, ACTION_DIM)
    assert logits.shape == (4, ACTION_DIM)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_critic_output_shapes():
    critic = DiscreteQCritic()
    s = torch.randn(4, STATE_DIM)
    q1, q2 = critic(s)
    assert q1.shape == (4, ACTION_DIM)
    assert q2.shape == (4, ACTION_DIM)


def test_sac_update_step_no_nan():
    device = 'cpu'  # tests must run on CI without GPU
    agent = ExitSAC(device=device, cql_alpha=1.0)
    batch_size = 16
    s = torch.randn(batch_size, STATE_DIM, device=device)
    a = torch.randint(0, ACTION_DIM, (batch_size,), device=device, dtype=torch.long)
    r = torch.randn(batch_size, device=device)
    s_next = torch.randn(batch_size, STATE_DIM, device=device)
    d = torch.zeros(batch_size, device=device, dtype=torch.bool)

    info = agent.update((s, a, r, s_next, d))
    for k, v in info.items():
        assert np.isfinite(v), f"{k} produced non-finite value: {v}"


def test_balanced_sampler_gives_higher_weight_to_rare_actions():
    rng = np.random.default_rng(0)
    # 900 HOLD, 50 PARTIAL, 10 RELEASE, 40 EXIT — intentionally skewed
    actions = np.concatenate([
        np.zeros(900, dtype=np.int64),
        np.ones(50, dtype=np.int64),
        np.full(10, 2, dtype=np.int64),
        np.full(40, 3, dtype=np.int64),
    ])
    sampler = BalancedSampler(actions, rng)
    # Weight of each HOLD sample should be much smaller than each RELEASE sample
    hold_weight = sampler.weights[0]
    release_weight = sampler.weights[950]
    assert release_weight > hold_weight * 10.0

    # Draw a big batch and check that the distribution is closer to uniform than the data
    idx = sampler.sample(batch_size=10000)
    drawn = actions[idx]
    dist = np.bincount(drawn, minlength=ACTION_DIM) / len(drawn)
    # Each action should appear with >5% frequency (vs 1% for RELEASE in raw data)
    assert dist.min() > 0.05


def test_checkpoint_round_trip(tmp_path):
    agent = ExitSAC(device='cpu')
    payload = {
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'action_names': ACTION_NAMES,
    }
    ckpt_path = tmp_path / 'ckpt.pt'
    torch.save(payload, ckpt_path)

    agent2 = ExitSAC(device='cpu')
    loaded = torch.load(ckpt_path, weights_only=False)
    agent2.actor.load_state_dict(loaded['actor_state_dict'])
    agent2.critic.load_state_dict(loaded['critic_state_dict'])

    # Reloaded network should produce identical outputs
    s = torch.randn(4, STATE_DIM)
    with torch.no_grad():
        p1, _ = agent.actor(s)
        p2, _ = agent2.actor(s)
    assert torch.allclose(p1, p2, atol=1e-7)
