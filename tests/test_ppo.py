"""Validate PPO components"""

import numpy as np
import torch

from circuitrl.agents.ppo_agent import ActorCritic, PPOAgent, RolloutBuffer
from circuitrl.envs.circuit_env import CircuitEnv

N_PARAMS = 10  # 10 circuit parameters
OBS_DIM = 18   # 10 params + 4 metrics + 4 targets


def test_actor_critic_forward():
    net = ActorCritic(OBS_DIM, N_PARAMS)
    obs = torch.randn(4, OBS_DIM)

    logits_list, values = net(obs)
    assert len(logits_list) == N_PARAMS
    assert logits_list[0].shape == (4, 3)  # 3 choices per param
    assert values.shape == (4,)


def test_actor_critic_get_action():
    net = ActorCritic(OBS_DIM, N_PARAMS)
    obs = torch.randn(1, OBS_DIM)

    actions, log_prob, value = net.get_action(obs)
    assert actions.shape == (1, N_PARAMS)
    assert torch.all((actions >= 0) & (actions < 3))
    assert log_prob.shape == (1,)
    assert value.shape == (1,)


def test_actor_critic_evaluate():
    net = ActorCritic(OBS_DIM, N_PARAMS)
    obs = torch.randn(8, OBS_DIM)
    actions = torch.randint(0, 3, (8, N_PARAMS))

    log_probs, entropy, values = net.evaluate(obs, actions)
    assert log_probs.shape == (8,)
    assert entropy.shape == (8,)
    assert values.shape == (8,)
    assert torch.all(log_probs <= 0)  # log probs are non-positive


def test_rollout_buffer_gae():
    buf = RolloutBuffer(n_steps=10, obs_dim=OBS_DIM, n_params=N_PARAMS)

    for i in range(10):
        buf.store(
            obs=np.random.randn(OBS_DIM).astype(np.float32),
            action=np.random.randint(0, 3, size=N_PARAMS),
            log_prob=-1.5,
            reward=-0.5,
            done=float(i == 9),
            value=0.1 * i,
        )

    buf.compute_gae(last_value=0.5, gamma=0.99, gae_lambda=0.95)
    assert buf.advantages[:10].shape == (10,)
    assert buf.returns[:10].shape == (10,)
    assert np.all(np.isfinite(buf.advantages[:10]))
    assert np.all(np.isfinite(buf.returns[:10]))


def test_rollout_buffer_batches():
    buf = RolloutBuffer(n_steps=10, obs_dim=OBS_DIM, n_params=N_PARAMS)
    for i in range(10):
        buf.store(np.zeros(OBS_DIM, dtype=np.float32), np.zeros(N_PARAMS, dtype=np.int64), -1.0, -0.5, 0.0, 0.0)
    buf.compute_gae(0.0, 0.99, 0.95)

    batches = list(buf.get_batches(batch_size=4))
    total_samples = sum(b[0].shape[0] for b in batches)
    assert total_samples == 10  # 4 + 4 + 2
    assert batches[0][1].shape == (4, N_PARAMS)  # actions are 2D


def test_ppo_collect_and_update():
    """Integration test: 1 collect + 1 update with the real env."""
    env = CircuitEnv()
    config = {
        "ppo": {
            "learning_rate": 3e-4,
            "n_steps": 8,  # small for test speed
            "batch_size": 4,
            "n_epochs": 2,
            "gamma": 0.99,
            "total_timesteps": 8,
        }
    }
    agent = PPOAgent(env, config)

    episode_stats = agent.collect_rollouts()
    assert isinstance(episode_stats, list)

    loss_stats = agent.update()
    assert "policy_loss" in loss_stats
    assert "value_loss" in loss_stats
    assert "entropy" in loss_stats
    assert np.isfinite(loss_stats["policy_loss"])
    assert np.isfinite(loss_stats["value_loss"])
