import numpy as np
import torch

from circuitrl.agents.ppo_agent import ActorCritic, PPOAgent, RolloutBuffer
from circuitrl.envs.opamp_env import OpAmpEnv


def test_actor_critic_forward():
    obs_dim, n_actions = 18, 20
    net = ActorCritic(obs_dim, n_actions)
    obs = torch.randn(4, obs_dim)

    logits, values = net(obs)
    assert logits.shape == (4, n_actions)
    assert values.shape == (4,)


def test_actor_critic_get_action():
    obs_dim, n_actions = 18, 20
    net = ActorCritic(obs_dim, n_actions)
    obs = torch.randn(1, obs_dim)

    action, log_prob, value = net.get_action(obs)
    assert action.shape == (1,)
    assert 0 <= action.item() < n_actions
    assert log_prob.shape == (1,)
    assert value.shape == (1,)


def test_actor_critic_evaluate():
    obs_dim, n_actions = 18, 20
    net = ActorCritic(obs_dim, n_actions)
    obs = torch.randn(8, obs_dim)
    actions = torch.randint(0, n_actions, (8,))

    log_probs, entropy, values = net.evaluate(obs, actions)
    assert log_probs.shape == (8,)
    assert entropy.shape == (8,)
    assert values.shape == (8,)
    assert torch.all(log_probs <= 0)  # log probs are non-positive


def test_rollout_buffer_gae():
    obs_dim = 18
    buf = RolloutBuffer(n_steps=10, obs_dim=obs_dim)

    for i in range(10):
        buf.store(
            obs=np.random.randn(obs_dim).astype(np.float32),
            action=np.random.randint(0, 20),
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
    obs_dim = 18
    buf = RolloutBuffer(n_steps=10, obs_dim=obs_dim)
    for i in range(10):
        buf.store(np.zeros(obs_dim, dtype=np.float32), 0, -1.0, -0.5, 0.0, 0.0)
    buf.compute_gae(0.0, 0.99, 0.95)

    batches = list(buf.get_batches(batch_size=4))
    total_samples = sum(b[0].shape[0] for b in batches)
    assert total_samples == 10  # 4 + 4 + 2


def test_ppo_collect_and_update():
    """Integration test: 1 collect + 1 update with the real env."""
    env = OpAmpEnv()
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
