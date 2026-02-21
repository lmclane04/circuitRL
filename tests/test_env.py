"""Tests for OpAmpEnv Gymnasium environment."""

import numpy as np

from circuitrl.envs.opamp_env import OpAmpEnv


def test_reset():
    env = OpAmpEnv()
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert "metrics" in info
    assert "params" in info


def test_step():
    env = OpAmpEnv()
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_action_space_size():
    env = OpAmpEnv()
    # 10 params × 2 directions (up/down)
    assert env.action_space.n == 20


def test_reward_negative():
    """Reward should be non-positive (negative distance to target)."""
    env = OpAmpEnv()
    env.reset()
    _, reward, _, _, _ = env.step(env.action_space.sample())
    assert reward <= 0.0, f"Expected non-positive reward, got {reward}"
