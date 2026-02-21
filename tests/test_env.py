"""Validate Gym environment (reset/step outputs, action space shape, etc)"""

import numpy as np

from circuitrl.envs.circuit_env import CircuitEnv


def test_reset():
    env = CircuitEnv()
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert "metrics" in info
    assert "params" in info


def test_step():
    env = CircuitEnv()
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_action_space_size():
    env = CircuitEnv()
    # MultiDiscrete: 10 params, each with 3 choices (decrease/no-op/increase)
    assert env.action_space.shape == (10,)
    assert all(n == 3 for n in env.action_space.nvec)


def test_reward_negative():
    """Reward should be non-positive (negative distance to target)."""
    env = CircuitEnv()
    env.reset()
    _, reward, _, _, _ = env.step(env.action_space.sample())
    assert reward <= 0.0, f"Expected non-positive reward, got {reward}"
