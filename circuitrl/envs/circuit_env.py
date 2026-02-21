import os

import gymnasium as gym
import numpy as np
import yaml

from circuitrl.simulators.ngspice_runner import NGSpiceRunner


class CircuitEnv(gym.Env):
    """Config-driven circuit sizing environment.

    State:  [normalized_params | normalized_metrics | normalized_targets]
    Action: MultiDiscrete([3] * n_params) — per param: 0=decrease, 1=no-op, 2=increase
    Reward: -mean(|metric_i - target_i| / target_i)
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path: str = "circuitrl/configs/opamp_default.yaml"):
        super().__init__()

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Parameter info
        self._param_names = list(cfg["parameters"].keys())
        self._param_bounds = np.array(
            [[float(p["min"]), float(p["max"])] for p in cfg["parameters"].values()]
        )
        self._param_defaults_norm = np.array([
            (float(p["default"]) - float(p["min"])) / (float(p["max"]) - float(p["min"]))
            for p in cfg["parameters"].values()
        ])
        self._n_params = len(self._param_names)

        # Target specs (cast to float — PyYAML may leave scientific notation as str)
        self._metric_names = list(cfg["target_specs"].keys())
        self._targets = np.array(
            [float(cfg["target_specs"][m]["value"]) for m in self._metric_names]
        )
        self._tolerances = np.array(
            [float(cfg["target_specs"][m]["tolerance"]) for m in self._metric_names]
        )
        self._n_metrics = len(self._metric_names)

        # Env settings
        env_cfg = cfg["env"]
        self._max_steps = env_cfg["max_steps"]
        self._step_size = env_cfg["action_step_size"]
        sim_timeout = env_cfg["sim_timeout"]

        # Simulator — netlist path is relative to config file location
        netlist_rel = cfg.get("netlist", "../envs/netlist_template.sp")
        config_dir = os.path.dirname(os.path.abspath(config_path))
        template_path = os.path.normpath(os.path.join(config_dir, netlist_rel))
        self._runner = NGSpiceRunner(
            template_path, timeout=sim_timeout,
            expected_metrics=tuple(self._metric_names),
        )

        # Spaces
        obs_dim = self._n_params + self._n_metrics + self._n_metrics
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete([3] * self._n_params)

        # State
        self._params_norm = None
        self._metrics = None
        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._params_norm = self._param_defaults_norm.copy()
        self._step_count = 0
        self._metrics = self._simulate()
        return self._build_obs(), self._build_info()

    def step(self, action):
        self._step_count += 1

        # Decode action: 0=decrease, 1=no-op, 2=increase (per param)
        deltas = (np.asarray(action) - 1).astype(np.float64) * self._step_size
        self._params_norm = np.clip(self._params_norm + deltas, 0.0, 1.0)

        # Simulate
        self._metrics = self._simulate()

        # Reward and termination
        if self._metrics is None:
            reward = -10.0
            terminated = False
            truncated = True
        else:
            reward = self._compute_reward()
            terminated = self._check_specs_met()
            truncated = self._step_count >= self._max_steps

        return self._build_obs(), reward, terminated, truncated, self._build_info()

    def _simulate(self) -> np.ndarray | None:
        """Run NGSpice and return metrics as an array, or None on failure."""
        params_si = self._denormalize_params()
        param_dict = {}
        for name, val in zip(self._param_names, params_si):
            param_dict[name] = f"{val:.6e}"

        result = self._runner.run(param_dict)
        if result is None:
            return None

        return np.array([result[m] for m in self._metric_names])

    def _denormalize_params(self) -> np.ndarray:
        """Convert normalized [0,1] params to SI values."""
        lo = self._param_bounds[:, 0]
        hi = self._param_bounds[:, 1]
        return lo + self._params_norm * (hi - lo)

    def _normalize_metrics(self) -> np.ndarray:
        """Normalize metrics by target values for the observation."""
        if self._metrics is None:
            return np.zeros(self._n_metrics, dtype=np.float32)
        return (self._metrics / np.where(self._targets != 0, self._targets, 1.0)).astype(np.float32)

    def _build_obs(self) -> np.ndarray:
        """Concatenate [normalized_params | normalized_metrics | normalized_targets]."""
        norm_metrics = self._normalize_metrics()
        norm_targets = np.ones(self._n_metrics, dtype=np.float32)  # targets / targets = 1
        return np.concatenate([
            self._params_norm.astype(np.float32),
            norm_metrics,
            norm_targets,
        ])

    def _compute_reward(self) -> float:
        """Dense reward: negative mean relative error across specs."""
        rel_errors = np.abs(self._metrics - self._targets) / np.abs(
            np.where(self._targets != 0, self._targets, 1.0)
        )
        return -float(np.mean(rel_errors))

    def _check_specs_met(self) -> bool:
        """Check if all metrics are within tolerance of targets."""
        if self._metrics is None:
            return False
        return bool(np.all(np.abs(self._metrics - self._targets) <= self._tolerances))

    def _build_info(self) -> dict:
        info = {"step": self._step_count}
        if self._metrics is not None:
            info["metrics"] = dict(zip(self._metric_names, self._metrics.tolist()))
        params_si = self._denormalize_params()
        info["params"] = dict(zip(self._param_names, params_si.tolist()))
        return info


# Backward-compatible alias.
OpAmpEnv = CircuitEnv
