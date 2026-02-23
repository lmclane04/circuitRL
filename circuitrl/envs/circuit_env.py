import os

import gymnasium as gym
import numpy as np
import yaml

from circuitrl.simulators.ngspice_runner import NGSpiceRunner


class CircuitEnv(gym.Env):
    """Config-driven circuit sizing environment (discrete index-based).

    Each parameter is discretized into a lookup array via np.arange(min, max, step)
    in SI units. The agent moves an integer index per parameter.

    State:  [normalized_params | normalized_metrics | normalized_targets]
    Action: MultiDiscrete([3] * n_params) — per param: 0=decrease, 1=no-op, 2=increase
    Reward: -mean(|metric_i - target_i| / target_i)
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path: str = "circuitrl/configs/opamp_default.yaml"):
        super().__init__()

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Build discrete lookup arrays per parameter
        self._param_names = list(cfg["parameters"].keys())
        self._n_params = len(self._param_names)
        self._param_arrays = []
        for p in cfg["parameters"].values():
            lo, hi, step = float(p["min"]), float(p["max"]), float(p["step"])
            arr = np.arange(lo, hi + step * 0.5, step)
            self._param_arrays.append(arr)

        self._max_indices = np.array([len(a) - 1 for a in self._param_arrays])

        # Default index: closest array entry to the configured default
        self._default_indices = np.array([
            int(np.argmin(np.abs(arr - float(p["default"]))))
            for arr, p in zip(self._param_arrays, cfg["parameters"].values())
        ])

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
        sim_timeout = env_cfg["sim_timeout"]

        # Optional parameter constraints (evaluated on SI values)
        self._constraints = cfg.get("constraints", [])

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
        self._param_indices = None
        self._metrics = None
        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._param_indices = self._default_indices.copy()
        self._step_count = 0
        self._metrics = self._simulate()
        return self._build_obs(), self._build_info()

    def step(self, action):
        self._step_count += 1

        # Decode action: 0=decrease, 1=no-op, 2=increase (per param)
        self._prev_indices = self._param_indices.copy()
        deltas = np.asarray(action) - 1
        self._param_indices = np.clip(self._param_indices + deltas, 0, self._max_indices)

        # Revert if constraints violated
        if self._constraints:
            self._enforce_constraints()

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
        params_si = self._get_params_si()
        param_dict = {}
        for name, val in zip(self._param_names, params_si):
            param_dict[name] = f"{val:.6e}"

        result = self._runner.run(param_dict)
        if result is None:
            return None

        return np.array([result[m] for m in self._metric_names])

    def _get_params_si(self) -> np.ndarray:
        """Look up SI values from current indices."""
        return np.array([arr[idx] for arr, idx in zip(self._param_arrays, self._param_indices)])

    def _enforce_constraints(self):
        """Check constraints on SI values; revert move if any violated."""
        params_si = self._get_params_si()
        local_vars = dict(zip(self._param_names, params_si.tolist()))
        for expr in self._constraints:
            if not eval(expr, {"__builtins__": {}}, local_vars):
                # Constraint violated — revert entire move
                self._param_indices = self._prev_indices.copy()
                return

    def _normalize_params(self) -> np.ndarray:
        """Normalize params as index / max_index → [0, 1]."""
        return (self._param_indices / self._max_indices).astype(np.float32)

    def _normalize_metrics(self) -> np.ndarray:
        """Normalize metrics by target values for the observation."""
        if self._metrics is None:
            return np.zeros(self._n_metrics, dtype=np.float32)
        return (self._metrics / np.where(self._targets != 0, self._targets, 1.0)).astype(np.float32)

    def _build_obs(self) -> np.ndarray:
        """Concatenate [normalized_params | normalized_metrics | normalized_targets]."""
        norm_params = self._normalize_params()
        norm_metrics = self._normalize_metrics()
        norm_targets = np.ones(self._n_metrics, dtype=np.float32)  # targets / targets = 1
        return np.concatenate([norm_params, norm_metrics, norm_targets])

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
        params_si = self._get_params_si()
        info["params"] = dict(zip(self._param_names, params_si.tolist()))
        return info


# Backward-compatible alias.
OpAmpEnv = CircuitEnv
