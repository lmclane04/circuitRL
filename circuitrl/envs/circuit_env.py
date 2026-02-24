import os

import gymnasium as gym
import numpy as np
import yaml

from circuitrl.simulators.ngspice_runner import NGSpiceRunner


class CircuitEnv(gym.Env):
    """Circuit sizing environment driven by a YAML config.

    Each circuit parameter is discretized into a lookup array of SI values via
    np.arange(min, max, step). The agent selects {decrease, no-op, increase} for
    every parameter simultaneously each step, moving integer indices through the grid.

    State:  [normalized_params | normalized_metrics | normalized_targets]
              params:   index / max_index        → [0, 1]
              metrics:  metric / target          → ~1.0 when on-spec
              targets:  target / nominal_target  → 1.0 at nominal, varies when randomized
    Action: MultiDiscrete([3] * n_params) —> per param: 0=decrease, 1=no-op, 2=increase
    Reward: asymmetric — only penalize specs that are violated (see _compute_reward)
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
            self._param_arrays.append(np.arange(lo, hi + step * 0.5, step))

        self._max_indices = np.array([len(a) - 1 for a in self._param_arrays])
        # Starting index per parameter: closest grid point to the configured default
        self._default_indices = np.array([
            int(np.argmin(np.abs(arr - float(p["default"]))))
            for arr, p in zip(self._param_arrays, cfg["parameters"].values())
        ])

        self._metric_names = list(cfg["target_specs"].keys())
        self._n_metrics = len(self._metric_names)
        self._target_nominals = np.array(
            [float(cfg["target_specs"][m]["value"]) for m in self._metric_names]
        )
        self._tolerances = np.array(
            [float(cfg["target_specs"][m]["tolerance"]) for m in self._metric_names]
        )
        self._spec_directions = [
            cfg["target_specs"][m].get("direction", "equal")
            for m in self._metric_names
        ]
        self._target_ranges = [
            (float(s["range"][0]), float(s["range"][1])) if "range" in (s := cfg["target_specs"][m]) else None
            for m in self._metric_names
        ]
        self._targets = self._target_nominals.copy()  # re-sampled each reset()

        # Env settings
        self._max_steps = cfg["env"]["max_steps"]
        self._constraints = cfg.get("constraints", [])

        # Simulator 
        netlist_rel = cfg.get("netlist", "../envs/netlist_template.sp")
        config_dir = os.path.dirname(os.path.abspath(config_path))
        self._runner = NGSpiceRunner(
            os.path.normpath(os.path.join(config_dir, netlist_rel)),
            timeout=cfg["env"]["sim_timeout"],
            expected_metrics=tuple(self._metric_names),
        )

        # Gym spaces 
        obs_dim = self._n_params + self._n_metrics + self._n_metrics
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete([3] * self._n_params)
        self._param_indices = None
        self._metrics = None
        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._param_indices = self._default_indices.copy()
        self._targets = self._sample_targets()
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

        self._metrics = self._simulate()

        # Reward and termination
        if self._metrics is None:
            reward = -2.0   # penalize simulation failure
            terminated = False
            truncated = True
        else:
            reward = self._compute_reward()
            terminated = self._check_specs_met()
            truncated = self._step_count >= self._max_steps

        return self._build_obs(), reward, terminated, truncated, self._build_info()

    def _sample_targets(self) -> np.ndarray:
        """Sample target values from configured ranges, or return nominals."""
        targets = self._target_nominals.copy()
        for i, r in enumerate(self._target_ranges):
            if r is not None:
                targets[i] = float(self.np_random.uniform(r[0], r[1]))
        return targets

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
                self._param_indices = self._prev_indices.copy()
                return

    def _normalize_params(self) -> np.ndarray:
        """Normalize params as index / max_index → [0, 1]."""
        return (self._param_indices / self._max_indices).astype(np.float32)

    def _normalize_metrics(self) -> np.ndarray:
        """Normalize metrics by current target values for the observation."""
        if self._metrics is None:
            return np.zeros(self._n_metrics, dtype=np.float32)
        return (self._metrics / np.where(self._targets != 0, self._targets, 1.0)).astype(np.float32)

    def _build_obs(self) -> np.ndarray:
        """Concatenate [normalized_params | normalized_metrics | normalized_targets]."""
        norm_targets = (self._targets / np.where(self._target_nominals != 0, self._target_nominals, 1.0)).astype(np.float32)
        return np.concatenate([self._normalize_params(), self._normalize_metrics(), norm_targets])

    def _compute_reward(self) -> float:
        """Asymmetric reward: only penalize specs that are violated.

        direction='max': penalize if metric < target (e.g. gain, bandwidth)
        direction='min': penalize if metric > target (e.g. power)
        direction='equal': symmetric penalty (default)
        """
        penalties = []
        for m, t, d in zip(self._metrics, self._targets, self._spec_directions):
            denom = abs(t) if t != 0 else 1.0
            if d == 'max':
                penalty = max(0.0, (t - m) / denom)
            elif d == 'min':
                penalty = max(0.0, (m - t) / denom)
            else:
                penalty = abs(m - t) / denom
            penalties.append(penalty)
        return -float(np.mean(penalties))

    def _check_specs_met(self) -> bool:
        """Check if all specs are satisfied, accounting for direction."""
        if self._metrics is None:
            return False
        for m, t, tol, d in zip(self._metrics, self._targets, self._tolerances, self._spec_directions):
            if d == 'max':
                if m < t - tol:
                    return False
            elif d == 'min':
                if m > t + tol:
                    return False
            else:
                if abs(m - t) > tol:
                    return False
        return True

    def _build_info(self) -> dict:
        info = {"step": self._step_count}
        if self._metrics is not None:
            info["metrics"] = dict(zip(self._metric_names, self._metrics.tolist()))
        params_si = self._get_params_si()
        info["params"] = dict(zip(self._param_names, params_si.tolist()))
        info["targets"] = dict(zip(self._metric_names, self._targets.tolist()))
        return info

