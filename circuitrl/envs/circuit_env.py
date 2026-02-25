import json
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
              metrics:  (m - g) / (m + g)        → 0.0 when on-spec, bounded [-1, 1]
              targets:  (target - pool_min) / (pool_max - pool_min) → [0, 1]
    Action: MultiDiscrete([3] * n_params) —> per param: 0=decrease, 1=no-op, 2=increase
    Reward: asymmetric — only penalize specs that are violated (see _compute_reward)
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path: str = "circuitrl/configs/opamp.yaml"):
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
        self._default_indices = np.array([
            int(np.argmin(np.abs(arr - float(p["default"]))))
            for arr, p in zip(self._param_arrays, cfg["parameters"].values())
        ])

        self._metric_names = list(cfg["target_specs"].keys())
        self._n_metrics = len(self._metric_names)
        self._tolerances = np.array(
            [float(cfg["target_specs"][m]["tolerance"]) for m in self._metric_names]
        )
        self._spec_directions = [
            cfg["target_specs"][m].get("direction", "equal")
            for m in self._metric_names
        ]
        self._targets = np.zeros(self._n_metrics)  # re-sampled each reset()

        config_dir = os.path.dirname(os.path.abspath(config_path))

        # Load spec pool (created by generate_specs.py).
        # Training targets are sampled from this pool, so all targets are
        # guaranteed to be physically achievable by the circuit.
        spec_file = cfg["target_spec_file"]
        pool_path = os.path.normpath(os.path.join(config_dir, spec_file))
        with open(pool_path) as f:
            raw = json.load(f)
        self._spec_pool = np.array([
            [float(entry[m]) for m in self._metric_names]
            for entry in raw
        ])
        # Per-metric min/max from the pool — used to normalize the target
        # component of the observation to [0, 1] without any hand-tuned nominal.
        self._pool_min = self._spec_pool.min(axis=0)
        self._pool_max = self._spec_pool.max(axis=0)
        print(f"Loaded {len(self._spec_pool)} spec targets from {pool_path}")

        # Env settings
        self._max_steps = cfg["env"]["max_steps"]

        # Simulator
        netlist_rel = cfg.get("netlist", "../envs/netlist_template.sp")
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
        deltas = np.asarray(action) - 1
        self._param_indices = np.clip(self._param_indices + deltas, 0, self._max_indices)

        self._metrics = self._simulate()

        # Reward and termination
        if self._metrics is None:
            reward = -2.0
            terminated = False
            truncated = True
        else:
            terminated = self._check_specs_met()
            reward = 10.0 if terminated else self._compute_reward()
            truncated = self._step_count >= self._max_steps

        return self._build_obs(), reward, terminated, truncated, self._build_info()

    def _sample_targets(self) -> np.ndarray:
        """Pick a random entry from the pre-generated spec pool."""
        idx = self.np_random.integers(0, len(self._spec_pool))
        return self._spec_pool[idx].copy()

    def _simulate(self) -> np.ndarray | None:
        """Run NGSpice and return metrics as an array, or None on failure."""
        params_si = self._get_params_si()
        param_dict = {name: f"{val:.6e}" for name, val in zip(self._param_names, params_si)}
        result = self._runner.run(param_dict)
        if result is None:
            return None
        return np.array([result[m] for m in self._metric_names])

    def _get_params_si(self) -> np.ndarray:
        """Look up SI values from current indices."""
        return np.array([arr[idx] for arr, idx in zip(self._param_arrays, self._param_indices)])

    def _normalize_params(self) -> np.ndarray:
        """Normalize params as index / max_index → [0, 1]."""
        return (self._param_indices / self._max_indices).astype(np.float32)

    def _normalize_metrics(self) -> np.ndarray:
        """Normalize metrics relative to target: (m - g) / (m + g) → [-1, 1].
        Zero when on-spec, negative when below target, positive when above."""
        if self._metrics is None:
            return np.zeros(self._n_metrics, dtype=np.float32)
        denom = np.where(self._metrics + self._targets != 0, self._metrics + self._targets, 1.0)
        return ((self._metrics - self._targets) / denom).astype(np.float32)

    def _build_obs(self) -> np.ndarray:
        """Concatenate [normalized_params | normalized_metrics | normalized_targets].

        norm_targets: (target - pool_min) / (pool_max - pool_min) → [0, 1]
          0 = easiest target in pool, 1 = hardest target in pool.
          Uses pool statistics so no hand-tuned nominal needed.
        """
        pool_range = np.where(self._pool_max != self._pool_min,
                              self._pool_max - self._pool_min, 1.0)
        norm_targets = ((self._targets - self._pool_min) / pool_range).astype(np.float32)
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
        info: dict = {"step": self._step_count}
        if self._metrics is not None:
            info["metrics"] = dict(zip(self._metric_names, self._metrics.tolist()))
        params_si = self._get_params_si()
        info["params"] = dict(zip(self._param_names, params_si.tolist()))
        info["targets"] = dict(zip(self._metric_names, self._targets.tolist()))
        return info
