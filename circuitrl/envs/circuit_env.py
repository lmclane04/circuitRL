import math
import os

import gymnasium as gym
import numpy as np
import yaml

from circuitrl.simulators.ngspice_runner import NGSpiceRunner

# ── CS-amp physics constants (from cs_amp_template.sp) ────────────────────────
# Used only when specs_range is configured for feasibility-checking.
_LAMBDA  = 0.04      # V⁻¹  (channel-length modulation)
_CL      = 0.5e-12   # F    (output load cap)
_VGS_VT  = 0.2       # V    (VG - VTO = 0.7 - 0.5)


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
        self._default_targets = np.array(
            [float(cfg["target_specs"][m]["value"]) for m in self._metric_names]
        )
        self._targets = self._default_targets.copy()
        self._tolerances = np.array(
            [float(cfg["target_specs"][m]["tolerance"]) for m in self._metric_names]
        )
        self._n_metrics = len(self._metric_names)

        # Optional per-episode target randomisation
        specs_range = cfg.get("specs_range", {})
        if specs_range:
            self._target_range_min = np.array(
                [float(specs_range[m]["min"]) for m in self._metric_names]
            )
            self._target_range_max = np.array(
                [float(specs_range[m]["max"]) for m in self._metric_names]
            )
            self._randomize_specs = True
        else:
            self._target_range_min = self._default_targets.copy()
            self._target_range_max = self._default_targets.copy()
            self._randomize_specs = False

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
        if self._randomize_specs:
            self._targets = self._sample_feasible_targets()
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

    def _sample_feasible_targets(self) -> np.ndarray:
        """Sample a random (gain_db, bandwidth) pair that is physically achievable.

        Feasibility condition from small-signal analysis (same as solve.py):
            ro = 1 / (λ·π·B·CL·G·VGS_VT) > Rout = 1 / (2π·B·CL)
        which simplifies to: λ·G·VGS_VT < 2, i.e. gain < ~48 dB for these constants.
        All targets in our range satisfy this, but we also need RD to land on the grid
        (1–50 kΩ), so we check ro > 0 and Rout > 0 as a quick sanity guard.
        """
        gain_idx = self._metric_names.index("gain_db")
        bw_idx   = self._metric_names.index("bandwidth")

        for _ in range(50):
            t = np.array([
                float(self.np_random.uniform(lo, hi))
                for lo, hi in zip(self._target_range_min, self._target_range_max)
            ])
            G    = 10 ** (t[gain_idx] / 20.0)
            B    = t[bw_idx]
            Rout = 1.0 / (2 * math.pi * B * _CL)
            ro   = 1.0 / (_LAMBDA * math.pi * B * _CL * G * _VGS_VT)
            if ro > Rout > 0:
                return t

        return self._default_targets.copy()  # fallback

    def _build_obs(self) -> np.ndarray:
        """Concatenate [normalized_params | normalized_metrics | norm_target_values].

        norm_target_values encodes the *current* targets (important when specs are
        randomised) as a [0, 1] value over the configured specs_range:
            norm = (target - range_min) / (range_max - range_min)
        When specs are fixed (no randomisation), this collapses to [1, 1] as before.
        """
        norm_params  = self._normalize_params()
        norm_metrics = self._normalize_metrics()
        denom = self._target_range_max - self._target_range_min
        # Avoid divide-by-zero when range_min == range_max (fixed target case)
        safe_denom = np.where(denom != 0, denom, 1.0)
        norm_target_values = ((self._targets - self._target_range_min) / safe_denom).astype(np.float32)
        return np.concatenate([norm_params, norm_metrics, norm_target_values])

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
        info["params"]   = dict(zip(self._param_names, params_si.tolist()))
        info["targets"]  = dict(zip(self._metric_names, self._targets.tolist()))
        return info


# Backward-compatible alias.
OpAmpEnv = CircuitEnv
