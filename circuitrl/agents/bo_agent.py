"""Bayesian Optimisation agent for circuit sizing.

Replaces the policy network with a Gaussian Process surrogate + UCB acquisition.
No gradient steps — each "timestep" is one simulator call selected by BO.

Key differences from PPO:
  - total_timesteps means number of simulator evaluations (not env steps)
  - No neural network; save/load persists the GP observations as a pickle
  - Phase 1: n_initial random evaluations to seed the GP
  - Phase 2: UCB acquisition over n_candidates random valid points per step

Callback interface is identical to PPOAgent:
  callback(timesteps_done, episode_stats, loss_stats)
  loss_stats will be all zeros (no gradient updates).
"""

import pickle
import warnings

import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.exceptions import ConvergenceWarning
except ImportError as e:
    raise ImportError("BOAgent requires scikit-learn: pip install scikit-learn") from e


class BOAgent:
    def __init__(self, env, config: dict):
        self.env = env

        bo_cfg = config.get("bo", {})
        self.n_initial      = int(bo_cfg.get("n_initial", 20))
        self.kappa          = float(bo_cfg.get("kappa", 2.576))   # ~99% UCB
        self.n_candidates   = int(bo_cfg.get("n_candidates", 1000))
        self.gp_update_freq = int(bo_cfg.get("gp_update_freq", 5))
        self.total_timesteps = int(bo_cfg.get("total_timesteps", 300))

        # Matern-5/2 is the standard GP kernel for Bayesian optimisation
        # Very wide length_scale_bounds to cover smooth and spiky reward landscapes
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5, length_scale_bounds=(1e-5, 1e3)),
            alpha=1e-6,
            n_restarts_optimizer=3,
            normalize_y=True,
        )

        self.X_obs: list[np.ndarray] = []   # normalized param vectors
        self.y_obs: list[float]      = []   # observed rewards
        self.best_reward  = -np.inf
        self.best_indices = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _normalize(self, indices: np.ndarray) -> np.ndarray:
        """Map integer indices → [0, 1]^n for the GP."""
        return (indices / self.env._max_indices).astype(np.float64)

    def _sample_valid(self) -> np.ndarray:
        """Uniform random parameter indices that satisfy all constraints."""
        for _ in range(100):
            idx = np.array([
                int(np.random.randint(0, m + 1))
                for m in self.env._max_indices
            ])
            if not self.env._constraints:
                return idx
            si = np.array([a[i] for a, i in zip(self.env._param_arrays, idx)])
            lv = dict(zip(self.env._param_names, si.tolist()))
            if all(eval(e, {"__builtins__": {}}, lv) for e in self.env._constraints):
                return idx
        return self.env._default_indices.copy()

    def _evaluate(self, indices: np.ndarray) -> tuple[float, dict]:
        """Directly set env params, simulate, return (reward, metrics)."""
        self.env._param_indices = indices.copy()
        self.env._metrics = self.env._simulate()
        if self.env._metrics is None:
            return -10.0, {}
        reward  = self.env._compute_reward()
        metrics = dict(zip(self.env._metric_names, self.env._metrics.tolist()))
        return reward, metrics

    def _ucb(self, X_cand: np.ndarray) -> np.ndarray:
        mu, sigma = self.gp.predict(X_cand, return_std=True)
        return mu + self.kappa * sigma

    def _next_point(self) -> np.ndarray:
        """Sample n_candidates valid points, return the one maximising UCB."""
        candidates = np.array([self._sample_valid() for _ in range(self.n_candidates)])
        X_cand = np.stack([self._normalize(c) for c in candidates])
        acq    = self._ucb(X_cand)
        return candidates[int(np.argmax(acq))]

    def _fit_gp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit(np.array(self.X_obs), np.array(self.y_obs))

    # ── public API ────────────────────────────────────────────────────────────

    def train(self, total_timesteps: int | None = None, callback=None):
        total  = total_timesteps or self.total_timesteps
        n_done = 0

        _dummy_loss = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        # BO uses fixed targets throughout (it bypasses env.reset)
        _ep_targets = dict(zip(self.env._metric_names, self.env._targets.tolist()))

        # ── Phase 1: random initial design ───────────────────────────────────
        for _ in range(min(self.n_initial, total)):
            idx           = self._sample_valid()
            reward, metrics = self._evaluate(idx)

            self.X_obs.append(self._normalize(idx))
            self.y_obs.append(reward)

            if reward > self.best_reward:
                self.best_reward, self.best_indices = reward, idx.copy()

            n_done += 1
            if callback:
                callback(n_done,
                         [{"reward": reward, "length": 1,
                           "final_metrics": metrics, "ep_targets": _ep_targets}],
                         _dummy_loss)

        self._fit_gp()

        # ── Phase 2: BO acquisition loop ─────────────────────────────────────
        while n_done < total:
            idx             = self._next_point()
            reward, metrics = self._evaluate(idx)

            self.X_obs.append(self._normalize(idx))
            self.y_obs.append(reward)

            if reward > self.best_reward:
                self.best_reward, self.best_indices = reward, idx.copy()

            n_done += 1

            if n_done % self.gp_update_freq == 0:
                self._fit_gp()

            if callback:
                callback(n_done,
                         [{"reward": reward, "length": 1,
                           "final_metrics": metrics, "ep_targets": _ep_targets}],
                         _dummy_loss)

    def evaluate(self, n_episodes: int = 20) -> list[dict]:
        """Evaluate the best-known parameter point found during training.

        BO has no stochastic policy, so we re-run its best-known indices once
        and return a list of length n_episodes with the same result (std = 0).
        """
        if self.best_indices is None:
            return []
        reward, metrics = self._evaluate(self.best_indices)
        ep_targets = dict(zip(self.env._metric_names, self.env._targets.tolist()))
        ep = {
            "reward":        reward,
            "length":        1,
            "final_metrics": metrics,
            "ep_targets":    ep_targets,
        }
        return [ep] * n_episodes

    def save(self, path: str):
        pkl_path = path.replace(".pt", ".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({
                "X_obs":        self.X_obs,
                "y_obs":        self.y_obs,
                "best_reward":  self.best_reward,
                "best_indices": self.best_indices,
            }, f)
        print(f"  (BO checkpoint saved to {pkl_path})")

    def load(self, path: str):
        pkl_path = path.replace(".pt", ".pkl")
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        self.X_obs        = d["X_obs"]
        self.y_obs        = d["y_obs"]
        self.best_reward  = d["best_reward"]
        self.best_indices = d["best_indices"]
        if self.X_obs:
            self._fit_gp()
