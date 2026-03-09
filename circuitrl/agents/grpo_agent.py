"""Group Relative Policy Optimisation (GRPO) agent for circuit sizing.

Based on DeepSeek-R1 / GRPO (Shao et al., 2024).

Key differences from PPO:
  - No value network / critic. Advantage is computed purely from group-relative
    episode returns, eliminating value function bias.
  - Per update: sample G full episodes from the *same* starting state. Normalise
    their returns to get per-episode advantages:
        A_i = (R_i - mean_G) / (std_G + ε)
  - Apply that scalar advantage uniformly to every step of episode i, using
    a PPO-style clipped surrogate objective.
  - Because the group shares a start, A_i measures how much better/worse
    episode i was than the group average — a clean, critic-free baseline.

Architecture: reuses the same ActorCritic trunk as PPO but ignores the value
head during training (it is still available for evaluation/logging if desired).
"""

import numpy as np
import torch
import torch.nn as nn

from circuitrl.agents.ppo_agent import ActorCritic


class GRPOAgent:
    def __init__(self, env, config: dict):
        self.env = env

        obs_dim          = env.observation_space.shape[0]
        n_params         = env.action_space.shape[0]
        actions_per_param = int(env.action_space.nvec[0])

        grpo_cfg = config.get("grpo", {})
        self.lr             = float(grpo_cfg.get("learning_rate", 3e-4))
        self.group_size     = int(grpo_cfg.get("group_size", 8))
        self.clip_eps       = float(grpo_cfg.get("clip_eps", 0.2))
        self.n_epochs       = int(grpo_cfg.get("n_epochs", 1))
        self.ent_coef       = float(grpo_cfg.get("ent_coef", 0.05))
        self.max_grad_norm  = float(grpo_cfg.get("max_grad_norm", 0.5))
        self.total_timesteps = int(grpo_cfg.get("total_timesteps", 50000))

        self.network   = ActorCritic(obs_dim, n_params, actions_per_param)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    # ── rollout ───────────────────────────────────────────────────────────────

    def _random_start_indices(self) -> np.ndarray:
        """Sample random parameter indices within valid ranges."""
        return np.array([
            np.random.randint(0, m + 1)
            for m in self.env._max_indices
        ])

    def _rollout(self, start_indices: np.ndarray,
                 start_targets: np.ndarray) -> tuple[list, float, dict]:
        """Run one episode from start_indices with fixed start_targets.

        All rollouts in a group share the same target so that their returns
        are on the same scale and group-relative advantages are meaningful.

        Returns:
            trajectory: list of (obs, action_np, old_log_prob)
            ep_return:  scalar cumulative reward
            metrics:    final info["metrics"] dict
        """
        # Reset env state without re-randomizing (we set our own start)
        self.env._step_count = 0
        self.env._active_param_idx = 0
        self.env._param_indices = start_indices.copy()
        self.env._targets = start_targets.copy()
        self.env._metrics = self.env._simulate()
        obs = self.env._build_obs()

        trajectory: list[tuple] = []
        ep_return = 0.0

        for _ in range(self.env._max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, _ = self.network.get_action(obs_t)

            action_np = action.squeeze(0).numpy()
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)

            trajectory.append((obs.copy(), action_np.copy(), log_prob.item()))
            ep_return += reward
            obs = next_obs

            if terminated or truncated:
                break

        return trajectory, ep_return, info.get("metrics", {})

    # ── update ────────────────────────────────────────────────────────────────

    def _update(self, group: list[tuple]) -> dict:
        """GRPO policy update on one group of G episodes.

        Args:
            group: list of (trajectory, ep_return, metrics)

        Returns:
            loss_stats dict
        """
        returns = torch.tensor([r for _, r, _ in group], dtype=torch.float32)

        # Group-relative advantage normalisation (the core of GRPO)
        if returns.std() > 1e-6:
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            advantages = torch.zeros(len(group))

        # Flatten all steps across the group, tagging each with its episode advantage
        all_obs, all_acts, all_old_lp, all_adv = [], [], [], []
        for (traj, _, _), adv in zip(group, advantages):
            for obs, act, old_lp in traj:
                all_obs.append(obs)
                all_acts.append(act)
                all_old_lp.append(old_lp)
                all_adv.append(adv.item())

        obs_t    = torch.tensor(np.array(all_obs),    dtype=torch.float32)
        act_t    = torch.tensor(np.array(all_acts),   dtype=torch.int64)
        old_lp_t = torch.tensor(all_old_lp,           dtype=torch.float32)
        adv_t    = torch.tensor(all_adv,              dtype=torch.float32)

        total_policy_loss = 0.0
        total_entropy     = 0.0

        for _ in range(self.n_epochs):
            new_lp, entropy, _ = self.network.evaluate(obs_t, act_t)

            ratio = torch.exp(new_lp - old_lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t

            policy_loss = -torch.min(surr1, surr2).mean()
            loss        = policy_loss - self.ent_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_entropy     += entropy.mean().item()

        return {
            "policy_loss": total_policy_loss / self.n_epochs,
            "value_loss":  0.0,
            "entropy":     total_entropy     / self.n_epochs,
        }

    # ── public API ────────────────────────────────────────────────────────────

    def train(self, total_timesteps: int | None = None, callback=None):
        total          = total_timesteps or self.total_timesteps
        timesteps_done = 0

        while timesteps_done < total:
            # All G episodes in this group share the same starting state AND targets.
            # We call reset() only to sample fresh targets, then override param indices
            # with a random start so the agent sees diverse starting points.
            self.env.reset()
            start_indices = self._random_start_indices()
            start_targets = self.env._targets.copy()  # snapshot targets for this group

            group: list[tuple] = []
            for _ in range(self.group_size):
                traj, ep_return, metrics = self._rollout(start_indices, start_targets)
                group.append((traj, ep_return, metrics))
                timesteps_done += len(traj)

            loss_stats = self._update(group)

            if callback:
                ep_targets_dict = dict(zip(self.env._metric_names, start_targets.tolist()))
                ep_stats = [
                    {
                        "reward":        r,
                        "length":        len(t),
                        "final_metrics": m,
                        "ep_targets":    ep_targets_dict,
                    }
                    for t, r, m in group
                ]
                callback(timesteps_done, ep_stats, loss_stats)

    def evaluate(self, n_episodes: int = 20) -> list[dict]:
        """Run n_episodes using the current policy and return episode stats."""
        results = []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            ep_reward = 0.0
            ep_len    = 0
            for _ in range(self.env._max_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = self.network.get_action(obs_t)
                action_np = action.squeeze(0).numpy()
                obs, reward, terminated, truncated, info = self.env.step(action_np)
                ep_reward += reward
                ep_len    += 1
                if terminated or truncated:
                    break
            results.append({
                "reward":        ep_reward,
                "length":        ep_len,
                "final_metrics": info.get("metrics", {}),
                "ep_targets":    info.get("targets", {}),
            })
        return results

    def save(self, path: str):
        torch.save({
            "network":   self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=True)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
