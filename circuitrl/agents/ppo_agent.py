import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic with independent per-parameter policy heads."""

    def __init__(self, obs_dim: int, n_params: int, actions_per_param: int = 3, hidden: int = 256):
        super().__init__()
        self.n_params = n_params
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        # One head per parameter, each outputs logits for {decrease, no-op, increase}
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden, actions_per_param) for _ in range(n_params)
        ])
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        logits = [head(h) for head in self.policy_heads]  # list of (batch, 3)
        return logits, self.value_head(h).squeeze(-1)

    def get_action(self, obs: torch.Tensor):
        """Sample an action per parameter. Returns (actions, summed_log_prob, value)."""
        logits_list, value = self(obs)
        actions = []
        log_prob_sum = torch.zeros(obs.shape[0], device=obs.device)
        for logits in logits_list:
            dist = Categorical(logits=logits)
            a = dist.sample()
            log_prob_sum = log_prob_sum + dist.log_prob(a)
            actions.append(a)
        actions = torch.stack(actions, dim=-1)  # (batch, n_params)
        return actions, log_prob_sum, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate given actions. Returns (summed_log_prob, mean_entropy, value)."""
        logits_list, value = self(obs)
        log_prob_sum = torch.zeros(obs.shape[0], device=obs.device)
        entropy_sum = torch.zeros(obs.shape[0], device=obs.device)
        for i, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            log_prob_sum = log_prob_sum + dist.log_prob(actions[:, i])
            entropy_sum = entropy_sum + dist.entropy()
        mean_entropy = entropy_sum / self.n_params
        return log_prob_sum, mean_entropy, value


class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int, n_params: int):
        self.n_steps = n_steps
        self.obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_params), dtype=np.int64)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)
        self.ptr = 0

    def store(self, obs, action, log_prob, reward, done, value):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        last_adv = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv

        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get_batches(self, batch_size: int):
        """Sample random mini-batches"""
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)
        for start in range(0, self.ptr, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield (
                torch.tensor(self.obs[batch_idx]),
                torch.tensor(self.actions[batch_idx]),
                torch.tensor(self.log_probs[batch_idx]),
                torch.tensor(self.returns[batch_idx]),
                torch.tensor(self.advantages[batch_idx]),
            )

    def reset(self):
        self.ptr = 0


class PPOAgent:
    def __init__(self, env, config: dict):
        self.env = env
        obs_dim = env.observation_space.shape[0]
        n_params = env.action_space.shape[0]  # MultiDiscrete → .shape[0] = n_params

        ppo_cfg = config["ppo"]
        self.lr = float(ppo_cfg["learning_rate"])
        self.n_steps = int(ppo_cfg["n_steps"])
        self.batch_size = int(ppo_cfg["batch_size"])
        self.n_epochs = int(ppo_cfg["n_epochs"])
        self.gamma = float(ppo_cfg["gamma"])
        self.total_timesteps = int(ppo_cfg["total_timesteps"])

        # PPO-specific (fixed, not in config right now but we could add them)
        self.clip_eps = 0.2
        self.vf_coef = 0.5
        self.ent_coef = 0.01
        self.gae_lambda = 0.95
        self.max_grad_norm = 0.5

        self.network = ActorCritic(obs_dim, n_params)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.buffer = RolloutBuffer(self.n_steps, obs_dim, n_params)

    def collect_rollouts(self) -> list[dict]:
        """Run environment for n_steps, fill self.buffer. Return list of episode stats."""
        self.buffer.reset()
        obs, _ = self.env.reset()
        episode_stats = []
        ep_reward = 0.0
        ep_len = 0

        for _ in range(self.n_steps):
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action, log_prob, value = self.network.get_action(obs_t)

            action_np = action.squeeze(0).numpy()  # (n_params,)
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            self.buffer.store(obs, action_np, log_prob.item(), reward, float(done), value.item())

            ep_reward += reward
            ep_len += 1

            if done:
                episode_stats.append({"reward": ep_reward, "length": ep_len})
                ep_reward = 0.0
                ep_len = 0
                obs, _ = self.env.reset()
            else:
                obs = next_obs

        # Bootstrap value for last state
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            _, _, last_value = self.network.get_action(obs_t)

        self.buffer.compute_gae(last_value.item(), self.gamma, self.gae_lambda)
        return episode_stats

    def update(self) -> dict:
        """Run clipped PPO update. Returns loss."""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for obs_b, act_b, old_lp_b, ret_b, adv_b in self.buffer.get_batches(self.batch_size):
                # Normalize advantages
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                new_lp, entropy, values = self.network.evaluate(obs_b, act_b)

                # Policy loss (clipped)
                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, ret_b)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def train(self, total_timesteps: int | None = None, callback=None):
        total = total_timesteps or self.total_timesteps
        timesteps_done = 0
        iteration = 0

        while timesteps_done < total:
            episode_stats = self.collect_rollouts()
            loss_stats = self.update()
            timesteps_done += self.n_steps
            iteration += 1

            if callback:
                callback(timesteps_done, episode_stats, loss_stats)

    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, weights_only=True)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
