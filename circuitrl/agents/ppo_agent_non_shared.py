import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    """Actor model with per-parameter policy heads."""

    def __init__(self, obs_dim: int, n_params: int, actions_per_param: int = 3, hidden_size: int = 256):
        """Initialize Actor model, setup trunk and all policy heads."""
        super().__init__()

        # Set up number of parameters
        self.n_params = n_params

        # Create base trunk model
        self.trunk = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.Tanh(),
        )

        # Create one head per parameter, the action space is categorical {decrease, no-op, increase}
        self.policy_heads = nn.ModuleList([nn.Linear(in_features=hidden_size, out_features=actions_per_param) for _ in range(n_params)])

    def forward(self, obs: torch.Tensor):
        """Step the model forward and produce action logits for each head. Returns list of logits each with size (batch_size, actions_per_param)"""
        base = self.trunk(obs)
        logits_list = [head(base) for head in self.policy_heads]
        return logits_list

    def sample_action(self, obs: torch.Tensor):
        """Sample an action per parameter. Returns (actions, summed_log_prob, value)."""
        sampled_actions = []
        log_prob_sum = torch.zeros(obs.shape[0], device=obs.device)

        logits_list = self(obs)
        for logits in logits_list:
            distribution = Categorical(logits=logits)
            sampled_action = distribution.sample()
            log_prob_sum = log_prob_sum + distribution.log_prob(sampled_action)
            sampled_actions.append(sampled_action)
        
        # Reshape sampled actions to be size (batch, n_params)
        sampled_actions = torch.stack(sampled_actions, dim=-1)
        return sampled_actions, log_prob_sum

    def evaluate_action(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate given actions to determine log probability and entropy. Returns (log_prob_sum, mean_entropy)."""
        log_prob_sum = torch.zeros(obs.shape[0], device=obs.device)
        entropy_sum = torch.zeros(obs.shape[0], device=obs.device)

        logits_list = self(obs)
        for i, logits in enumerate(logits_list):
            distribution = Categorical(logits=logits)
            log_prob_sum = log_prob_sum + distribution.log_prob(actions[:, i])
            entropy_sum = entropy_sum + distribution.entropy()

        mean_entropy = entropy_sum / self.n_params
        return log_prob_sum, mean_entropy


class Critic(nn.Module):
    """Critic model to estimate value function."""

    def __init__(self, obs_dim: int, hidden_size: int = 256):
        """Initialize Critic model, setup network"""
        super().__init__()

        # Use same size/number of layers as Actor
        # Output will be (batch_size, 1)
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )    

    def forward(self, obs: torch.Tensor):
        """Step critic model forward to get value estimate"""
        output_val = self.network(obs).squeeze(-1)
        assert output_val.ndim == 1
        return output_val


class RolloutBuffer:
    """Buffer to store all needed variables from rollouts."""

    def __init__(self, n_steps: int, obs_dim: int, n_params: int):
        """Initialize buffer as numpy arrays of size n_steps."""
        self.n_steps = n_steps
        self.obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_params), dtype=np.int64)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)

        # Index to keep track of current location in the buffer and start of current episode in buffer
        self.ep_begin_ptr = 0
        self.ptr = 0

    def store(self, obs, action, log_prob, reward, done, value):
        """Add a new set of data to the buffer, and increment the index."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute advantages and returns using generalized advantage estimator (GAE)"""
        last_adv = 0.0
        for t in reversed(range(self.ep_begin_ptr, self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv

        self.returns[self.ep_begin_ptr:self.ptr] = self.advantages[self.ep_begin_ptr:self.ptr] + self.values[self.ep_begin_ptr:self.ptr]

    def get_batches(self, batch_size: int):
        """Divide buffer randomly into batches of size batch_size, provide each batch one at a time."""
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
    
    def next_episode(self):
        """Indicate that we have started a new episode."""
        self.ep_begin_ptr = self.ptr

    def reset(self):
        """Reset current index and beginning of episode to 0."""
        self.ep_begin_ptr = 0
        self.ptr = 0


class PPOAgentNonShared:
    """PPO agent that utilizes separate Actor and Critic networks (not shared trunk implementation)."""

    def __init__(self, env, config: dict):
        """Use given config and environment to set up necessary variables."""
        self.env = env
        obs_dim = env.observation_space.shape[0]
        n_params = env.action_space.shape[0] 

        # PPO configurations from the config file
        ppo_cfg = config["ppo"]
        self.lr = float(ppo_cfg["learning_rate"])
        self.n_steps = int(ppo_cfg["n_steps"])
        self.batch_size = int(ppo_cfg["batch_size"])
        self.n_epochs = int(ppo_cfg["n_epochs"])
        self.gamma = float(ppo_cfg["gamma"])
        self.total_timesteps = int(ppo_cfg["total_timesteps"])

        # Hardcoded PPO parameters
        self.clip_eps = 0.2
        self.vf_coef = 0.5
        self.ent_coef = 0.01
        self.gae_lambda = 0.95
        self.max_grad_norm = 0.5

        # Set up the Actor and Critic networks and their optimizers
        self.actor_network = Actor(obs_dim, n_params)
        self.critic_network = Critic(obs_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=self.lr)

        # Create buffer to hold rollouts
        self.buffer = RolloutBuffer(self.n_steps, obs_dim, n_params)

    def collect_rollouts(self) -> list[dict]:
        """Run environment for n_steps, fill self.buffer. Return list of episode stats."""
        self.buffer.reset()
        
        episode_stats = []
        total_steps = 0

        # Step until we've filled up the buffer
        while total_steps < self.n_steps:
            # For each new episode, reset the environment and the counts
            obs, _ = self.env.reset()

            ep_reward = 0.0
            ep_len = 0

            # Take at most max_steps number of steps per episode
            for step in range(self.env._max_steps):
                # Sample an action and get the value
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action, log_prob = self.actor_network.sample_action(obs_t)
                    value = self.critic_network(obs_t)

                action_np = action.squeeze(0).numpy() 

                # Step the environment to get next observation and reward
                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated

                # Store all the data into the buffer
                self.buffer.store(obs, action_np, log_prob.item(), reward, float(done), value.item())

                # Increment counts
                ep_reward += reward
                ep_len += 1

                total_steps += 1

                # If episode is done (terminated or has reached horizon), or we've reached buffer size
                if done or step == self.env._max_steps - 1 or total_steps == self.n_steps:
                    episode_stats.append({"reward": ep_reward, "length": ep_len})

                    # Bootstrap value for last state
                    with torch.no_grad():
                        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                        last_value = self.critic_network(obs_t)

                    # Calculate advantges and returns for the episode using GAE
                    self.buffer.compute_gae(last_value.item(), self.gamma, self.gae_lambda)
                    self.buffer.next_episode()
                    break
                else:
                    obs = next_obs

        return episode_stats

    def update(self) -> dict:
        """Run PPO update for a group of batches (number of batches is buffer_size / batch_size)"""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        # Loop for n_epochs to perform update
        for _ in range(self.n_epochs):
            # Split buffer into mini batches and before PPO clipped update
            for obs_b, act_b, old_lp_b, ret_b, adv_b in self.buffer.get_batches(self.batch_size):
                # Normalize advantages
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                # Value loss
                values = self.critic_network(obs_b)
                value_loss = nn.functional.mse_loss(values, ret_b)

                # Update Critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # Policy loss (clipped)

                # Get new log_probs 
                new_lp, entropy = self.actor_network.evaluate_action(obs_b, act_b)

                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.ent_coef * entropy_loss

                # Update Actor 
                self.actor_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update statistics
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
        """"Train PPO Agent using Actor and Critic models."""
        total = total_timesteps or self.total_timesteps
        timesteps_done = 0
        iteration = 0

        # Loop until reached max number of interactions with environment
        while timesteps_done < total:
            episode_stats = self.collect_rollouts()
            loss_stats = self.update()
            timesteps_done += self.n_steps
            iteration += 1

            if callback:
                callback(timesteps_done, episode_stats, loss_stats)

    def save(self, path: str):
        """Save all state to path."""
        torch.save({
            "actor_network": self.actor_network.state_dict(),
            "critic_network": self.critic_network.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load all state from path."""
        checkpoint = torch.load(path, weights_only=True)
        self.actor_network.load_state_dict(checkpoint["actor_network"])
        self.critic_network.load_state_dict(checkpoint["critic_network"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
