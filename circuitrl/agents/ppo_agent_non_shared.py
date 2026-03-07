import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

class Actor(nn.Module):
    """Categorical actor policy"""

    def __init__(self, obs_dim: int, n_params: int, actions_per_param: int = 3, n_layers: int = 2, hidden_size: int = 256):
        """
        Initializing categorical actor policy for use in PPO
        """

        super().__init__()
        self.n_params = n_params
        self.actions_per_param = actions_per_param

        # Initialize the network
        model = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=hidden_size),
            nn.ReLU()
        )

        # Hidden layers
        for _ in range(n_layers - 1):
            model.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            model.append(nn.ReLU())

        # Output layer
        model.append(nn.Linear(in_features=hidden_size, out_features=(n_params * actions_per_param)))

        self.network = model

    def forward(self, obs: torch.Tensor):
        logits = self.network(obs)
        return logits

    def sample_action(self, obs: torch.Tensor):
        sampled_actions = []
        log_probs = torch.zeros((obs.shape[0])) 

        logits = self(obs)
        for i in range(self.n_params):
            cur_logits = logits[:, i:i+self.actions_per_param]
            distribution = Categorical(logits=cur_logits)
            cur_action = distribution.sample()
            sampled_actions.append(cur_action)
            log_probs = log_probs + distribution.log_prob(cur_action)

        sampled_actions = torch.stack(sampled_actions, dim=-1)
        return sampled_actions, log_probs
    
    def evaluate_action(self, obs: torch.Tensor, act: torch.Tensor):
        logits = self(obs)
        log_probs = torch.zeros((obs.shape[0])) 
        for i in range(self.n_params):
            cur_logits = logits[:, i:i+self.actions_per_param]
            distribution = Categorical(logits=cur_logits)
            log_probs = log_probs + distribution.log_prob(act[:,i])
        
        return log_probs

class Critic(nn.Module):
    """Critic policy (to estimate value function)"""

    def __init__(self, obs_dim: int, n_layers: int = 2, hidden_size: int = 256, lr: int = 3e-2):
        """
        Initializing baseline network 
        """
        super().__init__()

        # Initialize the network
        model = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=hidden_size),
            nn.ReLU()
        )

        # Hidden layers
        for _ in range(n_layers - 1):
            model.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            model.append(nn.ReLU())

        # Output layer (output a single value)
        model.append(nn.Linear(in_features=hidden_size, out_features=1))
        self.network = model

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
    

    def forward(self, obs: torch.Tensor):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]
        """
       
        output_val = self.network(obs).squeeze(-1)
        assert output_val.ndim == 1
        return output_val

    def calculate_advantage(self, ret: torch.Tensor, obs: torch.Tensor):
        """
        Args:
            returns: np.array of shape [batch size]
                all discounted future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]
        """
        obs = torch.tensor(obs, dtype=torch.float32)

        # Get baseline by running NN on observations
        baseline = self(obs)
        advantages = ret - baseline.detach().numpy()
        return advantages

    def update_baseline(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size], containing all discounted
                future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        """

        returns = torch.tensor(returns, dtype=torch.float32)
        observations = torch.tensor(observations, dtype=torch.float32)

        # Get baseline by running NN on observations
        baseline = self(observations)

        # Loss function is MSE loss
        loss = nn.functional.mse_loss(baseline, returns)

        # Reset gradient, backpropogate, then step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class PPOAgentNonShared:
    def __init__(self, env, config: dict):
        self.env = env
        obs_dim = env.observation_space.shape[0]
        n_params = env.action_space.shape[0]

        ppo_cfg = config["ppo"]
        self.lr = float(ppo_cfg["learning_rate"])
        self.n_steps = int(ppo_cfg["n_steps"])
        self.batch_size = int(ppo_cfg["batch_size"])
        self.n_epochs = int(ppo_cfg["n_epochs"])
        self.gamma = float(ppo_cfg["gamma"])
        self.total_timesteps = int(ppo_cfg["total_timesteps"])

        # PPO-specific
        self.clip_eps = 0.2
        self.vf_coef = 0.5
        self.ent_coef = 0.01
        self.gae_lambda = 0.95
        self.max_grad_norm = 0.5

        self.actor_network = Actor(obs_dim, n_params)
        self.optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=self.lr)
        self.critic_network = Critic(obs_dim)

    def collect_rollouts(self):
        """
        Sample paths (trajectories) from the environment.

        Args:
            num_episodes: the number of episodes to be sampled
                if none, sample one batch (size indicated by config file)
            env: open AI Gym envinronment

        Returns:
            paths: a list of paths. Each path in paths is a dictionary with
                path["observation"] a numpy array of ordered observations in the path
                path["actions"] a numpy array of the corresponding actions in the path
                path["reward"] a numpy array of the corresponding rewards in the path
            total_rewards: the sum of all rewards encountered during this "path"
        """
        episode_stats = [] 
        paths = []

        total_steps = 0

        while total_steps < self.batch_size:
            obs, _ = self.env.reset()
            
            episode_reward = 0.0
            episode_length = 0

            observations, actions, rewards, log_probs, dones = [], [], [], [], []

            for step in range(self.batch_size):
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action, log_prob = self.actor_network.sample_action(obs_tensor)

                action_np = action.squeeze(0).numpy()
                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                ep_done = terminated or truncated
                
                observations.append(obs)
                actions.append(action_np)
                log_probs.append(log_prob.item())
                rewards.append(reward)
                dones.append(float(ep_done))
                
                episode_reward += reward
                episode_length += 1

                total_steps += 1

                if ep_done or step == self.batch_size - 1:
                    episode_stats.append({"reward": episode_reward, "length": episode_length})
                    break
                elif total_steps == self.batch_size:
                    break
                else:
                    obs = next_obs

            path = {
                "observation": np.array(observations),
                "reward": np.array(rewards),
                "action": np.array(actions),
                "log_prob": np.array(log_probs),
                "done": np.array(dones),
            }
            paths.append(path)

        return paths, episode_stats
    

    def get_returns(self, paths):
        """
        Calculate the returns G_t for each timestep

        Args:
            paths: recorded sample paths. See sample_path() for details.

        Return:
            returns: return G_t for each timestep
        """
        all_returns = []
        for path in paths:
            rewards = path["reward"]

            # Initialize returns array and path length
            returns = np.zeros_like(rewards)
            path_length = returns.shape[0]

            # Initialize last return
            returns[path_length - 1] = rewards[path_length - 1]

            # Loop backwards to calculate returns
            for idx in range(path_length - 1):
                reverse_idx = path_length - 2 - idx
                returns[reverse_idx] = rewards[reverse_idx] + (self.gamma * returns[reverse_idx + 1])

            all_returns.append(returns)
        returns = np.concatenate(all_returns)

        return returns
    
    def normalize_advantage(self, advantages):
        """
        Args:
            advantages: np.array of shape [batch size]
        Returns:
            normalized_advantages: np.array of shape [batch size]
        """
        mean = np.mean(advantages)
        std = np.std(advantages)

        normalized_advantages = (advantages - mean) / std
        return normalized_advantages

    def calculate_advantage(self, returns, observations):
        """
        Calculates the advantage for each of the observations
        Args:
            returns: np.array of shape [batch size]
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]
        """

        advantages = self.critic_network.calculate_advantage(returns, observations)
        advantages = self.normalize_advantage(advantages)

        return advantages

    def update_policy(self, observations, actions, advantages, old_logprobs):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size, 1]
            old_logprobs: np.array of shape [batch size]

        Perform one update on the policy using the provided data using the PPO clipped
        objective function.
        """
        observations = torch.tensor(observations, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        old_logprobs = torch.tensor(old_logprobs, dtype=torch.float32)

        # Get distribution and log probs
        log_probs = self.actor_network.evaluate_action(observations, actions)

        # Determine z_ratior
        z_ratio = torch.exp(log_probs - old_logprobs)

        # Perform clipping
        op_1 = torch.clamp(z_ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        op_2 = z_ratio * advantages

        # Calculate min for each element
        clip_vals = torch.min(op_1, op_2)

        # Set objective function
        loss = -torch.mean(clip_vals)

        # Set loss function, zero gradient, backpropogate, and step optimizer
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()


    def train(self, total_timesteps: int | None = None, callback=None):
        """
        Performs training

        You do not have to change or use anything here, but take a look
        to see how all the code you've written fits together!
        """
        total = total_timesteps or self.total_timesteps
        timesteps_done = 0
        iteration = 0

        while timesteps_done < total:
            paths, episode_stats = self.collect_rollouts()
            
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            log_probs = np.concatenate([path["log_prob"] for path in paths])

            returns = self.get_returns(paths)
            advantages = self.calculate_advantage(returns, observations)

            # run training operations
            total_policy_loss = 0
            for k in range(self.n_epochs):
                self.critic_network.update_baseline(returns, observations)
                total_policy_loss += self.update_policy(observations, actions, advantages, 
                                   log_probs)

            loss_stats = {"policy_loss": total_policy_loss / self.n_epochs, 
                          "value_loss": -1,
                          "entropy": -1}
            timesteps_done += self.batch_size
            iteration += 1

            if callback:
                callback(timesteps_done, episode_stats, loss_stats)

    def save(self, path: str):
        torch.save({
            "actor_network": self.actor_network.state_dict(),
            "critic_network": self.critic_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, weights_only=True)
        self.actor_network.load_state_dict(checkpoint["actor_network"])
        self.critic_network.load_state_dict(checkpoint["critic_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


