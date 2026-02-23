"""Load a trained agent and run it on a circuit to see what it produces."""
import argparse
import os

import torch
import yaml

from circuitrl.envs.circuit_env import CircuitEnv
from circuitrl.agents.ppo_agent import ActorCritic


def find_original_config(run_dir: str) -> str | None:
    """Try to find the original config in circuitrl/configs/ matching the run's config."""
    run_config = os.path.join(run_dir, "config.yaml")
    with open(run_config) as f:
        first_line = f.readline().strip()
    # Match by first comment line (circuit name)
    configs_dir = os.path.join("circuitrl", "configs")
    if os.path.isdir(configs_dir):
        for name in os.listdir(configs_dir):
            if name.endswith(".yaml"):
                path = os.path.join(configs_dir, name)
                with open(path) as f:
                    if f.readline().strip() == first_line:
                        return path
    return None


def load_agent(run_dir: str, config_override: str | None = None):
    """Load config and network from a run directory."""
    checkpoint_path = os.path.join(run_dir, "model.pt")

    # Use original config for correct netlist path resolution
    config_path = config_override or find_original_config(run_dir)
    if config_path is None:
        config_path = os.path.join(run_dir, "config.yaml")
        print(f"Warning: could not find original config, using {config_path}")
        print("  (netlist path may not resolve — pass --config to fix)")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    env = CircuitEnv(config_path=config_path)
    obs_dim = env.observation_space.shape[0]
    n_params = len(config["parameters"])

    network = ActorCritic(obs_dim, n_params)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    network.load_state_dict(checkpoint["network"])
    network.eval()

    return env, network, config


def run_episode(env, network):
    """Run one episode with greedy actions, returning step-by-step data."""
    obs, info = env.reset()
    steps = []
    total_reward = 0.0

    for step in range(env._max_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits_list, value = network(obs_t)

        actions = torch.stack([logits.argmax(dim=-1) for logits in logits_list], dim=-1)
        action = actions.squeeze(0).numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        steps.append({
            "step": step + 1,
            "action": action.tolist(),
            "reward": reward,
            "params": info.get("params", {}),
            "metrics": info.get("metrics", {}),
        })

        if terminated or truncated:
            break

    return steps, total_reward, terminated


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CircuitRL agent")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to run directory (contains model.pt and config.yaml)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to original config YAML (auto-detected if omitted)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every step, not just the final result")
    args = parser.parse_args()

    env, network, config = load_agent(args.run_dir, config_override=args.config)

    target_specs = {name: spec["value"] for name, spec in config["target_specs"].items()}
    tolerances = {name: spec["tolerance"] for name, spec in config["target_specs"].items()}

    print(f"Loaded agent from {args.run_dir}")
    print(f"Target specs: {target_specs}")
    print(f"Tolerances:   {tolerances}")
    print()

    for ep in range(args.episodes):
        steps, total_reward, success = run_episode(env, network)

        if args.verbose:
            for s in steps:
                action_labels = ["dec", "nop", "inc"]
                actions_str = " ".join(action_labels[a] for a in s["action"])
                print(f"  step {s['step']:>3d}  [{actions_str}]  reward: {s['reward']:>8.3f}")

        final = steps[-1]
        print(f"Episode {ep + 1}:")
        print(f"  Steps: {len(steps)}  Total reward: {total_reward:.3f}  Success: {success}")
        print(f"  Final parameters:")
        for name, val in final["params"].items():
            print(f"    {name}: {val:.4e}")
        if final["metrics"]:
            print(f"  Final metrics vs targets:")
            for name, val in final["metrics"].items():
                target = target_specs[name]
                tol = tolerances[name]
                met = "OK" if abs(val - target) <= tol else "MISS"
                print(f"    {name}: {val:.4f}  (target: {target}, tol: {tol})  [{met}]")
        print()


if __name__ == "__main__":
    main()
