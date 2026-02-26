"""Load a trained agent and run it on a circuit to see what it produces."""
import argparse
import csv
import os

import torch
import yaml

from circuitrl.envs.circuit_env import CircuitEnv
from circuitrl.agents.ppo_agent import ActorCritic
from circuitrl.agents.grpo_agent import Actor


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


def load_agent(run_dir: str, agent:str, config_override: str | None = None):
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

    if agent == "ppo":
        network = ActorCritic(obs_dim, n_params)
    elif agent == "grpo":
        network = Actor(obs_dim, n_params)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    network.load_state_dict(checkpoint["network"])
    network.eval()

    return env, network, config


def spec_met(metric_val: float, target: float, tolerance: float, direction: str) -> bool:
    """Direction-aware spec check matching circuit_env._check_specs_met."""
    if direction == 'max':
        return metric_val >= target - tolerance
    elif direction == 'min':
        return metric_val <= target + tolerance
    else:
        return abs(metric_val - target) <= tolerance


def run_episode(env, network, agent):
    """Run one greedy episode. Returns (steps, total_reward, success, episode_targets)."""
    obs, info = env.reset()
    episode_targets = info["targets"]  # targets sampled for this episode
    steps = []
    total_reward = 0.0

    for _ in range(env._max_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            if agent == "ppo":
                logits_list, _ = network(obs_t)
            elif agent == "grpo":   
                logits_list = network(obs_t)

        actions = torch.stack([logits.argmax(dim=-1) for logits in logits_list], dim=-1)
        action = actions.squeeze(0).numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        steps.append({
            "step": len(steps) + 1,
            "action": action.tolist(),
            "reward": reward,
            "params": info.get("params", {}),
            "metrics": info.get("metrics", {}),
        })

        if terminated or truncated:
            break

    return steps, total_reward, terminated, episode_targets


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CircuitRL agent")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to run directory (contains model.pt and config.yaml)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to original config YAML (auto-detected if omitted)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to evaluate across")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every step, not just episode summary")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for target sampling")
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo", "grpo"])
    args = parser.parse_args()

    env, network, config = load_agent(args.run_dir, args.agent, config_override=args.config)
    env.reset(seed=args.seed)  # seed env's np_random for reproducible target sampling

    spec_names = list(config["target_specs"].keys())
    tolerances = {name: float(spec["tolerance"]) for name, spec in config["target_specs"].items()}
    directions = {name: spec.get("direction", "equal") for name, spec in config["target_specs"].items()}

    print(f"Loaded agent from {args.run_dir}")
    print(f"Evaluating {args.episodes} episodes  (seed={args.seed})")
    print()

    all_rewards = []
    all_successes = []
    all_steps = []
    spec_successes = {name: [] for name in spec_names}
    csv_rows = []

    for ep in range(args.episodes):
        steps, total_reward, success, episode_targets = run_episode(env, network, args.agent)
        all_rewards.append(total_reward)
        all_successes.append(success)
        all_steps.append(len(steps))

        final_metrics = steps[-1].get("metrics", {})
        per_spec = {}
        for name in spec_names:
            if name in final_metrics:
                met = spec_met(final_metrics[name], episode_targets[name],
                               tolerances[name], directions[name])
                per_spec[name] = met
                spec_successes[name].append(met)

        row = {"total_reward": total_reward, "success": int(success), "n_steps": len(steps)}
        for name in spec_names:
            row[f"target_{name}"] = episode_targets.get(name, float("nan"))
            row[f"final_{name}"] = final_metrics.get(name, float("nan"))
        csv_rows.append(row)

        # Always print episode summary; verbose adds per-step trace
        if args.verbose:
            for s in steps:
                action_labels = ["dec", "nop", "inc"]
                actions_str = " ".join(action_labels[a] for a in s["action"])
                print(f"  step {s['step']:>3d}  [{actions_str}]  reward: {s['reward']:>8.3f}")

        targets_str = "  ".join(f"{n}={episode_targets[n]:.3g}" for n in spec_names)
        print(f"Episode {ep + 1:>3d}:  steps={len(steps):>3d}  "
              f"reward={total_reward:>8.3f}  "
              f"{'SUCCESS' if success else 'FAIL   '}  "
              f"targets: [{targets_str}]")

        for name in spec_names:
            if name in final_metrics:
                val = final_metrics[name]
                tgt = episode_targets[name]
                tol = tolerances[name]
                met = per_spec.get(name, False)
                print(f"    {name}: {val:.4g}  (target: {tgt:.4g}, tol: {tol:.3g})  "
                      f"[{'OK  ' if met else 'MISS'}]")
        print()

    # Save eval results CSV for plot.py --eval
    if csv_rows:
        csv_path = os.path.join(args.run_dir, "eval_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved eval results to {csv_path}\n")

    # Aggregate summary
    n = args.episodes
    print("=" * 60)
    print(f"Summary over {n} episodes:")
    print(f"  Success rate:      {sum(all_successes)}/{n}  ({100*sum(all_successes)/n:.1f}%)")
    print(f"  Mean total reward: {sum(all_rewards)/n:.3f}")
    print(f"  Mean episode len:  {sum(all_steps)/n:.1f}")
    print(f"  Per-spec success rates:")
    for name in spec_names:
        if spec_successes[name]:
            rate = sum(spec_successes[name]) / len(spec_successes[name])
            print(f"    {name}: {100*rate:.1f}%")


if __name__ == "__main__":
    main()
