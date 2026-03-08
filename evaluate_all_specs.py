"""Load a trained agent, and run it on every single circuit in the spec pool."""
import argparse
import csv
import os

import numpy as np
import torch
import yaml

from circuitrl.envs.circuit_env import CircuitEnv


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


def _resolve_agent_name(agent: str, checkpoint: dict) -> str:
    if agent != "auto":
        return agent
    if "seq_network" in checkpoint:
        return "ppo-seq"
    if "actor_network" in checkpoint:
        return "ppo_non_shared"
    if "network" in checkpoint:
        return "ppo"
    raise ValueError("Unable to auto-detect agent type from checkpoint keys")


def load_agent(agent: str, run_dir: str, spec_pool_test: str, config_override: str | None = None):
    """Load config and actor network from a run directory."""
    checkpoint_path = os.path.join(run_dir, "model.pt")
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    chosen_agent = _resolve_agent_name(agent, checkpoint)
    is_sequential = chosen_agent == "ppo-seq"

    # Use original config for correct netlist path resolution
    config_path = config_override or find_original_config(run_dir)
    if config_path is None:
        config_path = os.path.join(run_dir, "config.yaml")
        print(f"Warning: could not find original config, using {config_path}")
        print("  (netlist path may not resolve — pass --config to fix)")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    env = CircuitEnv(config_path=config_path, sequential=is_sequential)

    # If specified spec pool to test on, set the new spec pool
    if spec_pool_test:
        env.set_spec_pool(spec_pool_test)

    obs_dim = env.observation_space.shape[0]
    n_params = len(config["parameters"])

    if chosen_agent == "ppo":
        from circuitrl.agents.ppo_agent import ActorCritic
        actions_per_param = checkpoint["network"]["policy_heads.0.weight"].shape[0]
        network = ActorCritic(obs_dim, n_params, actions_per_param=actions_per_param)
        network.load_state_dict(checkpoint["network"])
    elif chosen_agent == "ppo_non_shared":
        from circuitrl.agents.ppo_agent_non_shared import Actor
        actions_per_param = checkpoint["actor_network"]["policy_heads.0.weight"].shape[0]
        network = Actor(obs_dim, n_params, actions_per_param=actions_per_param)
        network.load_state_dict(checkpoint["actor_network"])
    elif chosen_agent == "ppo-seq":
        from circuitrl.agents.ppo_agent import SeqActorCritic
        network = SeqActorCritic(obs_dim)
        network.load_state_dict(checkpoint["seq_network"])
    else:
        raise ValueError(f"Unknown agent: {chosen_agent}")

    network.eval()
    return env, network, config, is_sequential, chosen_agent


def spec_met(metric_val: float, target: float, tolerance: float, direction: str) -> bool:
    """Direction-aware spec check matching circuit_env._check_specs_met."""
    if direction == 'max':
        return metric_val >= target - tolerance
    elif direction == 'min':
        return metric_val <= target + tolerance
    else:
        return abs(metric_val - target) <= tolerance


def greedy_action(network, obs_t, is_sequential):
    """Return greedy (argmax) action for either agent type."""
    with torch.no_grad():
        if is_sequential:
            h = network.trunk(obs_t)
            return int(network.policy_head(h).argmax(dim=-1).item())
        logits_list = network.get_logits_list(obs_t)
        actions = torch.stack([l.argmax(dim=-1) for l in logits_list], dim=-1)
        return actions.squeeze(0).numpy()


def _action_to_list(action):
    if np.isscalar(action):
        return [int(action)]
    return np.asarray(action).astype(int).tolist()


def run_episode(env, network, is_sequential, seed, target_idx):
    """Run one greedy episode. Returns (steps, total_reward, success, episode_targets)."""
    obs, info = env.reset_with_target_idx(seed=seed, target_idx=target_idx)
    episode_targets = info["targets"]
    steps = []
    total_reward = 0.0

    for _ in range(env._max_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        action = greedy_action(network, obs_t, is_sequential)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        steps.append({
            "step": len(steps) + 1,
            "action": _action_to_list(action),
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
    parser.add_argument("--verbose", action="store_true",
                        help="Print every step, not just episode summary")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for env sampling")
    parser.add_argument("--spec_pool_test", type=str, help="Spec pool to evaluate on")
    parser.add_argument("--agent", type=str, default="auto", choices=["auto", "ppo", "ppo_non_shared", "ppo-seq"],
                        help="Agent type (auto detects from checkpoint)")
    args = parser.parse_args()

    env, network, config, is_sequential, chosen_agent = load_agent(
        args.agent, args.run_dir, args.spec_pool_test, config_override=args.config
    )

    spec_names = list(config["target_specs"].keys())
    tolerances = {name: float(spec["tolerance"]) for name, spec in config["target_specs"].items()}
    directions = {name: spec.get("direction", "equal") for name, spec in config["target_specs"].items()}

    print(f"Loaded {chosen_agent} agent from {args.run_dir}")
    print(f"Evaluating {len(env._spec_pool)} episodes  (seed={args.seed})")
    print()

    all_rewards = []
    all_successes = []
    all_steps = []
    spec_successes = {name: [] for name in spec_names}
    csv_rows = []

    # Loop through everything in the spec pool
    for ep in range(len(env._spec_pool)):
        steps, total_reward, success, episode_targets = run_episode(env, network, is_sequential, args.seed, ep)
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
    n = len(env._spec_pool)
    lines = [
        "=" * 60,
        "Summary over full spec pool:",
        f"  Success rate:      {sum(all_successes)}/{n}  ({100*sum(all_successes)/n:.1f}%)",
        f"  Mean total reward: {sum(all_rewards)/n:.3f}",
        f"  Mean episode len:  {sum(all_steps)/n:.1f}",
        "  Per-spec success rates:",
    ]
    for name in spec_names:
        if spec_successes[name]:
            rate = sum(spec_successes[name]) / len(spec_successes[name])
            lines.append(f"    {name}: {100*rate:.1f}%")

    summary = "\n".join(lines)
    print(summary)

    summary_path = os.path.join(args.run_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
