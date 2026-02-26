import argparse
import csv
import os
import shutil
import time

import numpy as np
import torch
import yaml

from circuitrl.envs.circuit_env import CircuitEnv

CSV_FIELDS = [
    "timestep", "elapsed", "episodes", "circuit", "mean_reward", "mean_len",
    "policy_loss", "value_loss", "entropy",
]


def make_callback(run_dir: str, circuit_name: str):
    start_time = time.time()
    csv_path = os.path.join(run_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()

    def callback(timesteps_done, episode_stats, loss_stats):
        if not episode_stats:
            return
        elapsed = time.time() - start_time
        rewards = [ep["reward"] for ep in episode_stats]
        lengths = [ep["length"] for ep in episode_stats]
        mean_reward = sum(rewards) / len(rewards)
        mean_len = sum(lengths) / len(lengths)

        print(
            f"[{timesteps_done:>7d} steps | {elapsed:6.1f}s] "
            f"episodes: {len(episode_stats):>3d}  "
            f"mean_reward: {mean_reward:>8.3f}  "
            f"mean_len: {mean_len:>5.1f}  "
            f"policy_loss: {loss_stats['policy_loss']:.4f}  "
            # f"value_loss: {loss_stats['value_loss']:.4f}  "
            f"entropy: {loss_stats['entropy']:.4f}"
        )

        writer.writerow({
            "timestep": timesteps_done,
            "elapsed": f"{elapsed:.1f}",
            "episodes": len(episode_stats),
            "circuit": circuit_name,
            "mean_reward": f"{mean_reward:.4f}",
            "mean_len": f"{mean_len:.1f}",
            "policy_loss": f"{loss_stats['policy_loss']:.4f}",
            # "value_loss": f"{loss_stats['value_loss']:.4f}",
            "entropy": f"{loss_stats['entropy']:.4f}",
        })
        csv_file.flush()

    def close():
        csv_file.close()

    callback.close = close
    return callback


def main():
    parser = argparse.ArgumentParser(description="Train a CircuitRL agent")
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo", "grpo"])
    parser.add_argument("--config", type=str, default="circuitrl/configs/opamp_default.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total_timesteps from config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Extract circuit name from config comment line (e.g. "# Common-Source Amplifier")
    with open(args.config) as f:
        first_line = f.readline().strip()
    circuit_name = first_line.lstrip("# ").strip() if first_line.startswith("#") else "unknown"

    circuit_tag = os.path.splitext(os.path.basename(args.config))[0]  # e.g. "cs_amp"
    run_name = args.run_name or f"{args.agent}_{circuit_tag}_seed{args.seed}"
    run_dir = os.path.join("runs", run_name)
    # Auto-increment to avoid overwriting previous runs
    if os.path.exists(run_dir):
        i = 1
        while os.path.exists(f"{run_dir}_{i}"):
            i += 1
        run_dir = f"{run_dir}_{i}"
    os.makedirs(run_dir)
    shutil.copy2(args.config, os.path.join(run_dir, "config.yaml"))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = CircuitEnv(config_path=args.config)
    env.reset(seed=args.seed)  # seeds env's np_random for _sample_targets
    
    if args.agent == "ppo":
        from circuitrl.agents.ppo_agent import PPOAgent
        agent = PPOAgent(env, config)
    elif args.agent == "grpo":
        from circuitrl.agents.grpo_agent import GRPOAgent
        agent = GRPOAgent(env, config)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    print(f"Training {args.agent} | run: {run_name} | seed: {args.seed}")
    print(f"Checkpoint dir: {run_dir}/")
    print()

    total = args.timesteps or int(config[args.agent]["total_timesteps"])
    cb = make_callback(run_dir, circuit_name)
    agent.train(total_timesteps=total, callback=cb)
    cb.close()

    checkpoint_path = os.path.join(run_dir, "model.pt")
    agent.save(checkpoint_path)
    print(f"\nSaved checkpoint to {checkpoint_path}")
    print(f"Metrics log: {os.path.join(run_dir, 'metrics.csv')}")


if __name__ == "__main__":
    main()
