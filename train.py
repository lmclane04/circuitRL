import argparse
import os
import time

import yaml

from circuitrl.envs.opamp_env import OpAmpEnv


def make_callback(log_interval: int = 1):
    start_time = time.time()

    def callback(timesteps_done, episode_stats, loss_stats):
        if not episode_stats:
            return
        elapsed = time.time() - start_time
        rewards = [ep["reward"] for ep in episode_stats]
        lengths = [ep["length"] for ep in episode_stats]
        print(
            f"[{timesteps_done:>7d} steps | {elapsed:6.1f}s] "
            f"episodes: {len(episode_stats):>3d}  "
            f"mean_reward: {sum(rewards)/len(rewards):>8.3f}  "
            f"mean_len: {sum(lengths)/len(lengths):>5.1f}  "
            f"policy_loss: {loss_stats['policy_loss']:.4f}  "
            f"value_loss: {loss_stats['value_loss']:.4f}  "
            f"entropy: {loss_stats['entropy']:.4f}"
        )

    return callback


def main():
    parser = argparse.ArgumentParser(description="Train a CircuitRL agent")
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo"])
    parser.add_argument("--config", type=str, default="circuitrl/configs/opamp_default.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total_timesteps from config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_name = args.run_name or f"{args.agent}_seed{args.seed}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    env = OpAmpEnv(config_path=args.config)

    if args.agent == "ppo":
        from circuitrl.agents.ppo_agent import PPOAgent
        agent = PPOAgent(env, config)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    print(f"Training {args.agent} | run: {run_name} | seed: {args.seed}")
    print(f"Checkpoint dir: {run_dir}/")
    print()

    total = args.timesteps or int(config[args.agent]["total_timesteps"])
    agent.train(total_timesteps=total, callback=make_callback())

    checkpoint_path = os.path.join(run_dir, "model.pt")
    agent.save(checkpoint_path)
    print(f"\nSaved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
