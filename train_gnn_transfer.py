import argparse
import csv
import os
import shutil
import time

import numpy as np
import torch
import yaml

from circuitrl.agents.ppo_gnn_agent import (
    GraphObservationAdapter,
    GraphSpec,
    MultiCircuitPPOGNNAgent,
    PPOGNNAgent,
)
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
            f"value_loss: {loss_stats['value_loss']:.4f}  "
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
            "value_loss": f"{loss_stats['value_loss']:.4f}",
            "entropy": f"{loss_stats['entropy']:.4f}",
        })
        csv_file.flush()

    def close():
        csv_file.close()

    callback.close = close
    return callback


def _extract_circuit_name(config_path: str) -> str:
    with open(config_path) as f:
        first_line = f.readline().strip()
    return first_line.lstrip("# ").strip() if first_line.startswith("#") else "unknown"


def _resolve_graph_spec_path(config_path: str, explicit_path: str | None) -> str | None:
    if explicit_path:
        return explicit_path

    base = os.path.splitext(os.path.basename(config_path))[0]
    candidate = os.path.join("circuitrl", "configs", "graph_specs", f"{base}_graph.yaml")
    if os.path.exists(candidate):
        return candidate
    return None


def _build_graph_spec(graph_spec_path: str | None, param_names: list[str]) -> GraphSpec:
    if graph_spec_path:
        print(f"Using graph spec: {graph_spec_path}")
        return GraphSpec.from_path(graph_spec_path, param_names)
    print("Using default chain graph (no graph spec file provided)")
    return GraphSpec.default(param_names)


def _parse_circuit_spec(spec: str) -> tuple[str, str, str | None]:
    """Parse 'id:config' or 'id:config:graph_spec' into (circuit_id, config_path, graph_spec_path)."""
    parts = spec.split(":")
    if len(parts) == 2:
        return parts[0], parts[1], None
    elif len(parts) == 3:
        return parts[0], parts[1], parts[2]
    else:
        raise ValueError(
            f"Invalid --circuits entry '{spec}'. "
            f"Expected 'circuit_id:config_path' or 'circuit_id:config_path:graph_spec_path'."
        )


def _build_circuit_def(
    circuit_id: str,
    config_path: str,
    graph_spec_path: str | None,
    seed: int,
    metric_slots: int,
) -> dict:
    """Build a circuit definition dict for MultiCircuitPPOGNNAgent."""
    env = CircuitEnv(config_path=config_path)
    env.reset(seed=seed)
    resolved_gs = graph_spec_path or _resolve_graph_spec_path(config_path, None)
    graph_spec = _build_graph_spec(resolved_gs, env._param_names)
    adapter = GraphObservationAdapter(
        n_params=env._n_params,
        n_metrics=env._n_metrics,
        graph_spec=graph_spec,
        metric_slots=metric_slots,
    )
    return {"id": circuit_id, "env": env, "adapter": adapter, "graph_spec": graph_spec}


def main():
    parser = argparse.ArgumentParser(description="Train graph-based PPO with optional transfer initialization")
    parser.add_argument("--config", type=str, required=True,
                        help="Circuit config YAML (single-circuit mode) or PPO hyperparams source (multi-circuit mode)")
    parser.add_argument("--circuits", nargs="+", default=None,
                        help="Multi-circuit mode: space-separated 'id:config' or 'id:config:graph_spec' entries. "
                             "Example: --circuits cs_amp:circuitrl/configs/cs_amp.yaml "
                             "opamp:circuitrl/configs/opamp.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps from config")
    parser.add_argument("--graph-spec", type=str, default=None, help="Path to graph YAML (single-circuit only)")
    parser.add_argument("--metric-slots", type=int, default=4, help="Fixed metric padding slots for transfer")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Checkpoint from previous gnn run for transfer initialization")
    parser.add_argument("--freeze-encoder-steps", type=int, default=0,
                        help="Freeze graph encoder for first N timesteps (useful in transfer finetuning)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    phase = "ft" if args.init_checkpoint else "pre"
    multi_mode = bool(args.circuits)

    # ---- Run directory setup ------------------------------------------------
    if multi_mode:
        parsed_circuits = [_parse_circuit_spec(s) for s in args.circuits]
        circuit_tag = "+".join(cid for cid, _, _ in parsed_circuits)
        circuit_name = f"multi({circuit_tag})"
    else:
        circuit_name = _extract_circuit_name(args.config)
        circuit_tag = os.path.splitext(os.path.basename(args.config))[0]

    run_name = args.run_name or f"gnn_{phase}_{circuit_tag}_s{args.seed}"
    run_dir = os.path.join("runs", run_name)
    if os.path.exists(run_dir):
        i = 1
        while os.path.exists(f"{run_dir}_{i}"):
            i += 1
        run_dir = f"{run_dir}_{i}"
    os.makedirs(run_dir)

    shutil.copy2(args.config, os.path.join(run_dir, "config.yaml"))
    if args.graph_spec and not multi_mode:
        shutil.copy2(args.graph_spec, os.path.join(run_dir, "graph_spec.yaml"))

    if args.timesteps:
        with open(os.path.join(run_dir, "config.yaml"), "r") as f:
            first_line = f.readline().strip()
            dup_config = yaml.safe_load(f)
        dup_config["ppo"]["total_timesteps"] = args.timesteps
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            f.write(first_line + "\n")
            yaml.dump(dup_config, f, sort_keys=False)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Agent construction -------------------------------------------------
    if multi_mode:
        circuit_defs = [
            _build_circuit_def(cid, cfg, gs, args.seed, args.metric_slots)
            for cid, cfg, gs in parsed_circuits
        ]
        agent = MultiCircuitPPOGNNAgent(
            circuits=circuit_defs,
            config=config,
            hidden_dim=args.hidden_dim,
            gnn_layers=args.gnn_layers,
            init_checkpoint=args.init_checkpoint,
        )
        print(
            f"gnn multi ({phase}) | circuits: {circuit_tag} | "
            f"run: {os.path.basename(run_dir)} | seed: {args.seed}"
        )
    else:
        env = CircuitEnv(config_path=args.config, sequential=False)
        env.reset(seed=args.seed)
        graph_spec_path = _resolve_graph_spec_path(args.config, args.graph_spec)
        graph_spec = _build_graph_spec(graph_spec_path, env._param_names)
        adapter = GraphObservationAdapter(
            n_params=env._n_params,
            n_metrics=env._n_metrics,
            graph_spec=graph_spec,
            metric_slots=args.metric_slots,
        )
        agent = PPOGNNAgent(
            env=env,
            adapter=adapter,
            graph_spec=graph_spec,
            config=config,
            hidden_dim=args.hidden_dim,
            gnn_layers=args.gnn_layers,
            init_checkpoint=args.init_checkpoint,
        )
        print(f"gnn ({phase}) | run: {os.path.basename(run_dir)} | seed: {args.seed}")

    if args.init_checkpoint:
        print(f"Transfer init from: {args.init_checkpoint}")
    if args.freeze_encoder_steps > 0:
        print(f"Encoder frozen for first {args.freeze_encoder_steps} timesteps")
    print(f"Checkpoint dir: {run_dir}/")
    print()

    total = args.timesteps or int(config["ppo"]["total_timesteps"])
    cb = make_callback(run_dir, circuit_name)
    agent.train(
        total_timesteps=total,
        callback=cb,
        freeze_encoder_steps=args.freeze_encoder_steps,
    )
    cb.close()

    checkpoint_path = os.path.join(run_dir, "model.pt")
    agent.save(checkpoint_path)
    print(f"\nSaved checkpoint to {checkpoint_path}")
    print(f"Metrics log: {os.path.join(run_dir, 'metrics.csv')}")


if __name__ == "__main__":
    main()
