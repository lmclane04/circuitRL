"""Evaluate graph-based PPO checkpoints created by train_gnn_transfer.py."""

import argparse
import csv
import os

import numpy as np
import torch
import yaml

from circuitrl.agents.ppo_gnn_agent import GraphActorCritic, GraphObservationAdapter, GraphSpec
from circuitrl.envs.circuit_env import CircuitEnv


def find_original_config(run_dir: str) -> str | None:
    """Try to locate the matching config in circuitrl/configs/ via first-line comment."""
    run_config = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(run_config):
        return None
    with open(run_config) as f:
        first_line = f.readline().strip()

    configs_dir = os.path.join("circuitrl", "configs")
    if os.path.isdir(configs_dir):
        for name in os.listdir(configs_dir):
            if not name.endswith(".yaml"):
                continue
            path = os.path.join(configs_dir, name)
            with open(path) as f:
                if f.readline().strip() == first_line:
                    return path
    return None


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


def _spec_met(metric_val: float, target: float, tolerance: float, direction: str) -> bool:
    if direction == "max":
        return metric_val >= target - tolerance
    if direction == "min":
        return metric_val <= target + tolerance
    return abs(metric_val - target) <= tolerance


def _run_episode(env, adapter, network, adj, param_to_node, param_to_slot, node_type_ids, target_idx: int | None = None):
    if target_idx is None:
        obs, info = env.reset()
    else:
        obs, info = env.reset_with_target_idx(target_idx=target_idx)
    episode_targets = info["targets"]
    steps = []
    total_reward = 0.0

    for _ in range(env._max_steps):
        node_obs, spec_ctx = adapter.transform(obs)
        node_obs_t = torch.tensor(node_obs, dtype=torch.float32).unsqueeze(0)
        spec_ctx_t = torch.tensor(spec_ctx, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = network(node_obs_t, adj, param_to_node, param_to_slot, node_type_ids, spec_ctx_t)
            action = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps.append({
            "step": len(steps) + 1,
            "action": action.astype(int).tolist(),
            "reward": reward,
            "params": info.get("params", {}),
            "metrics": info.get("metrics", {}),
        })
        if terminated or truncated:
            break

    return steps, total_reward, terminated, episode_targets


def main():
    parser = argparse.ArgumentParser(description="Evaluate graph-based PPO checkpoint")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Run directory that contains model.pt and config.yaml")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional config override (defaults to run_dir/config.yaml)")
    parser.add_argument("--graph-spec", type=str, default=None,
                        help="Optional graph spec override")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--all-specs", action="store_true",
                        help="Evaluate one episode per target in the current spec pool")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--spec_pool_test", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config or find_original_config(args.run_dir)
    if config_path is None:
        config_path = os.path.join(args.run_dir, "config.yaml")
        print(f"Warning: could not find original config, using {config_path}")
        print("  (spec/netlist relative paths may fail; pass --config to fix)")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    checkpoint_path = os.path.join(args.run_dir, "model.pt")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    meta = checkpoint.get("meta", {})

    env = CircuitEnv(config_path=config_path, sequential=False)
    env.reset(seed=args.seed)
    if args.spec_pool_test:
        env.set_spec_pool(args.spec_pool_test)

    graph_spec_path = _resolve_graph_spec_path(config_path, args.graph_spec)
    graph_spec = _build_graph_spec(graph_spec_path, env._param_names)

    adapter = GraphObservationAdapter(
        n_params=env._n_params,
        n_metrics=env._n_metrics,
        graph_spec=graph_spec,
        metric_slots=int(meta.get("metric_slots", 4)),
    )
    network = GraphActorCritic(
        node_feature_dim=int(meta.get("node_feature_dim", adapter.node_feature_dim)),
        actions_per_param=int(meta.get("actions_per_param", env.n_actions_per_param)),
        max_params_per_component=int(meta.get("max_params_per_component", graph_spec.max_params_per_component)),
        metric_slots=int(meta.get("metric_slots", 4)),
        hidden_dim=int(meta.get("hidden_dim", 128)),
        gnn_layers=int(meta.get("gnn_layers", 2)),
    )
    network.load_state_dict(checkpoint["network"])
    network.eval()

    spec_names = list(config["target_specs"].keys())
    tolerances = {name: float(spec["tolerance"]) for name, spec in config["target_specs"].items()}
    directions = {name: spec.get("direction", "equal") for name, spec in config["target_specs"].items()}

    n_eval = len(env._spec_pool) if args.all_specs else args.episodes

    print(f"Loaded ppo_gnn agent from {args.run_dir}")
    print(f"Circuit: {_extract_circuit_name(config_path)}")
    mode = "full spec pool" if args.all_specs else f"{n_eval} episodes"
    print(f"Evaluating {mode} (seed={args.seed})")
    print()

    param_to_node = torch.tensor(graph_spec.param_to_node, dtype=torch.long)
    param_to_slot = torch.tensor(graph_spec.param_to_slot, dtype=torch.long)
    node_type_ids = torch.tensor(graph_spec.node_type_ids, dtype=torch.long)

    all_rewards = []
    all_successes = []
    all_steps = []
    spec_successes = {name: [] for name in spec_names}
    csv_rows = []
    log_lines = []

    def _log(line: str = ""):
        print(line)
        log_lines.append(line)

    for ep in range(n_eval):
        steps, total_reward, success, episode_targets = _run_episode(
            env,
            adapter,
            network,
            graph_spec.adjacency,
            param_to_node,
            param_to_slot,
            node_type_ids,
            target_idx=ep if args.all_specs else None,
        )
        all_rewards.append(total_reward)
        all_successes.append(success)
        all_steps.append(len(steps))

        final_metrics = steps[-1].get("metrics", {})
        per_spec = {}
        for name in spec_names:
            if name in final_metrics:
                met = _spec_met(final_metrics[name], episode_targets[name], tolerances[name], directions[name])
                per_spec[name] = met
                spec_successes[name].append(met)

        row = {"total_reward": total_reward, "success": int(success), "n_steps": len(steps)}
        for name in spec_names:
            row[f"target_{name}"] = episode_targets.get(name, float("nan"))
            row[f"final_{name}"] = final_metrics.get(name, float("nan"))
        csv_rows.append(row)

        if args.verbose:
            for s in steps:
                _log(f"  step {s['step']:>3d}  action={s['action']}  reward={s['reward']:>8.3f}")

        targets_str = "  ".join(f"{n}={episode_targets[n]:.3g}" for n in spec_names)
        _log(
            f"Episode {ep + 1:>3d}: steps={len(steps):>3d}  "
            f"reward={total_reward:>8.3f}  "
            f"{'SUCCESS' if success else 'FAIL   '}  "
            f"targets: [{targets_str}]"
        )

        for name in spec_names:
            if name in final_metrics:
                val = final_metrics[name]
                tgt = episode_targets[name]
                tol = tolerances[name]
                met = per_spec.get(name, False)
                _log(
                    f"    {name}: {val:.4g}  (target: {tgt:.4g}, tol: {tol:.3g})  "
                    f"[{'OK  ' if met else 'MISS'}]"
                )
        _log()

    if csv_rows:
        csv_path = os.path.join(args.run_dir, "eval_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved eval results to {csv_path}\n")

    n = n_eval
    _log("=" * 60)
    _log(f"Summary over {n} episodes:")
    _log(f"  Success rate:      {sum(all_successes)}/{n}  ({100*sum(all_successes)/n:.1f}%)")
    _log(f"  Mean total reward: {sum(all_rewards)/n:.3f}")
    _log(f"  Mean episode len:  {sum(all_steps)/n:.1f}")
    _log("  Per-spec success rates:")
    for name in spec_names:
        if spec_successes[name]:
            rate = sum(spec_successes[name]) / len(spec_successes[name])
            _log(f"    {name}: {100*rate:.1f}%")

    txt_path = os.path.join(args.run_dir, "eval_summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Saved eval summary to {txt_path}")


if __name__ == "__main__":
    main()
