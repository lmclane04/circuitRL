"""
Generate a pool of achievable target specifications by randomly sampling
parameter combinations and simulating them with NGSpice.

The resulting JSON file contains a list of metric dicts — each entry is a
set of specifications a real circuit can achieve, so every training target
is guaranteed to be physically reachable.

Usage:
    python generate_specs.py --config circuitrl/configs/cs_amp.yaml --n 500
    python generate_specs.py --config circuitrl/configs/opamp.yaml --n 1000

The output is saved next to the config file as <circuit>_specs_pool.json.
"""
import argparse
import json
import os

import numpy as np
import yaml

from circuitrl.simulators.ngspice_runner import NGSpiceRunner


def generate(config_path: str, n_samples: int, seed: int = 0):
    config_dir = os.path.dirname(os.path.abspath(config_path))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Build parameter arrays
    param_names = list(cfg["parameters"].keys())
    param_arrays = []
    for p in cfg["parameters"].values():
        lo, hi, step = float(p["min"]), float(p["max"]), float(p["step"])
        param_arrays.append(np.arange(lo, hi + step * 0.5, step))
    max_indices = np.array([len(a) - 1 for a in param_arrays])

    constraints = cfg.get("constraints", [])
    metric_names = list(cfg["target_specs"].keys())

    netlist_rel = cfg.get("netlist", "../envs/netlist_template.sp")
    runner = NGSpiceRunner(
        os.path.normpath(os.path.join(config_dir, netlist_rel)),
        timeout=cfg["env"]["sim_timeout"],
        expected_metrics=tuple(metric_names),
    )

    rng = np.random.default_rng(seed)
    pool = []
    failed = 0

    print(f"Generating {n_samples} spec targets from random simulations...")
    for i in range(n_samples):
        # Sample random parameter indices
        indices = np.array([rng.integers(0, max_idx + 1) for max_idx in max_indices])
        params_si = np.array([arr[idx] for arr, idx in zip(param_arrays, indices)])

        # Skip constraint-violating combinations
        local_vars = dict(zip(param_names, params_si.tolist()))
        if not all(eval(expr, {"__builtins__": {}}, local_vars) for expr in constraints):
            failed += 1
            continue

        param_dict = {name: f"{val:.6e}" for name, val in zip(param_names, params_si)}
        result = runner.run(param_dict)
        if result is None:
            failed += 1
            continue

        pool.append({m: result[m] for m in metric_names})

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_samples} done  |  {len(pool)} valid  |  {failed} failed/invalid")

    print(f"\nDone: {len(pool)} valid spec targets from {n_samples} samples ({failed} failed/invalid)\n")

    if not pool:
        print("No valid results — check your netlist/config.")
        return

    # Print distribution summary
    for name in metric_names:
        vals = [r[name] for r in pool]
        print(f"  {name:<20}  min={min(vals):.4g}  p25={np.percentile(vals,25):.4g}"
              f"  p50={np.percentile(vals,50):.4g}  p75={np.percentile(vals,75):.4g}"
              f"  max={max(vals):.4g}")

    circuit_name = os.path.splitext(os.path.basename(config_path))[0]
    out_path = os.path.join(config_dir, f"{circuit_name}_specs_pool.json")
    with open(out_path, "w") as f:
        json.dump(pool, f, indent=2)
    print(f"\nSaved {len(pool)} spec targets to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate achievable spec target pool")
    parser.add_argument("--config", required=True, help="Path to circuit YAML config")
    parser.add_argument("--n", type=int, default=500,
                        help="Number of random parameter combinations to simulate (default: 500)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()
    generate(args.config, args.n, args.seed)


if __name__ == "__main__":
    main()
