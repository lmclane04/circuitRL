#!/usr/bin/env python3
"""Compare PPO, BO, and GRPO on the CS amp circuit sizing task.

Each agent is given the same simulator-call budget so the comparison is fair.
One simulator call = one NGSpice execution (true for all three agents).

When specs_range is configured in the YAML, targets are randomised each episode
and quality is measured relative to each episode's own targets (so the score
is always "how close did the agent get to whatever it was asked to achieve").

Tracks per-agent:
  - Best solution quality found (negative mean relative % error)
  - Per-metric % errors at the best point
  - Wall time
  - Running best-so-far curve (for plotting)

Usage:
  python compare.py                                    # default 3000-call budget
  python compare.py --n-evals 1000 --seeds 3
  python compare.py --n-evals 5000 --agents ppo bo    # subset of agents
  python compare.py --n-evals 3000 --no-plot
"""

import argparse
import time

import numpy as np
import yaml

from circuitrl.envs.circuit_env import CircuitEnv

CONFIG_PATH = "circuitrl/configs/cs_amp.yaml"


# ── per-agent runner ──────────────────────────────────────────────────────────

def run_agent(agent_name: str, config: dict, n_evals: int, seed: int,
              n_eval_eps: int = 30) -> dict:
    """
    Run one agent for exactly n_evals simulator calls, then evaluate with
    n_eval_eps held-out episodes.

    Returns a result dict:
        agent, seed, elapsed, sim_calls,
        best_quality, best_metrics, best_ep_targets,  ← training best
        eval_quality_mean, eval_quality_std,           ← evaluation aggregate
        eval_pct_errors, eval_specs_met_rate,          ← evaluation aggregate
        pct_errors, specs_met,                         ← kept for compat
        history
    where history = list of (sim_calls_so_far, best_quality_so_far).
    """
    np.random.seed(seed)

    # Default (fixed) targets from config — used as column headers in the table
    # and as fallback when an episode doesn't report its own targets.
    default_targets    = {m: float(config["target_specs"][m]["value"])
                          for m in config["target_specs"]}
    default_tolerances = {m: float(config["target_specs"][m]["tolerance"])
                          for m in config["target_specs"]}

    env = CircuitEnv(config_path=CONFIG_PATH)

    best_quality  = -np.inf
    best_metrics: dict = {}
    best_ep_targets: dict = default_targets.copy()   # targets for the best episode
    history: list[tuple[int, float]] = []
    sim_calls = [0]

    def _quality(metrics: dict, ep_targets: dict) -> float:
        """Negative mean relative error relative to the episode's own targets."""
        errs = [abs(metrics[m] - ep_targets[m]) / abs(ep_targets[m])
                for m in ep_targets if m in metrics and ep_targets[m] != 0]
        return -float(np.mean(errs)) if errs else -np.inf

    def callback(timesteps_done, episode_stats, loss_stats):
        nonlocal best_quality, best_metrics, best_ep_targets
        for ep in episode_stats:
            # Accumulate simulator calls (length=1 for BO, episode length for RL)
            sim_calls[0] += ep.get("length", 1)
            m  = ep.get("final_metrics", {})
            t  = ep.get("ep_targets", default_targets)
            if m and t:
                q = _quality(m, t)
            else:
                # Simulation failed at terminal step — use mean per-step reward
                # (reward = -mean_rel_error already relative to episode targets)
                ep_len = max(ep.get("length", 1), 1)
                q = ep.get("reward", -np.inf) / ep_len
            if q > best_quality:
                best_quality    = q
                if m:
                    best_metrics    = m.copy()
                    best_ep_targets = t.copy()
            history.append((sim_calls[0], best_quality))

    # ── instantiate ───────────────────────────────────────────────────────────
    if agent_name == "ppo":
        from circuitrl.agents.ppo_agent import PPOAgent
        agent = PPOAgent(env, config)
    elif agent_name == "bo":
        from circuitrl.agents.bo_agent import BOAgent
        agent = BOAgent(env, config)
    elif agent_name == "grpo":
        from circuitrl.agents.grpo_agent import GRPOAgent
        agent = GRPOAgent(env, config)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    # ── train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    agent.train(total_timesteps=n_evals, callback=callback)
    elapsed = time.time() - t0

    # ── post-training evaluation over n_eval_eps independent episodes ─────────
    eval_eps = agent.evaluate(n_eval_eps)

    eval_qualities = []
    for ep in eval_eps:
        m = ep.get("final_metrics", {})
        t = ep.get("ep_targets", default_targets)
        if m and t:
            q = _quality(m, t)
        else:
            ep_len = max(ep.get("length", 1), 1)
            q = ep.get("reward", -np.inf) / ep_len
        eval_qualities.append(q)

    eval_quality_mean = float(np.mean(eval_qualities)) if eval_qualities else -np.inf
    eval_quality_std  = float(np.std(eval_qualities))  if eval_qualities else 0.0

    # Per-metric mean %err across eval episodes (relative to each ep's own target)
    eval_pct_errors: dict[str, float] = {}
    for metric in default_targets:
        errs = []
        for ep in eval_eps:
            mv = ep.get("final_metrics", {}).get(metric)
            tv = ep.get("ep_targets", default_targets).get(metric)
            if mv is not None and tv:
                errs.append(100.0 * abs(mv - tv) / abs(tv))
        eval_pct_errors[metric] = float(np.mean(errs)) if errs else float("nan")

    # specs_met_rate: fraction of eval episodes where ALL specs were within tolerance
    specs_met_count = 0
    for ep in eval_eps:
        m = ep.get("final_metrics", {})
        t = ep.get("ep_targets", default_targets)
        if m and t and all(
            abs(m.get(k, np.inf) - t.get(k, np.inf)) <= default_tolerances[k]
            for k in default_tolerances
        ):
            specs_met_count += 1
    eval_specs_met_rate = specs_met_count / len(eval_eps) if eval_eps else 0.0

    # ── training-best quality (kept for the learning-curve plot) ─────────────
    pct_errors = {
        m: 100.0 * abs(best_metrics.get(m, np.nan) - best_ep_targets.get(m, np.nan))
               / abs(best_ep_targets.get(m, 1.0))
        for m in default_targets
    }
    specs_met = all(
        abs(best_metrics.get(m, np.inf) - best_ep_targets.get(m, np.inf))
            <= default_tolerances[m]
        for m in default_targets
    )

    return {
        "agent":               agent_name,
        "seed":                seed,
        "elapsed":             elapsed,
        "sim_calls":           sim_calls[0],
        "best_quality":        best_quality,
        "best_metrics":        best_metrics,
        "best_ep_targets":     best_ep_targets,
        "pct_errors":          pct_errors,
        "specs_met":           specs_met,
        "eval_quality_mean":   eval_quality_mean,
        "eval_quality_std":    eval_quality_std,
        "eval_pct_errors":     eval_pct_errors,
        "eval_specs_met_rate": eval_specs_met_rate,
        "history":             history,
    }


# ── summary printing ──────────────────────────────────────────────────────────

def print_summary(all_results: list[dict], targets: dict):
    metric_names = list(targets.keys())
    col_w = 10

    # ── Training-best summary (one row per seed) ───────────────────────────────
    hdr_metrics = "  ".join(f"{'%err_' + m:>{col_w}}" for m in metric_names)
    print(f"\n── Training best (single best episode during training) ──")
    print(f"{'Agent':<6}  {'Seed':>4}  {'Calls':>6}  {'Time':>7}  "
          f"{'Best Q':>8}  {hdr_metrics}  {'Specs?':>6}")
    print("-" * (6 + 4 + 6 + 7 + 8 + len(metric_names) * (col_w + 2) + 6 + 30))

    for r in all_results:
        errs = "  ".join(
            f"{r['pct_errors'].get(m, float('nan')):>{col_w}.1f}%"
            for m in metric_names
        )
        print(f"{r['agent']:<6}  {r['seed']:>4}  {r['sim_calls']:>6}  "
              f"{r['elapsed']:>6.1f}s  {r['best_quality']:>8.4f}  "
              f"{errs}  {'✓' if r['specs_met'] else '✗':>6}")

    # ── Evaluation summary (averaged across eval episodes) ─────────────────────
    hdr_eval = "  ".join(f"{'eval%_' + m:>{col_w}}" for m in metric_names)
    print(f"\n── Evaluation (mean over held-out episodes) ──")
    print(f"{'Agent':<6}  {'Seed':>4}  {'Eval Q (mean±std)':>20}  "
          f"{hdr_eval}  {'Specs%':>7}")
    print("-" * (6 + 4 + 22 + len(metric_names) * (col_w + 2) + 7 + 20))

    for r in all_results:
        q_str = f"{r['eval_quality_mean']:>8.4f}±{r['eval_quality_std']:.4f}"
        errs  = "  ".join(
            f"{r['eval_pct_errors'].get(m, float('nan')):>{col_w}.1f}%"
            for m in metric_names
        )
        met_pct = f"{100 * r['eval_specs_met_rate']:.0f}%"
        print(f"{r['agent']:<6}  {r['seed']:>4}  {q_str:>20}  {errs}  {met_pct:>7}")

    # ── Per-agent averages across seeds (if multiple seeds) ────────────────────
    agents = sorted({r["agent"] for r in all_results})
    if len(all_results) > len(agents):
        print("\nPer-agent averages across seeds:")
        for ag in agents:
            rows = [r for r in all_results if r["agent"] == ag]
            q_mean  = np.mean([r["eval_quality_mean"]   for r in rows])
            q_std   = np.std( [r["eval_quality_mean"]   for r in rows])
            t_mean  = np.mean([r["elapsed"]              for r in rows])
            met_avg = np.mean([r["eval_specs_met_rate"]  for r in rows])
            print(f"  {ag:<6}  eval_Q={q_mean:.4f}±{q_std:.4f}  "
                  f"time={t_mean:.1f}s  specs_met={met_avg*100:.0f}%")


# ── optional plot ─────────────────────────────────────────────────────────────

def plot_curves(all_results: list[dict], n_evals: int):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available — skipping plot)")
        return

    agent_styles = {"ppo": ("tab:blue", "-"), "bo": ("tab:orange", "--"),
                    "grpo": ("tab:green", "-.")}
    fig, ax = plt.subplots(figsize=(8, 5))

    agents = sorted({r["agent"] for r in all_results})
    for ag in agents:
        rows = [r for r in all_results if r["agent"] == ag]
        color, ls = agent_styles.get(ag, ("gray", ":"))

        if len(rows) == 1:
            xs, ys = zip(*rows[0]["history"]) if rows[0]["history"] else ([0], [-1])
            ax.plot(xs, ys, color=color, ls=ls, label=ag)
        else:
            # Interpolate all seeds to a common x grid then plot mean ± std
            x_max = max(r["history"][-1][0] for r in rows if r["history"])
            xs_common = np.linspace(0, x_max, 300)
            interp_ys = []
            for r in rows:
                if not r["history"]:
                    continue
                hx, hy = zip(*r["history"])
                interp_ys.append(np.interp(xs_common, hx, hy))
            ys_arr  = np.array(interp_ys)
            ys_mean = ys_arr.mean(0)
            ys_std  = ys_arr.std(0)
            ax.plot(xs_common, ys_mean, color=color, ls=ls, label=ag)
            ax.fill_between(xs_common, ys_mean - ys_std, ys_mean + ys_std,
                            color=color, alpha=0.15)

    ax.set_xlabel("Simulator calls")
    ax.set_ylabel("Best quality (−mean rel. error)")
    ax.set_title(f"Agent comparison — CS Amp ({n_evals} sim-call budget)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = f"comparison_{n_evals}evals.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nLearning curve saved to {out}")
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare PPO / BO / GRPO agents")
    parser.add_argument("--n-evals",  type=int,   default=50000,
                        help="Simulator-call budget per agent per seed (default: 50000)")
    parser.add_argument("--seeds",    type=int,   default=1,
                        help="Number of random seeds to average over (default: 1)")
    parser.add_argument("--agents",   nargs="+",  default=["ppo", "bo", "grpo"],
                        choices=["ppo", "bo", "grpo"],
                        help="Which agents to run (default: all three)")
    parser.add_argument("--config",   type=str,   default=CONFIG_PATH)
    parser.add_argument("--no-plot",  action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    default_targets = {m: float(config["target_specs"][m]["value"])
                       for m in config["target_specs"]}
    specs_range = config.get("specs_range", {})

    print(f"Budget: {args.n_evals} simulator calls per agent")
    print(f"Agents: {args.agents}   Seeds: {args.seeds}")
    if specs_range:
        range_str = ", ".join(
            f"{m}=[{specs_range[m]['min']}, {specs_range[m]['max']}]"
            for m in specs_range
        )
        print(f"Spec randomisation: {range_str}")
        print("(Quality = −mean_rel_error relative to each episode's sampled target)")
    else:
        print(f"Fixed targets: {', '.join(f'{m}={v}' for m, v in default_targets.items())}")
    print()

    all_results: list[dict] = []

    for agent_name in args.agents:
        for seed in range(args.seeds):
            print(f"Running {agent_name.upper()} seed={seed} …", flush=True)
            result = run_agent(agent_name, config, args.n_evals, seed)
            all_results.append(result)
            eval_errs = "  ".join(
                f"{m}: {result['eval_pct_errors'].get(m, float('nan')):.1f}%"
                for m in default_targets
            )
            met_pct = f"{100 * result['eval_specs_met_rate']:.0f}%"
            print(f"  done in {result['elapsed']:.1f}s | "
                  f"eval_Q={result['eval_quality_mean']:.4f}±{result['eval_quality_std']:.4f} | "
                  f"{eval_errs} | specs_met={met_pct}")

    print_summary(all_results, default_targets)

    if not args.no_plot:
        plot_curves(all_results, args.n_evals)


if __name__ == "__main__":
    main()
