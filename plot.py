"""Plot training curves or evaluation results from CircuitRL runs."""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


METRICS = [
    ("mean_reward", "Mean Reward"),
    ("mean_len", "Mean Episode Length"),
    ("policy_loss", "Policy Loss"),
    ("entropy", "Entropy"),
]


def plot_runs(run_dirs: list[str]):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for run_dir in run_dirs:
        csv_path = os.path.join(run_dir, "metrics.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        circuit = df["circuit"].iloc[0] if "circuit" in df.columns else ""
        run_label = os.path.basename(run_dir)
        label = f"{run_label} ({circuit})" if circuit else run_label

        for ax, (col, title) in zip(axes, METRICS):
            ax.plot(df["timestep"], df[col], label=label, alpha=0.8)
            ax.set_title(title)
            ax.set_xlabel("Timesteps")
            ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.legend()

    fig.suptitle("CircuitRL Training Curves", fontsize=14)
    fig.tight_layout()

    save_path = os.path.join(run_dirs[0], "training_curves.png")
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

    plt.show()


def plot_eval(run_dir: str):
    """Plot evaluation results from eval_results.csv."""
    csv_path = os.path.join(run_dir, "eval_results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run evaluate.py first.")
        return

    df = pd.read_csv(csv_path)

    # Load config to get spec names, directions, tolerances
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    spec_names = list(config["target_specs"].keys())
    directions = {n: config["target_specs"][n].get("direction", "equal") for n in spec_names}
    tolerances = {n: float(config["target_specs"][n]["tolerance"]) for n in spec_names}

    n_specs = len(spec_names)
    # Layout: n_specs scatter plots + 1 reward histogram + 1 success bar
    n_plots = n_specs + 2
    ncols = min(n_plots, 3)
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten().tolist()

    run_label = os.path.basename(run_dir)

    # Achieved-vs-target scatter per spec
    for i, name in enumerate(spec_names):
        ax = axes_flat[i]
        target_col = f"target_{name}"
        final_col = f"final_{name}"
        if target_col not in df.columns or final_col not in df.columns:
            ax.set_title(f"{name} (data missing)")
            continue

        targets = df[target_col].values
        achieved = df[final_col].values
        tol = tolerances[name]
        direction = directions[name]

        # Color by whether spec was met
        def _met(m, t):
            if direction == "max":
                return m >= t - tol
            elif direction == "min":
                return m <= t + tol
            else:
                return abs(m - t) <= tol

        colors = ["green" if _met(m, t) else "red" for m, t in zip(achieved, targets)]

        ax.scatter(targets, achieved, c=colors, alpha=0.7, s=50)

        # Diagonal reference line
        lo = min(targets.min(), achieved.min()) * 0.95
        hi = max(targets.max(), achieved.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1, label="achieved=target")

        ax.set_xlabel(f"Target {name}")
        ax.set_ylabel(f"Achieved {name}")
        ax.set_title(f"{name}  (dir={direction})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Reward histogram
    ax_rew = axes_flat[n_specs]
    ax_rew.hist(df["total_reward"], bins=20, color="steelblue", alpha=0.8, edgecolor="white")
    ax_rew.set_xlabel("Total Episode Reward")
    ax_rew.set_ylabel("Count")
    ax_rew.set_title("Reward Distribution")
    ax_rew.grid(True, alpha=0.3)

    # Per-spec success rate bar chart
    ax_bar = axes_flat[n_specs + 1]
    success_rates = []
    for name in spec_names:
        final_col = f"final_{name}"
        target_col = f"target_{name}"
        if final_col in df.columns and target_col in df.columns:
            tol = tolerances[name]
            direction = directions[name]
            met = [_met(m, t) for m, t in zip(df[final_col].values, df[target_col].values)]
            success_rates.append(100 * sum(met) / len(met))
        else:
            success_rates.append(0.0)

    bars = ax_bar.bar(spec_names, success_rates, color="steelblue", alpha=0.8)
    ax_bar.set_ylim(0, 105)
    ax_bar.set_ylabel("Success Rate (%)")
    ax_bar.set_title("Per-Spec Success Rates")
    ax_bar.grid(True, alpha=0.3, axis="y")
    for bar, rate in zip(bars, success_rates):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{rate:.0f}%", ha="center", va="bottom", fontsize=9)

    # Hide unused subplots
    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    overall_success = 100 * df["success"].sum() / len(df)
    fig.suptitle(f"Eval Results: {run_label}  |  {len(df)} episodes  |  "
                 f"Overall success: {overall_success:.1f}%", fontsize=12)
    fig.tight_layout()

    save_path = os.path.join(run_dir, "eval_curves.png")
    fig.savefig(save_path, dpi=150)
    print(f"Saved eval plot to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot CircuitRL training curves or eval results")
    parser.add_argument("--run-dir", nargs="+", required=True,
                        help="Path(s) to run directories containing metrics.csv / eval_results.csv")
    parser.add_argument("--eval", action="store_true",
                        help="Plot evaluation results (eval_results.csv) instead of training curves")
    args = parser.parse_args()

    if args.eval:
        if len(args.run_dir) > 1:
            print("Warning: --eval only supports a single run directory; using the first one.")
        plot_eval(args.run_dir[0])
    else:
        plot_runs(args.run_dir)


if __name__ == "__main__":
    main()
