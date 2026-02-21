"""Plot training curves from one or more CircuitRL runs."""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


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


def main():
    parser = argparse.ArgumentParser(description="Plot CircuitRL training curves")
    parser.add_argument("--run-dir", nargs="+", required=True,
                        help="Path(s) to run directories containing metrics.csv")
    args = parser.parse_args()

    plot_runs(args.run_dir)


if __name__ == "__main__":
    main()
