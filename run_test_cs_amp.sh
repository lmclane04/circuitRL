#!/usr/bin/env bash
# Quick smoke test: train + evaluate all 5 agents on CS amp only.
# Uses reduced timesteps for fast iteration.
# Outputs per-agent plots + a comparison figure.
set -euo pipefail
cd "$(dirname "$0")"

export MPLBACKEND=Agg  # non-interactive plotting

CONFIG="circuitrl/configs/cs_amp.yaml"
SEED=0
EVAL_EPISODES=20

# Reduced timesteps for quick testing
PPO_STEPS=5000
GRPO_STEPS=5000
BO_STEPS=50

AGENTS=("ppo" "ppo_non_shared" "ppo-seq" "grpo" "bo")
TIMESTEPS=($PPO_STEPS $PPO_STEPS $PPO_STEPS $GRPO_STEPS $BO_STEPS)

echo "=========================================="
echo " CS Amp Smoke Test — all agents"
echo "=========================================="
echo ""

RUN_DIRS=()

for i in "${!AGENTS[@]}"; do
    agent="${AGENTS[$i]}"
    steps="${TIMESTEPS[$i]}"
    run_name="${agent}_cs_amp_test"

    # Remove stale run dir so train.py doesn't auto-increment
    if [ -d "runs/${run_name}" ]; then
        echo " Removing stale runs/${run_name}..."
        rm -rf "runs/${run_name}"
    fi

    echo "------------------------------------------"
    echo " Training: ${agent}  (${steps} timesteps)"
    echo "------------------------------------------"
    python3 train.py --agent "$agent" --config "$CONFIG" --seed "$SEED" \
        --timesteps "$steps" --run-name "$run_name" --checkpoint-best
    echo ""

    echo " Evaluating: ${agent}"
    echo "------------------------------------------"
    # Use best checkpoint if available (skip for BO — it always uses model.pkl)
    CKPT_FLAG=""
    if [ "$agent" != "bo" ] && [ -f "runs/${run_name}/model_best.pt" ]; then
        CKPT_FLAG="--checkpoint model_best.pt"
        echo " (using model_best.pt)"
    fi
    python3 evaluate.py --run-dir "runs/${run_name}" --episodes "$EVAL_EPISODES" \
        --seed "$SEED" --agent "$agent" $CKPT_FLAG
    echo ""

    # Per-agent eval plot
    if [ -f "runs/${run_name}/eval_results.csv" ]; then
        echo " Plotting eval results for ${agent}..."
        python3 plot.py --run-dir "runs/${run_name}" --eval
        echo ""
    fi

    RUN_DIRS+=("runs/${run_name}")
done

# ── Comparison figures across all agents ──────────────────────────────────
echo "------------------------------------------"
echo " Generating comparison figures..."
echo "------------------------------------------"

# Training curves overlay (all agents on one plot)
python3 plot.py --run-dir "${RUN_DIRS[@]}"

# Agent comparison bar chart
python3 -c "
import os, sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

agents = '${AGENTS[*]}'.split()
run_dirs = '${RUN_DIRS[*]}'.split()

# --- Success rate comparison ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1) Success rate bar chart
success_rates = []
for rd in run_dirs:
    csv = os.path.join(rd, 'eval_results.csv')
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        success_rates.append(100 * df['success'].mean())
    else:
        success_rates.append(0)

bars = axes[0].bar(agents, success_rates, color=['#4C72B0','#55A868','#C44E52','#8172B2','#CCB974'], alpha=0.85)
axes[0].set_ylim(0, 105)
axes[0].set_ylabel('Success Rate (%)')
axes[0].set_title('Overall Success Rate')
axes[0].grid(True, alpha=0.3, axis='y')
for bar, rate in zip(bars, success_rates):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, f'{rate:.0f}%', ha='center', va='bottom', fontsize=9)

# 2) Training reward curves overlay (all agents, colored with legend)
colors = ['#4C72B0','#55A868','#C44E52','#8172B2','#CCB974']
for rd, agent, color in zip(run_dirs, agents, colors):
    csv = os.path.join(rd, 'metrics.csv')
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        axes[1].plot(df['timestep'], df['mean_reward'], label=agent, color=color, alpha=0.8)
axes[1].set_xlabel('Timesteps')
axes[1].set_ylabel('Mean Reward')
axes[1].set_title('Training Reward Curves')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# 3) Mean eval reward bar chart
mean_rewards = []
for rd in run_dirs:
    csv = os.path.join(rd, 'eval_results.csv')
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        mean_rewards.append(df['total_reward'].mean())
    else:
        mean_rewards.append(0)

axes[2].bar(agents, mean_rewards, color=colors, alpha=0.85)
axes[2].set_ylabel('Mean Reward')
axes[2].set_title('Mean Eval Reward')
axes[2].grid(True, alpha=0.3, axis='y')

fig.suptitle('CS Amp — Agent Comparison', fontsize=14)
fig.tight_layout()
fig.savefig('runs/cs_amp_comparison.png', dpi=150)
print('Saved comparison figure to runs/cs_amp_comparison.png')
"

echo ""
echo "=========================================="
echo " All CS amp tests complete."
echo " Figures saved in runs/*/"
echo "   - Per-agent: runs/<agent>_cs_amp_test/eval_curves.png"
echo "   - Comparison: runs/cs_amp_comparison.png"
echo "=========================================="
