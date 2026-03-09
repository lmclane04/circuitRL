#!/usr/bin/env bash
# Full test: train + evaluate all 5 agents on all 8 circuits.
# Uses the full timesteps from each config file.
# Results go to runs/<agent>_<circuit>_seed<SEED>/
# Generates per-agent plots, per-circuit comparison figures, and a grand summary.
set -euo pipefail
cd "$(dirname "$0")"

export MPLBACKEND=Agg  # non-interactive plotting

SEED=0
EVAL_EPISODES=50

CONFIGS=(
    "circuitrl/configs/cs_amp.yaml"
    "circuitrl/configs/opamp.yaml"
    "circuitrl/configs/cascaded_amp.yaml"
    "circuitrl/configs/folded_cascode_ota.yaml"
    "circuitrl/configs/buck_converter.yaml"
    "circuitrl/configs/ldo.yaml"
    "circuitrl/configs/two_stage_tia.yaml"
    "circuitrl/configs/three_stage_tia.yaml"
)

AGENTS=("ppo" "ppo_non_shared" "ppo-seq" "grpo" "bo")

total_runs=$(( ${#CONFIGS[@]} * ${#AGENTS[@]} ))
run_num=0

echo "=========================================="
echo " Full Test Suite"
echo " ${#AGENTS[@]} agents x ${#CONFIGS[@]} circuits = ${total_runs} runs"
echo "=========================================="
echo ""

for config in "${CONFIGS[@]}"; do
    circuit=$(basename "$config" .yaml)

    for agent in "${AGENTS[@]}"; do
        run_num=$((run_num + 1))
        run_name="${agent}_${circuit}_seed${SEED}"

        echo "=========================================="
        echo " [${run_num}/${total_runs}] ${agent} on ${circuit}"
        echo "=========================================="

        # Remove stale run dir so train.py doesn't auto-increment
        if [ -d "runs/${run_name}" ]; then
            echo " Removing stale runs/${run_name}..."
            rm -rf "runs/${run_name}"
        fi

        # Train
        echo " Training..."
        python3 train.py --agent "$agent" --config "$config" --seed "$SEED" \
            --run-name "$run_name"
        echo ""

        # Evaluate
        echo " Evaluating (${EVAL_EPISODES} episodes)..."
        python3 evaluate.py --run-dir "runs/${run_name}" --episodes "$EVAL_EPISODES" \
            --seed "$SEED" --agent "$agent"
        echo ""

        # Full spec pool eval
        echo " Evaluating on full spec pool..."
        python3 evaluate_all_specs.py --run-dir "runs/${run_name}" \
            --seed "$SEED" --agent "$agent"
        echo ""

        # Per-agent eval plot
        if [ -f "runs/${run_name}/eval_results.csv" ]; then
            echo " Plotting eval results..."
            python3 plot.py --run-dir "runs/${run_name}" --eval
            echo ""
        fi
    done

    # ── Per-circuit comparison: training curves overlay ────────────────────
    echo " Generating comparison plots for ${circuit}..."
    circuit_run_dirs=()
    for agent in "${AGENTS[@]}"; do
        circuit_run_dirs+=("runs/${agent}_${circuit}_seed${SEED}")
    done
    python3 plot.py --run-dir "${circuit_run_dirs[@]}"
    echo ""
done

echo "=========================================="
echo " All ${total_runs} runs complete."
echo " Generating summary figures..."
echo "=========================================="

# ── Grand summary figure: success rates heatmap + bar charts ──────────────
python3 -c "
import os, sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

agents = '${AGENTS[*]}'.split()
configs = '${CONFIGS[*]}'.split()
circuits = [os.path.basename(c).replace('.yaml','') for c in configs]
seed = ${SEED}

# Build success rate matrix (circuits x agents)
matrix = np.full((len(circuits), len(agents)), np.nan)
reward_matrix = np.full((len(circuits), len(agents)), np.nan)

for i, circuit in enumerate(circuits):
    for j, agent in enumerate(agents):
        rd = f'runs/{agent}_{circuit}_seed{seed}'
        summary = os.path.join(rd, 'eval_summary.txt')
        csv_path = os.path.join(rd, 'eval_results.csv')
        if os.path.exists(summary):
            with open(summary) as f:
                for line in f:
                    if 'Success rate' in line:
                        # Parse 'Success rate:      372/396  (93.9%)'
                        pct = line.split('(')[1].split('%')[0]
                        matrix[i, j] = float(pct)
                        break
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            reward_matrix[i, j] = df['total_reward'].mean()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 1) Success rate heatmap
im = axes[0].imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
axes[0].set_xticks(range(len(agents)))
axes[0].set_xticklabels(agents, rotation=45, ha='right')
axes[0].set_yticks(range(len(circuits)))
axes[0].set_yticklabels(circuits)
axes[0].set_title('Success Rate (%)')
for i in range(len(circuits)):
    for j in range(len(agents)):
        val = matrix[i, j]
        if not np.isnan(val):
            color = 'white' if val < 50 else 'black'
            axes[0].text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=8, color=color)
fig.colorbar(im, ax=axes[0], shrink=0.8)

# 2) Grouped bar chart: mean reward per agent per circuit
x = np.arange(len(circuits))
width = 0.15
colors = ['#4C72B0','#55A868','#C44E52','#8172B2','#CCB974']
for j, agent in enumerate(agents):
    vals = reward_matrix[:, j]
    vals_clean = np.where(np.isnan(vals), 0, vals)
    axes[1].bar(x + j*width - width*2, vals_clean, width, label=agent, color=colors[j], alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels(circuits, rotation=45, ha='right')
axes[1].set_ylabel('Mean Reward')
axes[1].set_title('Mean Eval Reward by Agent')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, axis='y')

fig.suptitle('CircuitRL — Full Test Suite Summary', fontsize=14)
fig.tight_layout()
fig.savefig('runs/full_suite_summary.png', dpi=150)
print('Saved summary figure to runs/full_suite_summary.png')

# Per-circuit comparison bar charts
for i, circuit in enumerate(circuits):
    fig2, ax = plt.subplots(figsize=(8, 5))
    rates = [matrix[i, j] if not np.isnan(matrix[i, j]) else 0 for j in range(len(agents))]
    bars = ax.bar(agents, rates, color=colors, alpha=0.85)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'{circuit} — Agent Comparison (Full Spec Pool)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, f'{rate:.0f}%', ha='center', va='bottom', fontsize=9)
    fig2.tight_layout()
    fig2.savefig(f'runs/{circuit}_agent_comparison.png', dpi=150)
    print(f'Saved runs/{circuit}_agent_comparison.png')
    plt.close(fig2)
"

# Print summary table
echo ""
echo "Summary of results:"
echo "-------------------"
for config in "${CONFIGS[@]}"; do
    circuit=$(basename "$config" .yaml)
    echo ""
    echo "Circuit: ${circuit}"
    for agent in "${AGENTS[@]}"; do
        run_name="${agent}_${circuit}_seed${SEED}"
        summary="runs/${run_name}/eval_summary.txt"
        if [ -f "$summary" ]; then
            rate=$(grep "Success rate" "$summary" | head -1)
            echo "  ${agent}: ${rate}"
        else
            echo "  ${agent}: (no summary found)"
        fi
    done
done

echo ""
echo "=========================================="
echo " Figures:"
echo "   Per-agent:  runs/<agent>_<circuit>_seed${SEED}/eval_curves.png"
echo "   Per-circuit: runs/<circuit>_agent_comparison.png"
echo "   Training:    runs/<agent>_<circuit>_seed${SEED}/training_curves.png"
echo "   Grand:       runs/full_suite_summary.png"
echo "=========================================="
