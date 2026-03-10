# CircuitRL

RL for analog circuit sizing. An agent learns to size a circuit to hit target performance specs (gain, bandwidth, phase margin, power) using NGSpice simulation for a reward signal.

## Setup

```bash
# Create and activate conda env
conda create -n circuitrl python=3.10 -y
conda activate circuitrl

# Install NGSpice (macOS)
brew install ngspice

# Install Python dependencies
pip install -r requirements.txt
```

## Available Circuits

- Common-Source Amplifier: 3 parameters, 3 specs (`cs_amp.yaml`)
- Two-Stage Voltage Amplifier (Op-Amp): 10 parameters, 4 specs (`opamp.yaml`)
- Two-Stage Cascaded Common-Source Amplifier: 6 parameters, 3 specs (`cascaded_amp.yaml`)
- Folded Cascode Operational Transconductance Amplifier: 7 parameters, 4 specs (`folded_cascode_ota.yaml`)
- Buck Converter: 2 parameters, 2 specs (`buck_converter.yaml`)
- Two-Stage Transimpendance Amplifier: 7 parameters, 3 specs (`two_stage_tia.yaml`)
- Three-Stage Transimpendance Amplifier: 10 parameters, 3 specs (`three_stage_tia.yaml`)
- PMOS Low-Dropout Regulator: 6 parameters, 3 specs (`ldo.yaml`)

Configurations are inside `circuitrl/configs/`
There are structured as follows:

```yaml
netlist: ../envs/your_template.sp # relative path to the SPICE template

parameters:
  PARAM1:
    min: 1.0e-6
    max: 10.0e-6
    default: 5.0e-6
    step: 1.0e-6

target_spec_file: your_specs_pool.json

target_specs:
  metric_name:
    tolerance: 1.0
    direction: max

env:
  max_steps: 100
  sim_timeout: 30
  sequential: false
  action_deltas: [-1, 0, 1] # Change this to give the agent more flexible step sizes ie. [-2, -1, 0, 1, 2] 
  # note: ppo-seq is hardcoded to [-1, 0, 1]. 

ppo:
  learning_rate: 3.0e-4
  n_steps: 1024
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  total_timesteps: 100000
```


## Agents

### Flat PPO (`train.py`)
- `ppo`: shared-trunk PPO actor-critic with simultaneous parameter updates
- `ppo_non_shared`: separate actor/critic PPO, with simultaneous parameter updates
- `ppo-seq`: shared-trunk sequential PPO, one parameter update per step

### GNN PPO (`train_gnn_transfer.py`)

The flat PPO policy is tied to a fixed observation dimension, so it must be trained from scratch for every new circuit. The GNN agent replaces the MLP trunk with a graph neural network that encodes each circuit as a graph (nodes = components, edges = connections). Because the graph structure is supplied at runtime, a single set of weights can operate on any topology — a checkpoint trained on circuit A can be fine-tuned on circuit B, reusing the learned representations of transistors, resistors, and their relationships.

#### Graph representation

Each circuit is described by a graph spec YAML (see `circuitrl/configs/graph_specs/`). Components become nodes; electrical connections become edges. The graph is converted to a degree-normalized adjacency matrix `D⁻¹A` for message passing.

Each node carries a 14-dimensional feature vector:

```
[degree_scalar(1) | type_onehot(6) | model_features(5) | param_slots(2)] = 14 dims
```

| Field | Dims | Description |
|---|---|---|
| `degree_scalar` | 1 | Raw node degree / max degree. Encodes structural role (a bias current source with 1 connection scores lower than a diff-pair transistor with 3). Topology-derived, transfers across circuits — unlike the old YAML-order index. |
| `type_onehot` | 6 | One-hot over the global type vocabulary: `{capacitor, current, generic, nmos, pmos, resistor}`. Fixed width across ALL circuits so the node encoder weight matrix is always fully transferable. Unknown types fall back to `generic`. |
| `model_features` | 5 | Hand-tuned physics priors from the graph spec YAML: `active_device`, `gm_efficiency_prior`, `ro_prior`, `parasitic_load_prior`, `mismatch_sensitivity`. Static per component — encode circuit-design intuition the GNN can use as a starting point. |
| `param_slots` | 2 | Normalized current parameter values owned by this node (e.g. `[W_norm, L_norm]` for a transistor, `[R_norm, 0]` for a resistor). At most 2 params per component across all supported topologies. |

Goal conditioning (`spec_context`) is kept **separate** from node features. It is a `2 × metric_slots = 8` vector of `[current_metrics | target_specs]`, both normalized by target values. Mixing it into node features would force every node embedding to carry redundant copies of the same goal information; instead it is projected once through a `spec_encoder` and injected as a residual into both the actor and critic.

#### Network architecture

```
                 node_features (n_nodes × 14)
                         │
              ┌──────────┴──────────┐
         actor_node_encoder    critic_node_encoder
              (Linear→Tanh)        (Linear→Tanh)
                  + type_embedding      + type_embedding   ← learnable, shared
              │                    │
         actor_gnn_layers[0..1]  critic_gnn_layers[0..1]
              (residual GraphConv)   (residual GraphConv)
                         │                    │
              h_actor (n_nodes × 128)    h_critic (n_nodes × 128)
                         │                    │
         ┌── param lookup ──┐          mean-pool over nodes
         │  + slot_embedding │                │
         │  + mean-pool ctx  │          + spec_encoder(spec_context)
         │  + spec_encoder   │                │
         └── h_params ───────┘          value_head → scalar V
                │
         policy_head → logits (n_params × n_actions)
```

**GraphConvLayer** (residual GraphSAGE-style):
```
h = h + tanh(W_self · h + W_neigh · (D⁻¹A · h))
```
The residual connection prevents over-smoothing when stacking 2+ layers and preserves pretrained signal during fine-tuning.

**Learnable type embedding** (`nn.Embedding(6, 128)`): shared between actor and critic, injected after each trunk's node encoder. Lets the network learn what "nmos" or "resistor" means from the RL reward signal, on top of the static `model_features` priors.

**Separate actor/critic trunks**: actor and critic have independent `node_encoder` + `gnn_layers`. During fine-tuning, the actor encoder is frozen (`--freeze-encoder-steps`) so pretrained representations are not corrupted while the critic's value function adapts to the new circuit's reward scale.

**Policy head**: for each parameter, the policy samples independently from a `Categorical` distribution over `n_actions` choices (e.g. `{-2, -1, 0, +1, +2}` index steps). The joint log-probability is the sum of per-parameter log-probs. The shared trunk encodes inter-parameter state so each head implicitly sees coupling — the factorization is an approximation, but the same one AutoCkt uses successfully.

#### What transfers and what doesn't

| Module | Transfers? | Notes |
|---|---|---|
| `actor_node_encoder` | Yes | Fixed `node_feature_dim=14` across all circuits |
| `actor_gnn_layers` | Yes | Hidden dim is constant |
| `type_embedding` | Yes | Global 6-type vocab is constant |
| `slot_embedding` | Yes | Max 2 params/component across all topologies |
| `spec_encoder` | Yes | If circuits share the same spec space (same metrics, same order) |
| `critic_node_encoder` | No | Reinitialised on fine-tune; critic adapts to new reward scale |
| `policy_head` | Yes | If `n_actions_per_param` matches (standardised to 5) |

## Commands

```bash
# Train an agent of a specific circuit
python train.py --agent ppo --config circuitrl/configs/cs_amp.yaml --seed 1

# Plot a single run
python plot.py --run-dir runs/ppo_cs_amp_seed0

# Plot (and overlay) multiple runs
python plot.py --run-dir runs/ppo_cs_amp_seed0 runs/ppo_cs_amp_seed1

# Evaluate a trained agent on 50 episodes
# use --verbose to see every step's action and reward
python evaluate.py --run-dir runs/ppo_cs_amp_seed0 --episodes 50 --seed 1

# Evaluate a trained agent on all episodes
python evaluate_all_specs.py --run-dir runs/ppo_cs_amp_seed0

# Plot evaluation results
python plot.py --run-dir runs/ppo_cs_amp_seed1 --eval

# Generate target specs and save them in circuitrl/configs/cs_amp_specs_pool.json
# these are using in training to make sure we have diverse achieveable specs 
# may also use this for supervised learning pretraining
python generate_specs.py --config circuitrl/configs/cs_amp.yaml --n 500
```

## GNN Transfer Learning

Uses `train_gnn_transfer.py` and `evaluate_gnn.py`. Does not affect the `train.py` / `evaluate.py` workflow.

The core idea: each circuit is encoded as a graph (nodes = components, edges = connections). A single shared GNN learns representations that generalize across topologies, so a checkpoint trained on circuit A can be fine-tuned on circuit B in far fewer steps than training from scratch.

### Step 1: Prerequisites

Each circuit needs a **spec pool** (achievable target samples) and a **graph spec** (topology definition).

**Spec pool** — simulate random parameter combos to collect achievable targets:
```bash
python generate_specs.py --config circuitrl/configs/cs_amp.yaml --n 500
python generate_specs.py --config circuitrl/configs/cascaded_amp.yaml --n 500
python generate_specs.py --config circuitrl/configs/opamp.yaml --n 500
python generate_specs.py --config circuitrl/configs/folded_cascode_ota.yaml --n 500
```
This writes e.g. `circuitrl/configs/cs_amp_specs_pool_train_500.json`. The config's `target_spec_file` field must point to this file.

**Graph spec** — YAML file in `circuitrl/configs/graph_specs/` that defines the circuit topology. Four are already provided:
- `cs_amp_graph.yaml` — 2 nodes (M1 NMOS, RD resistor), 1 edge
- `cascaded_amp_graph.yaml` — 4 nodes (M1, RD1, M2, RD2), 3 edges
- `opamp_graph.yaml` — 6 nodes (M1, M3, M5, M7, Cc, Ibias), 5 edges
- `folded_cascode_ota_graph.yaml` — 4 nodes (Min, Mpcas, Mncas, Ibias), 3 edges

> **Important**: circuits trained together must share the same spec space (same metrics in the same slot order). The `spec_encoder` learns slot semantics — mixing circuits with different specs breaks transfer. Two natural groups are provided: `cs_amp + cascaded_amp` (3 specs: gain_db, bandwidth, power) and `opamp + folded_cascode_ota` (4 specs: gain_db, ugbw, phase_margin, power).

Graph specs are auto-detected by name convention: `<config_basename>_graph.yaml`. Pass `--graph-spec` to override.

### Graph Spec Format

```yaml
model_feature_dim: 5
model_feature_names:   # optional, for documentation only
  - active_device          # 1.0 for transistors/sources, 0.0 for passives
  - gm_efficiency_prior    # higher = better transconductance per current
  - ro_prior               # higher = higher output resistance
  - parasitic_load_prior   # higher = more parasitic capacitance
  - mismatch_sensitivity   # higher = more sensitive to process variation

components:
  - name: M1               # must match parameter names below
    type: nmos             # one of: nmos, pmos, resistor, capacitor, current, generic
    parameters: [W1, L1]   # env parameter names owned by this node (max 2)
    model_features: [1.0, 0.80, 0.60, 0.50, 0.70]

  - name: RD1
    type: resistor
    parameters: [RD]
    model_features: [0.0, 0.00, 0.90, 0.10, 0.20]

undirected: true
edges:
  - [M1, RD1]              # component names, not parameter names
```

Every parameter in the circuit config must appear in exactly one component's `parameters` list. Components can own 1 or 2 parameters (e.g. W+L for a transistor, or just a single value for a current source).

### Step 2: Training

**Option A — Single circuit (baseline, no transfer):**
```bash
# Train GNN on one circuit  →  runs/gnn_pre_opamp_s1/
python train_gnn_transfer.py --config circuitrl/configs/cascaded_amp.yaml --timesteps 100000 --seed 1
```

**Option B — Multi-circuit simultaneous (recommended for transfer):**

Trains one shared network on all listed circuits at once. The GNN is forced to learn topology-agnostic representations, making subsequent fine-tuning faster and more stable than sequential pretrain → fine-tune.

*Easy pair — cs_amp + cascaded_amp (same 3 specs: gain_db, bandwidth, power):*
```bash
# cs_amp is one stage; cascaded_amp is two stages — simplest transfer test
python train_gnn_transfer.py \
--config circuitrl/configs/cs_amp.yaml \
  --circuits cs_amp:circuitrl/configs/cs_amp.yaml \
             cascaded_amp:circuitrl/configs/cascaded_amp.yaml \
  --timesteps 200000 --seed 1
```

*OTA pair — opamp + folded_cascode_ota (same 4 specs: gain_db, ugbw, phase_margin, power):*
```bash
python train_gnn_transfer.py \
  --config circuitrl/configs/opamp.yaml \
  --circuits opamp:circuitrl/configs/opamp.yaml \
             folded_cascode_ota:circuitrl/configs/folded_cascode_ota.yaml \
  --timesteps 200000 --seed 1
```

`--circuits` entries: `id:config_path` or `id:config_path:graph_spec_path`. The `--config` flag sets PPO hyperparameters and is required but its circuit is not trained unless also listed in `--circuits`. Use `--run-name` to override the auto-generated run name.

**Option C — Fine-tune a checkpoint on a new circuit:**

The actor encoder is frozen for `--freeze-encoder-steps` to let the new circuit's value function stabilize before the shared representation adapts.

```bash
# Fine-tune the OTA-pair checkpoint on folded_cascode_ota
python train_gnn_transfer.py \
  --config circuitrl/configs/folded_cascode_ota.yaml \
  --init-checkpoint runs/gnn_pre_opamp+folded_cascode_ota_s1/model.pt \
  --freeze-encoder-steps 20000 \
  --timesteps 50000 --seed 1

  python train_gnn_transfer.py \
  --config circuitrl/configs/cascaded_amp.yaml \
  --init-checkpoint runs/gnn_pre_cs_amp_s1_1/model.pt \
  --freeze-encoder-steps 10000 \
  --timesteps 50000 --seed 1
```

### Step 3: Plotting training curves

GNN training writes the same `metrics.csv` format as flat PPO, so the same `plot.py` works:

```bash
# Plot training curves for a GNN run
python plot.py --run-dir runs/gnn_pre_opamp+folded_cascode_ota_s1

# Overlay two GNN runs for comparison
python plot.py --run-dir runs/gnn_pre_opamp+folded_cascode_ota_s1 runs/gnn_ft_folded_cascode_ota_s1

# Also save one PNG per metric
python plot.py --run-dir runs/gnn_pre_opamp+folded_cascode_ota_s1 --separate
```

### Step 4: Evaluation

```bash
# Evaluate on randomly sampled specs (default 10 episodes)
python evaluate_gnn.py --run-dir runs/gnn_pre_opamp_s1 --episodes 50 --seed 1

# Evaluate on every spec in the training pool
python evaluate_gnn.py --run-dir runs/gnn_pre_opamp_s1 --all-specs

# Evaluate on a held-out test pool
python evaluate_gnn.py --run-dir runs/gnn_pre_cascaded_amp_s1 \
  --spec_pool_test circuitrl/configs/cascaded_amp_specs_pool_test_500.json \
  --all-specs

# Evaluate a multi-circuit checkpoint on a specific circuit
python evaluate_gnn.py \
  --run-dir runs/gnn_pre_opamp+folded_cascode_ota_s1 \
  --config circuitrl/configs/opamp.yaml \
  --episodes 50
```

Each `evaluate_gnn.py` run saves two files to the run directory:
- `eval_results.csv` — per-episode targets and achieved metrics
- `eval_summary.txt` — full printed output (episode-by-episode + summary)

```bash
# Plot eval results (scatter + success rates) after running evaluate_gnn.py
python plot.py --run-dir runs/gnn_pre_opamp_s1 --eval
```

Use `evaluate_gnn.py` for GNN checkpoints, not `evaluate.py`.

### Adding a New Circuit

1. Add a circuit config YAML to `circuitrl/configs/` (see config format above)
2. Add a netlist template `.sp` to `circuitrl/envs/`
3. Create a graph spec YAML in `circuitrl/configs/graph_specs/<name>_graph.yaml` following the format above
4. Generate a spec pool: `python generate_specs.py --config circuitrl/configs/<name>.yaml --n 500`
5. Train: `python train_gnn_transfer.py --config circuitrl/configs/<name>.yaml --circuits <name>:circuitrl/configs/<name>.yaml`

## Health checks

```bash
# Run all tests
pytest tests/ -v

# Smoke-test the NGSpice runner (runs on simulation with some random parameters and prints returned metrics)
python -c "
from circuitrl.simulators.ngspice_runner import NGSpiceRunner
runner = NGSpiceRunner('circuitrl/envs/netlist_template.sp')
result = runner.run({'W1': '10u', 'L1': '0.5u', 'W3': '20u', 'L3': '0.5u',
                     'W5': '10u', 'L5': '0.5u', 'W7': '1u', 'L7': '2u',
                     'Cc': '1p', 'Ib': '10u'})
print(result)
"

# Smoke-test the Gym environment (creates a circuit environment, calls reset, samples action, takes one step, prints observation shape and reward)
python -c "
from circuitrl.envs.circuit_env import CircuitEnv
env = CircuitEnv()
obs, info = env.reset()
print('obs shape:', obs.shape, 'metrics:', info['metrics'])
obs, reward, term, trunc, info = env.step(env.action_space.sample())
print('reward:', reward)
"
```
