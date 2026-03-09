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

- `ppo`: shared-trunk PPO actor-critic with simultaneous parameter updates
- `ppo_non_shared`: separate actor/critic PPO, with simultaneous parameter updates
- `ppo-seq`: shared-trunk sequential PPO, one parameter update per step

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
