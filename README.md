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

## Commands

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

# Run training

# 2 stage op amp (10 parameters)
python train.py --agent ppo --config circuitrl/configs/opamp_default.yaml

# cs amp (3 parameters only)
python train.py --agent ppo --config circuitrl/configs/cs_amp.yaml

# Plot a single run example
python plot.py --run-dir runs/ppo_cs_amp_seed0

# Plot (and overlay) multiple runs
python plot.py --run-dir runs/ppo_seed0 runs/ppo_seed1 runs/ppo_seed2

# evaluate a trained agent, see final params + specs
python evaluate.py --run-dir runs/ppo_cs_amp_seed0

# See every step's action and reward
python evaluate.py --run-dir runs/ppo_cs_amp_seed0 --verbose