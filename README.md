# CircuitRL

Goal-conditioned RL for analog circuit sizing. An agent learns to size a two-stage op-amp to hit target performance specs (gain, bandwidth, phase margin, power) using NGSpice simulation as a reward signal.

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

# Smoke-test the NGSpice runner
python -c "
from circuitrl.simulators.ngspice_runner import NGSpiceRunner
runner = NGSpiceRunner('circuitrl/envs/netlist_template.sp')
result = runner.run({'W1': '10u', 'L1': '0.5u', 'W3': '20u', 'L3': '0.5u',
                     'W5': '10u', 'L5': '0.5u', 'W7': '1u', 'L7': '2u',
                     'Cc': '1p', 'Ib': '10u'})
print(result)
"

# Smoke-test the Gym environment
python -c "
from circuitrl.envs.opamp_env import OpAmpEnv
env = OpAmpEnv()
obs, info = env.reset()
print('obs shape:', obs.shape, 'metrics:', info['metrics'])
obs, reward, term, trunc, info = env.step(env.action_space.sample())
print('reward:', reward)
"

# Run training (coming soon)
# python train.py --agent ppo --config circuitrl/configs/opamp_default.yaml
```
