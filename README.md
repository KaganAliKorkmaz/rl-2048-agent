# 2048 Reinforcement Learning

Policy gradient algorithms for 2048 game. Training with REINFORCE, A2C, and PPO.

## Setup

```bash
pip install torch numpy matplotlib pygame
```

## Usage

### Running experiments

```bash
# REINFORCE (without entropy)
python experiment_1_no_entropy.py

# REINFORCE (with entropy)
python experiment_2_with_entropy.py

# A2C
python experiment_3_a2c.py

# PPO
python experiment_4_ppo.py

# PPO Advanced (LSTM)
python experiment_5_ppo_advanced.py
```

### Manual play

```bash
python Game_2048_FE.py
```

Use arrow keys to play.

## Files

- `Policy_Gradient.py` - All algorithms and networks
- `Game_2048_BE.py` - 2048 environment
- `Game_2048_FE.py` - Pygame interface
- `experiment_*.py` - Experiment files for different algorithms
