# Reinforcement Learning Agents for 2048

This repository contains an optimization-driven reinforcement learning study on the game **2048**, focusing on why naive policy-gradient methods plateau early and which stabilization and representation techniques enable progress toward higher tiles (512 / 1024).

The project systematically evolves from basic Monte Carlo policy gradients to a stabilized PPO-based pipeline with lookahead features and recurrent policies (LSTM), with controlled ablations to isolate the contribution of each component.

---

## Core Objective

Rather than competing with heuristic or search-based solvers, the goal of this project is to:

- Understand **failure modes of policy-gradient learning** in long-horizon stochastic environments
- Identify **which algorithmic stabilizations materially improve learning**
- Evaluate whether **temporal context (LSTM)** improves long-horizon board control under otherwise identical PPO settings

---

## Methodological Progression

### 1. REINFORCE Baselines (Diagnostic Stage)

We begin with Monte Carlo policy gradients to expose fundamental optimization issues:

- **REINFORCE (no entropy):**
  - Rapid collapse to deterministic action loops
  - Severe exploration loss
- **REINFORCE (+ entropy):**
  - Improved exploration
  - High variance persists
  - Early plateau (typically ≤256)

These baselines serve as controlled diagnostics rather than final solutions.

---

### 2. Variance Reduction via Actor–Critic (A2C)

To address Monte Carlo variance, we introduce a learned value baseline:

- Advantage Actor–Critic (A2C)
- State-dependent baseline improves average performance
- Still struggles with long-horizon planning
- Performance typically plateaus around **512**

Conclusion: variance reduction alone is insufficient.

---

### 3. PPO-Advanced: Stabilized Policy Optimization

The main performance gains emerge after transitioning to a stabilized PPO pipeline:

**Key components**
- PPO clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Entropy regularization
- Action masking (invalid moves removed from policy support)
- Numerically robust optimization
- Periodic evaluation and checkpointing
- Curriculum-style difficulty staging

This configuration yields significantly more stable training dynamics than REINFORCE or A2C.

---

### 4. Lookahead Feature Augmentation

To improve short-horizon decision quality, we augment the state with **deterministic lookahead features** computed per action (before stochastic tile spawn):

- Merge count
- Total merge value
- Change in empty cell count
- Action validity indicators

These features expose immediate mechanical consequences of actions, reducing the burden on the policy to infer them implicitly.

---

### 5. Recurrent Policy: PPO + LSTM (Ablation Focus)

Despite full observability, effective 2048 play requires **temporal consistency** (structure preservation, avoiding oscillatory patterns).

We introduce an LSTM-based policy trained with truncated BPTT:

- Sequence length: 20
- Hidden states reset at episode boundaries
- Identical PPO-Advanced configuration used for fair ablation

**Key result**
- PPO-LSTM significantly outperforms feed-forward PPO
- Higher average evaluation score
- Occasional reach of **1024** in periodic evaluations
- Demonstrates the importance of short-term temporal context for long-horizon board control

---

## Quantitative Summary

| Method | Avg Eval Score | Best Tile |
|--------|---------------|-----------|
| REINFORCE (no entropy) | ~2188 | 256 |
| REINFORCE (+ entropy) | ~2264 | 512 |
| A2C | ~2423 | 512 |
| PPO-Advanced (no LSTM) | ~1441 | 256 |
| **PPO-Advanced + LSTM** | **~3484** | **1024 (periodic)** |

---

## Key Takeaways

- Monte Carlo policy gradients learn non-trivial behaviors but plateau early
- Entropy prevents premature determinism but does not solve variance
- Actor–critic improves averages but not tile ceiling
- PPO stabilization is essential for robustness
- Lookahead features improve local decision quality
- **LSTM recurrence provides the strongest observed gain**, even in a fully observable environment

---

## Scope & Limitations

- Reaching 2048 consistently remains rare
- High-tile events are sensitive to stochastic spawning
- Performance gap between best-case and typical behavior remains large

Future work includes longer training horizons, larger evaluation budgets, and systematic shaping/lookahead ablations.

---

## Project Structure

```
rl-2048-agent/
├── src/                    # Source code
│   ├── Game_2048_BE.py     # 2048 game environment
│   ├── Game_2048_FE.py     # Pygame interface for manual play
│   └── Policy_Gradient.py  # RL algorithms and neural networks
├── experiments/             # Experiment scripts
│   ├── experiment_1_no_entropy.py
│   ├── experiment_2_with_entropy.py
│   ├── experiment_3_a2c.py
│   ├── experiment_4_ppo.py
│   ├── experiment_5_ppo_advanced.py
│   ├── experiment_5_ppo_advanced_no_lstm.py
│   ├── experiment_6_reinforce_advanced.py
│   └── experiment_7_reinforce_lstm.py
├── results/                 # Training results (plots, reports)
├── checkpoints/             # Model checkpoints
├── requirements.txt         # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch numpy matplotlib pygame
```




