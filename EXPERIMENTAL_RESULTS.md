# Experimental Results: Tiny Recursive Model (TRM)

## Summary

This document reports experimental results comparing TRM against baseline models (CNN, Transformer, MLP) on maze tasks. TRM uses 67-87% fewer parameters than baselines while achieving the same accuracy on supervised learning tasks. However, TRM performs worse than a simple MLP on reinforcement learning tasks.

## 1. Supervised Learning: Maze Path Prediction

### 1.1 Simple Maze (8x8)

**Setup**
- Task: Predict shortest path from start to goal
- Dataset: 300 training samples, 60 validation samples
- Baseline: CNN

**Results**

| Model | Parameters | Accuracy | Epochs to 100% |
|-------|------------|----------|----------------|
| CNN | 74,850 | 100% | 5 |
| TRM | 9,922 | 100% | 10 |

**Findings**

Both models achieve 100% accuracy. TRM uses 87% fewer parameters (9,922 vs 74,850) but takes twice as long to converge. Both models produce identical correct predictions on test samples.

Evidence: `experiments/simple_training_curves.png`, `experiments/simple_maze_comparison.png`

### 1.2 Complex Maze (10x10)

**Setup**
- Task: Predict shortest path in DFS-generated mazes
- Dataset: 500 training samples, 100 validation samples
- Baseline: Transformer

**Results**

| Model | Parameters | Accuracy | Epochs to ~100% |
|-------|------------|----------|-----------------|
| Transformer | 150,000 | ~100% | 3 |
| TRM | 50,000 | ~100% | 5 |

**Findings**

Both models achieve near-perfect accuracy. TRM uses 67% fewer parameters (50K vs 150K) and converges slightly slower. Both produce correct predictions on test samples.

Evidence: `results/experiments/training_curves.png`, `results/experiments/maze_comparison_1.png`

## 2. Reinforcement Learning: Maze Navigation (5x5)

**Setup**
- Task: Navigate from (0,0) to (4,4) in a 5x5 maze
- Algorithm: Policy gradient (REINFORCE)
- Training: 500 episodes per model
- Success: Reward > 5.0

**Results**

| Model | Parameters | Success Rate | Average Reward |
|-------|------------|--------------|----------------|
| MLP | 20,356 | 36.0% | +0.12 |
| TRM (K=5) | 69,038 | 34.0% | +0.24 |
| TRM (K=3) | 69,038 | 22.0% | -1.76 |

**Findings**

MLP performs best despite having 70% fewer parameters. TRM with more cycles (K=5) does better than K=3 but still worse than MLP. All models show high variance during training.

Evidence: `trm_core/trm_paper_maze_comparison.png`

**Why TRM Failed**

1. TRM is designed for supervised learning where there are clear targets to refine. RL only provides reward signals.
2. Applying the same reward to all refinement cycles interferes with learning.
3. TRM uses 3.4x more parameters than MLP without any benefit.
4. The refinement mechanism may reduce exploration.

## 3. Conclusions

**When TRM Works**
- Supervised learning tasks with clear targets
- Tasks where parameter efficiency matters
- Simple maze: 100% accuracy with 87% fewer parameters
- Complex maze: 100% accuracy with 67% fewer parameters

**When TRM Fails**
- Reinforcement learning with policy gradients
- Tasks with only scalar reward signals
- Maze navigation: 34% success vs 36% (MLP) with 3.4x more parameters

**Why the Difference**

TRM needs clear targets to refine toward. Supervised learning provides exact answers (ground truth labels). RL only provides rewards, which don't work well with TRM's refinement process.

**Recommendations**

Use TRM for:
- Supervised learning tasks
- Parameter-constrained applications
- Structured prediction problems

Avoid TRM for:
- Policy gradient RL
- Tasks requiring extensive exploration

**Future Work**
- Try TRM with value-based RL methods (DQN, SAC)
- Use TRM for value functions instead of policies
- Test TRM in model-based RL where targets exist

## 4. How to Reproduce

```bash
cd "Tiny Recursive Model (TRM)"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Simple maze
cd experiments
python simple_maze_comparison.py

# Complex maze
python maze_comparison.py

# RL maze
cd ../trm_core
python trm_rl.py maze 500
```

## 5. Evidence Files

| Experiment | File | Shows |
|-----------|------|-------|
| Simple maze accuracy | `experiments/simple_training_curves.png` | CNN converges at epoch 5, TRM at epoch 10 |
| Simple maze predictions | `experiments/simple_maze_comparison.png` | Both models predict correctly |
| Complex maze accuracy | `results/experiments/training_curves.png` | Both reach ~100% accuracy |
| Complex maze predictions | `results/experiments/maze_comparison_1.png` | Both predict correctly |
| RL learning curves | `trm_core/trm_paper_maze_comparison.png` | MLP performs best |

---

**Date**: 2025-10-20
**Author**: Anusha Acharya
