
## 🧩 TINY RECURSIVE MODEL (TRM)

Original Implementation of TRM with Reinforcement Learning Environments
(Maze, Sudoku, etc.)

This repository implements multiple versions of the Tiny Recursivr Model (TRM) —
a recurrent reasoning model designed for structured decision-making and temporal reasoning tasks.

##  TL;DR SUMMARY

| Task                   | File                                                   | Status         |
|------------------------|-------------------------------------------------------|----------------|
| Learn TRM basics       | trm.py                                                | ✅ Basic       |
| Debug RL setup         | trm_rl_debug.py                                       | ✅ Stable      |
| Run RL experiments     | trm_rl_improved.py, trm_rl_optimized_updated_sudoku   | ✅✅ Improved   |
| Paper-level TRM        | trm_rl_paper_implementation.py                        | ⭐ Recommended |
| Compare baselines      | maze_comparison.py                                    | ✅ Done  



## 1. OVERVIEW

TRM introduces a hierarchical recursive architecture for reasoning tasks.

This repository includes multiple versions for different stages of experimentation:

| Level         | File Name                     | Description |
|----------------|------------------------------|--------------|
| Basic          | trm.py                       | Minimal TRM on toy datasets |
| Intermediate   | trm_rl_evns.py               | TRM adapted for RL (Maze, Sudoku) |
| Improved       | trm_rl_improved.py, trm_rl_optimized_updated_sudoku           | Adds shaped rewards, entropy bonus, gradient clipping |
| Full (Paper)   | trm_rl_paper_implementation.py | Hierarchical TRM with K-cycles, carry, SwiGLU |
| Debug          | trm_rl_debug.py              | Diagnostic tools and failure analysis |

## 2. FOLDER STRUCTURE

```
Tiny Recursive Model/
├── README.txt
├── requirements.txt
├── trm_core/
│   ├── README_TRM_PAPER.md
│   ├── trm.py
│   ├── trm_rl_debug.py
│   ├── trm_rl_evns.py
│   ├── trm_rl_improved.py
│   ├── trm_rl_optimized.py
│   ├── trm_rl_optimzed_updated_sudoku.py
│   ├── trm_rl_paper_implementation.py
│   └── utils/
│       └── improved_sudoku_env.py
├── experiments/
│   ├── simple_maze_comparison.py
│   └── maze_comparison.py
├── results/
│   └── experiments/
└── docs/
    └── README_TRM_PAPER.md
```

## 3. INSTALLATION

Clone the repository and install dependencies:

    git clone https://github.com/acharyaanusha/Tiny-Recursive-Model.git
    cd Tiny-Recursive-Model
    pip install -r requirements.txt


## 4. QUICK START

Step 1: Run a debug check
    python trm_core/trm_rl_debug.py

Step 2: Train the full TRM (Paper Implementation)
    python trm_core/trm_rl_paper_implementation.py

This will:
- Train TRM with K=3 and K=5 recursive cycles
- Train an MLP baseline
- Save comparison plots and logs under "results/"


## 5. ARCHITECTURE SUMMARY

### Basic TRM (evns, improved)
```
State → Embed → [Inner Steps × (Tiny Net)] → Head → Logits
                 ↑____ y_embed feedback ______|
```

### Paper TRM (paper_implementation)
```
State → Embed → H-Level → L-Level → Feedback ┐
                   ↑_______carry_______________|
                   (repeat K cycles)
                         ↓
                    Action Head → Logits
```

Key Features:
- Recursive reasoning with K cycles
- SwiGLU activations
- Carry mechanism for information retention
- Deep supervision
- Multiple baseline comparisons (MLP, TRM-K3, TRM-K5)


## 6. HYPERPARAMETER RECOMMENDATIONS

### For Maze (5×5)
```python
# trm_rl_paper_implementation.py
TRMPolicy(
    h_dim=64,
    l_dim=32,
    k_cycles=3,      # Start with 3
    n_h_layers=2,
    n_l_layers=1
)

# Training
lr = 2e-3          # Lower than MLP
entropy_coef = 0.01
n_episodes = 500
```

### For Sudoku (4×4)
```python
TRMPolicy(
    h_dim=128,       # Bigger for combinatorial
    l_dim=64,
    k_cycles=5,      # More reasoning needed
    n_h_layers=3,
    n_l_layers=2
)

# Training
lr = 1e-3          # Lower due to complexity
entropy_coef = 0.02  # More exploration
n_episodes = 1000
```

## 7. EXPECTED RESULTS

### Maze Environment (5×5)

| Implementation | Typical Success Rate | Training Speed |
|----------------|---------------------|----------------|
| trm_rl_evns.py | ~20-30% | Medium |
| trm_rl_improved.py | ~70-80% | Medium |
| **trm_rl_paper_implementation.py** | **~85-90%** | Slower (K cycles) |
| MLP Baseline | ~60-70% | Fast |


## 8. RESEARCH DIRECTIONS

### Easy Experiments
- [ ] Test different K values (2, 3, 5, 7)
- [ ] Compare SwiGLU vs ReLU
- [ ] Ablation: with/without carry mechanism
- [ ] Different maze sizes (3×3, 7×7, 9×9)

### Medium Experiments
- [ ] Add attention in H-level
- [ ] Implement adaptive halting (ACT)
- [ ] Multi-task learning (Maze + Sudoku)
- [ ] Curriculum learning

### Advanced Experiments
- [ ] Image-based RL (Atari)
- [ ] Continuous control
- [ ] Compare with Transformers
- [ ] Publish results

## 9. LOGGING & VISUALIZATION

All implementations now include:
- ✅ Episode rewards
- ✅ Success rates
- ✅ Average performance windows
- ✅ Best performance tracking
- ✅ Training curves
- ✅ Success rate over time

The paper implementation additionally provides:
- ✅ Multiple model comparison
- ✅ Parameter count comparison
- ✅ Smooth curves with raw data overlay

## 10. REQUIREMENTS


- Python >= 3.9
- PyTorch >= 2.0
- NumPy
- Matplotlib
- tqdm
- gymnasium (or gym)

Install with:
    pip install torch numpy matplotlib tqdm gymnasium

## 11. CITATION

If you use this repository in your research, please cite:
```
@software{anushaacharya2025trm,
  title  = {Tiny Recursive Models (TRM)},
  author = {Anusha Acharya},
  year   = {2025},
  url    = {https://github.com/acharyaanusha/Tiny-Recursive-Model}
}
```
