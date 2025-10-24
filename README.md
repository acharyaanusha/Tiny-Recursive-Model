# Tiny Recursive Model (TRM)

**A Parameter-Efficient Hierarchical Architecture for Iterative Reasoning**

## Abstract

This repo implements the Tiny Recursive Model (TRM) — a lightweight recurrent architecture designed for structured reasoning through hierarchical recursion.

We experiment with TRM in both supervised and reinforcement learning settings.

In supervised tasks, TRM achieves strong parameter efficiency (up to 67–87% fewer parameters).

In reinforcement learning, we find that policy gradient methods struggle to fully exploit TRM’s recursive reasoning ability.

Overall, our results suggest that iterative refinement models work best when trained with structured or explicit supervision.

**Full experimental analysis:** [EXPERIMENTAL_RESULTS.md](EXPERIMENTAL_RESULTS.md)

## Key Contributions

1. **Parameter Efficiency**: TRM achieves competitive accuracy with 67-87% fewer parameters than baseline models on supervised spatial reasoning tasks
2. **Architectural Innovation**: Hierarchical recursive design with K-cycle iterative refinement and carry mechanisms
3. **Empirical Analysis**: Systematic comparison across learning paradigms revealing task-specific architectural benefits
4. **Open Implementation**: Modular codebase enabling replication and extension of results

---

## 1. Overview

TRM is a hierarchical recursive architecture where representations are iteratively refined through multiple reasoning cycles. The model uses two nested processing levels (H-level and L-level) with a carry mechanism for information persistence.

### Implementation Variants

| Implementation | Module | Description |
|----------------|--------|-------------|
| **Baseline** | trm.py | Minimal recursive architecture |
| **RL-Adapted** | trm_rl_evns.py | Environment-specific RL implementations |
| **Enhanced** | trm_rl_improved.py | Improved training strategies |
| **Full** | trm_rl.py | Complete hierarchical TRM with K-cycles |
| **Debug** | trm_rl_debug.py | Diagnostic tools |

---

## 2. Repository Structure

```
Tiny Recursive Model/
├── README.md
├── EXPERIMENTAL_RESULTS.md
├── requirements.txt
├── trm_core/
│   ├── trm.py
│   ├── trm_rl.py
│   ├── trm_rl_debug.py
│   ├── trm_rl_evns.py
│   ├── trm_rl_improved.py
│   ├── trm_rl_optimized.py
│   ├── trm_rl_optimized_updated_sudoku.py
│   └── utils/
│       └── improved_sudoku_env.py
├── experiments/
│   ├── simple_maze_comparison.py
│   └── maze_comparison.py
└── results/
    └── experiments/
```

---

## 3. Installation

```bash
git clone https://github.com/acharyaanusha/Tiny-Recursive-Model.git
cd Tiny-Recursive-Model
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, NumPy, Matplotlib, tqdm, gymnasium

---

## 4. Running Experiments

**Supervised Learning:**
```bash
python experiments/simple_maze_comparison.py  # 8×8 maze
python experiments/maze_comparison.py         # 10×10 maze
```

**Reinforcement Learning:**
```bash
python trm_core/trm_rl_debug.py  # Diagnostics
python trm_core/trm_rl.py        # Full training
```

---

## 5. Architecture

**Baseline:**
```
State → Embedding → [Recursive Processing] → Output
                           ↑___feedback____|
```

**Hierarchical TRM:**
```
State → Embedding → H-Level → L-Level → Output
                      ↑______carry______|
                   (repeat K cycles)
```

**Key Features:** K-cycle recursion, hierarchical processing, carry mechanism, SwiGLU activation, deep supervision

---

## 6. Hyperparameters

**Maze (5×5):** h_dim=64, l_dim=32, k_cycles=3, lr=2e-3, episodes=500

**Sudoku (4×4):** h_dim=128, l_dim=64, k_cycles=5, lr=1e-3, episodes=1000

---

## 7. Results

**Full analysis:** [EXPERIMENTAL_RESULTS.md](EXPERIMENTAL_RESULTS.md)

### Supervised Learning

| Task | Model | Parameters | Accuracy |
|------|-------|------------|----------|
| Simple Maze (8×8) | CNN | 74,850 | 100% |
| | **TRM** | **9,922** | **100%** (87% fewer params) |
| Complex Maze (10×10) | Transformer | ~150K | ~100% |
| | **TRM** | **~50K** | **~100%** (67% fewer params) |

### Reinforcement Learning

| Model | Parameters | Success Rate |
|-------|------------|--------------|
| MLP Baseline | 20,356 | **36.0%** |
| TRM (K=5) | 69,038 | 34.0% |
| TRM (K=3) | 69,038 | 22.0% |

### Key Findings

- **Supervised:** TRM achieves competitive accuracy with 67-87% fewer parameters
- **RL:** TRM does not outperform MLP despite 3× more parameters
- **Hypothesis:** Iterative refinement requires structured targets (labels) rather than scalar rewards for effective learning

---

## 8. Citation

```bibtex
@software{acharya2025trm,
  title  = {Tiny Recursive Model (TRM): A Parameter-Efficient Hierarchical Architecture for Iterative Reasoning},
  author = {Anusha Acharya},
  year   = {2025},
  url    = {https://github.com/acharyaanusha/Tiny-Recursive-Model}
}
```
