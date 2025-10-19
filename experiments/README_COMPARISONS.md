# TRM Maze Solving Comparisons

This folder contains two comparison scripts that demonstrate the Tiny Recursive Model (TRM) solving maze problems versus baseline models.

## Files

### 1. `simple_maze_comparison.py` (Recommended to start)
**Comparison:** Simple CNN vs TRM

A streamlined comparison focusing on core architectural differences.

**Features:**
- ✅ Simple 8x8 mazes with random obstacles
- ✅ CNN baseline (standard convolutional architecture)
- ✅ Simplified TRM implementation
- ✅ Side-by-side visualizations
- ✅ Training curves comparison
- ✅ Faster training (~20 epochs)

**Usage:**
```bash
python simple_maze_comparison.py
```

**Output:**
- `simple_maze_comparison.png` - Visual comparison of predictions
- `simple_training_curves.png` - Training loss and validation accuracy
- Console output showing training progress


### 2. `maze_comparison.py` (Full-featured)
**Comparison:** Transformer Encoder vs TRM

A comprehensive comparison with more complex mazes and a stronger baseline.

**Features:**
- ✅ Complex 10x10 mazes generated with DFS algorithm
- ✅ Transformer baseline (multi-head attention architecture)
- ✅ Full TRM implementation from the paper
- ✅ BFS-based ground truth path finding
- ✅ Multiple visualization examples
- ✅ Detailed performance metrics

**Usage:**
```bash
python maze_comparison.py
```

**Output:**
- `maze_comparison_1.png`, `maze_comparison_2.png`, `maze_comparison_3.png` - Individual maze predictions
- `training_curves.png` - Training metrics over time
- Console output with detailed epoch-by-epoch comparison


## Key Differences Between Models

### CNN Baseline (simple version)
```
Input (maze + markers) → Conv layers → Output (path prediction)
```
- **Pros:** Simple, fast, proven architecture
- **Cons:** Fixed capacity, no iterative refinement

### Transformer Baseline (complex version)
```
Input → Embedding + Positional Encoding → Transformer Encoder → Output
```
- **Pros:** Attention mechanism, better at long-range dependencies
- **Cons:** Larger model, more parameters, no iterative refinement

### TRM (both versions)
```
Input → Initial prediction
  ↓
For each supervision step:
  For each inner step:
    Refine latent z using tiny network
  Generate improved prediction
```
- **Pros:** Iterative refinement, learns to improve predictions recursively
- **Cons:** More complex training, requires multiple supervision steps


## Understanding the Task

**Goal:** Predict which cells are on the shortest path from start to goal

**Input:**
- Channel 1: Maze layout (0=path, 1=wall)
- Channel 2: Start position marker
- Channel 3: Goal position marker

**Output:**
- Binary prediction for each cell (on_path=1, not_on_path=0)

**Why This Task?**
Maze solving requires:
- Global reasoning (considering start and goal)
- Path constraints (no going through walls)
- Optimization (finding shortest path)
- Iterative refinement (TRM's strength)


## Expected Results

Based on the TRM paper, you should observe:

1. **TRM shows iterative improvement:** Each supervision step produces better predictions
2. **TRM handles complex constraints:** Better at respecting maze walls and finding valid paths
3. **Comparable or better accuracy:** TRM should match or exceed baseline accuracy
4. **Different learning dynamics:** TRM may converge differently due to recursive refinement


## Customization

### Adjust maze difficulty:
```python
# In simple_maze_comparison.py
train_dataset = SimpleMazeDataset(n_samples=300, size=12)  # Larger mazes

# In maze_comparison.py
train_dataset = MazeDataset(n_samples=500, height=15, width=15)  # Larger mazes
```

### Adjust TRM recursion depth:
```python
# More inner refinement steps
trm_model = SimpleTRM(inner_steps=5, sup_steps=8)

# More supervision steps
trm_model = TRMMazeSolver(inner_steps=4, sup_steps=10)
```

### Adjust model capacity:
```python
# Larger latent dimensions
trm_model = SimpleTRM(z_dim=64, x_dim=32, y_dim=16)
```


## Requirements

```bash
pip install torch torchvision numpy matplotlib
```



## Troubleshooting

**Issue:** CUDA out of memory
- Reduce `BATCH_SIZE`
- Use smaller mazes
- Reduce `z_dim` and other dimensions

**Issue:** Poor accuracy for both models
- Increase training epochs
- Try different learning rates
- Check that mazes are solvable (they should be)

**Issue:** TRM not improving iteratively
- Ensure `detach_inner_steps=True`
- Try different `inner_steps` and `sup_steps` values
- Check learning rate (TRM often needs lower LR)



## Next Steps

1. **Start with:** `simple_maze_comparison.py` to understand the basics
2. **Then try:** `maze_comparison.py` for more rigorous comparison
3. **Experiment:** Modify architectures, hyperparameters, and maze complexity
4. **Visualize:** Use the generated images to understand how TRM refines predictions

