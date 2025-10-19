"""
Improved Sudoku Environment for RL

Problems with original:
1. Too large action space (64 actions)
2. Starting from empty grid (too hard)
3. Poor state representation
4. Sparse rewards

Improvements:
1. Start with partially filled grid (easier)
2. Smaller 4×4 sudoku (2×2 boxes)
3. Better state encoding (value + mask)
4. Shaped rewards for progress
5. Guided exploration
"""

import torch
import torch.nn as nn
import numpy as np
import random


class ImprovedSudokuEnv:
    """
    4x4 Sudoku with partial starting state

    Grid layout (2×2 boxes):
    ┌─────┬─────┐
    │ 1 2 │ 3 4 │
    │ 3 4 │ 1 2 │
    ├─────┼─────┤
    │ 2 1 │ 4 3 │
    │ 4 3 │ 2 1 │
    └─────┴─────┘
    """

    def __init__(self, size=4, difficulty=0.5):
        """
        Args:
            size: Grid size (4 for 4×4)
            difficulty: 0.0 (very easy) to 1.0 (very hard)
                       Controls how many cells start empty
        """
        self.size = size
        self.difficulty = difficulty
        self.box_size = int(np.sqrt(size))
        self.reset()

    def _generate_solved_grid(self):
        """Generate a valid solved Sudoku grid"""
        # For 4×4, use a simple valid solution
        if self.size == 4:
            return np.array([
                [1, 2, 3, 4],
                [3, 4, 1, 2],
                [2, 1, 4, 3],
                [4, 3, 2, 1]
            ])
        else:
            raise NotImplementedError("Only 4×4 supported")

    def _create_puzzle(self, solution, difficulty):
        """Remove cells to create puzzle"""
        puzzle = solution.copy()
        mask = np.ones_like(puzzle, dtype=bool)  # True = can modify

        # Determine how many cells to remove
        total_cells = self.size * self.size
        n_remove = int(total_cells * difficulty)
        n_remove = min(n_remove, total_cells - 4)  # Keep at least 4 cells

        # Randomly remove cells
        positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        remove_positions = random.sample(positions, n_remove)

        for i, j in remove_positions:
            puzzle[i, j] = 0
            mask[i, j] = True

        # Mark given cells as immutable
        for i in range(self.size):
            for j in range(self.size):
                if puzzle[i, j] != 0:
                    mask[i, j] = False

        return puzzle, mask

    def reset(self):
        """Reset to a new puzzle"""
        self.solution = self._generate_solved_grid()
        self.grid, self.mask = self._create_puzzle(self.solution, self.difficulty)
        self.initial_grid = self.grid.copy()
        self.steps = 0
        self.filled_correctly = np.sum(self.grid == self.solution)
        return self._get_state()

    def _get_state(self):
        """
        State encoding with multiple channels:
        - Grid values (0-4, normalized)
        - Mask (can modify or not)
        - Row conflicts
        - Col conflicts
        - Box conflicts
        """
        # Channel 1: Grid values (normalized)
        grid_norm = self.grid.astype(np.float32) / self.size

        # Channel 2: Modifiable mask
        mask_channel = self.mask.astype(np.float32)

        # Channel 3-5: Conflict indicators (helps agent learn)
        row_conflicts = np.zeros_like(self.grid, dtype=np.float32)
        col_conflicts = np.zeros_like(self.grid, dtype=np.float32)
        box_conflicts = np.zeros_like(self.grid, dtype=np.float32)

        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] > 0:
                    val = self.grid[i, j]
                    # Check row conflict
                    if np.sum(self.grid[i, :] == val) > 1:
                        row_conflicts[i, j] = 1.0
                    # Check col conflict
                    if np.sum(self.grid[:, j] == val) > 1:
                        col_conflicts[i, j] = 1.0
                    # Check box conflict
                    r0, c0 = (i // self.box_size) * self.box_size, (j // self.box_size) * self.box_size
                    box = self.grid[r0:r0+self.box_size, c0:c0+self.box_size]
                    if np.sum(box == val) > 1:
                        box_conflicts[i, j] = 1.0

        # Stack all channels and flatten
        state = np.stack([
            grid_norm,
            mask_channel,
            row_conflicts,
            col_conflicts,
            box_conflicts
        ]).flatten()

        return torch.tensor(state, dtype=torch.float32)

    def _is_valid_placement(self, row, col, val):
        """Check if placing val at (row, col) is valid"""
        # Check row
        if val in self.grid[row, :]:
            return False
        # Check column
        if val in self.grid[:, col]:
            return False
        # Check box
        r0 = (row // self.box_size) * self.box_size
        c0 = (col // self.box_size) * self.box_size
        if val in self.grid[r0:r0+self.box_size, c0:c0+self.box_size]:
            return False
        return True

    def step(self, action):
        """
        Take action: place a value in a cell
        Action space: 64 (4 rows × 4 cols × 4 values)
        """
        # Decode action
        row = (action // (self.size * self.size))
        col = (action // self.size) % self.size
        val = (action % self.size) + 1

        reward = 0.0
        done = False
        info = {}

        # Can't modify given cells
        if not self.mask[row, col]:
            reward = -0.5
            info['reason'] = 'immutable_cell'
        # Cell already filled
        elif self.grid[row, col] != 0:
            reward = -0.3
            info['reason'] = 'already_filled'
        # Valid placement
        elif self._is_valid_placement(row, col, val):
            self.grid[row, col] = val

            # Check if correct
            if val == self.solution[row, col]:
                reward = 1.0
                info['reason'] = 'correct_placement'
                self.filled_correctly += 1
            else:
                reward = 0.2  # Valid but wrong (still following rules)
                info['reason'] = 'valid_but_wrong'
        # Invalid placement (breaks rules)
        else:
            reward = -1.0
            info['reason'] = 'invalid_placement'

        self.steps += 1

        # Check if solved correctly
        if np.array_equal(self.grid, self.solution):
            reward += 10.0
            done = True
            info['reason'] = 'solved'

        # Check if all cells filled (might be wrong)
        elif np.all(self.grid > 0):
            reward -= 2.0  # Filled everything but wrong
            done = True
            info['reason'] = 'filled_incorrectly'

        # Timeout
        if self.steps > 50:
            done = True
            info['reason'] = 'timeout'

        return self._get_state(), reward, done, info

    def render(self):
        """Print the grid nicely"""
        print("\nCurrent Grid:")
        print("┌" + "─" * 9 + "┬" + "─" * 9 + "┐")
        for i in range(self.size):
            row_str = "│ "
            for j in range(self.size):
                val = self.grid[i, j]
                if val == 0:
                    row_str += ". "
                else:
                    # Color given cells differently
                    if self.mask[i, j]:
                        row_str += f"{val} "  # Agent placed
                    else:
                        row_str += f"\033[1m{val}\033[0m "  # Given (bold)

                if j == 1:
                    row_str += "│ "
            row_str += "│"
            print(row_str)

            if i == 1:
                print("├" + "─" * 9 + "┼" + "─" * 9 + "┤")
        print("└" + "─" * 9 + "┴" + "─" * 9 + "┘")

        # Show statistics
        total_empty = np.sum(self.initial_grid == 0)
        currently_filled = np.sum((self.grid != 0) & self.mask)
        print(f"Progress: {currently_filled}/{total_empty} cells filled")
        print(f"Correct: {self.filled_correctly}/{self.size*self.size}")


class SimplifiedSudokuEnv:
    """
    Even simpler: Only fill ONE missing cell per episode
    This makes learning much easier!
    """

    def __init__(self, size=4):
        self.size = size
        self.reset()

    def reset(self):
        """Generate a puzzle with only ONE cell missing"""
        # Start with solved grid
        self.solution = np.array([
            [1, 2, 3, 4],
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 1]
        ])

        # Remove ONE random cell
        self.target_row = random.randint(0, self.size - 1)
        self.target_col = random.randint(0, self.size - 1)
        self.target_val = self.solution[self.target_row, self.target_col]

        self.grid = self.solution.copy()
        self.grid[self.target_row, self.target_col] = 0

        return self._get_state()

    def _get_state(self):
        """Simple state: just the grid"""
        return torch.tensor(self.grid.flatten(), dtype=torch.float32) / self.size

    def step(self, action):
        """Action: choose which VALUE (1-4) to place in the missing cell"""
        # Action is just the value (0-3 mapped to 1-4)
        val = action + 1

        if val == self.target_val:
            reward = 10.0
            done = True
        else:
            reward = -1.0
            done = True  # Only one attempt per episode

        return self._get_state(), reward, done, {}


def test_environments():
    """Test both environments"""
    print("=" * 70)
    print("TESTING IMPROVED SUDOKU ENVIRONMENTS")
    print("=" * 70)

    # Test 1: Simplified (one cell)
    print("\n1. Simplified Sudoku (fill ONE cell)")
    print("-" * 70)
    env = SimplifiedSudokuEnv()
    state = env.reset()
    print(f"State shape: {state.shape}")
    print(f"Action space: {env.size} (values 1-4)")
    print(f"Target: Fill cell [{env.target_row}, {env.target_col}] with {env.target_val}")

    # Random agent
    correct = 0
    for _ in range(100):
        state = env.reset()
        action = random.randint(0, env.size - 1)
        _, reward, _, _ = env.step(action)
        if reward > 0:
            correct += 1
    print(f"Random agent success: {correct}% (should be ~25%)")

    # Test 2: Improved (multiple cells, easier)
    print("\n2. Improved Sudoku (multiple cells, difficulty=0.5)")
    print("-" * 70)
    env = ImprovedSudokuEnv(difficulty=0.5)
    state = env.reset()
    print(f"State shape: {state.shape}")
    print(f"Action space: {env.size ** 3} = 64")
    env.render()

    # Take a few random actions
    print("\nTaking 3 random valid actions...")
    for _ in range(3):
        # Find empty cell
        empty_cells = [(i, j) for i in range(env.size) for j in range(env.size)
                      if env.grid[i, j] == 0 and env.mask[i, j]]
        if empty_cells:
            i, j = random.choice(empty_cells)
            val = random.randint(1, env.size)
            action = i * (env.size * env.size) + j * env.size + (val - 1)
            state, reward, done, info = env.step(action)
            print(f"  Action: row={i}, col={j}, val={val} → reward={reward:.1f} ({info['reason']})")

    env.render()

    # Test 3: Very easy (difficulty=0.25)
    print("\n3. Very Easy Sudoku (difficulty=0.25)")
    print("-" * 70)
    env = ImprovedSudokuEnv(difficulty=0.25)
    state = env.reset()
    env.render()
    print(f"Only need to fill {np.sum(env.grid == 0)} cells!")

    print("\n" + "=" * 70)
    print("Recommendation: Start with SimplifiedSudokuEnv for initial testing!")
    print("=" * 70)


if __name__ == "__main__":
    test_environments()
