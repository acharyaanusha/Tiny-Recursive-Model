# Copyright 2025 Anusha Acharya
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# ---------------------------------------
# Tiny Recursive Model (policy backbone)
# ---------------------------------------
class TinyNetwork(nn.Module):
    def __init__(self, z_dim=64, x_dim=64, y_dim=32, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + x_dim + y_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim)
        )

    def forward(self, z, x, y):
        inp = torch.cat([z, x, y], dim=-1)
        return self.net(inp)

class OutputHead(nn.Module):
    def __init__(self, z_dim=64, y_dim=32, n_actions=10, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, z, y):
        inp = torch.cat([z, y], dim=-1)
        return self.net(inp)

class TRMPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, z_dim=64, y_dim=32, inner_steps=3, sup_steps=2):
        super().__init__()
        self.tiny = TinyNetwork(z_dim, state_dim, y_dim)
        self.head = OutputHead(z_dim, y_dim, action_dim)
        self.y_embed = nn.Embedding(action_dim, y_dim)
        self.z0 = nn.Parameter(torch.randn(1, 1, z_dim) * 0.1)
        self.inner_steps = inner_steps
        self.sup_steps = sup_steps

    def forward(self, state, prev_action=None):
        batch = state.shape[0]
        seq_len = 1  # we treat each observation as one "position"
        z = self.z0.expand(batch, seq_len, -1)  # (batch, 1, z_dim)

        if prev_action is None:
            prev_action = torch.zeros(batch, seq_len, dtype=torch.long, device=state.device)

        # Ensure prev_action is (batch, seq_len)
        if prev_action.dim() == 2 and prev_action.shape[1] == seq_len:
            y_emb = self.y_embed(prev_action)  # (batch, seq_len, y_dim)
        else:
            # Handle case where prev_action might have wrong shape
            prev_action = prev_action.view(batch, seq_len)
            y_emb = self.y_embed(prev_action)  # (batch, seq_len, y_dim)

        # Ensure x has shape (batch, seq_len, state_dim)
        if state.dim() == 2:
            x = state.unsqueeze(1)  # (batch, 1, state_dim)
        else:
            x = state  # assume already correct shape
        for _ in range(self.sup_steps):
            for _ in range(self.inner_steps):
                z = self.tiny(z, x, y_emb)
            logits = self.head(z, y_emb)
            probs = torch.softmax(logits, dim=-1)
            y_emb = probs @ self.y_embed.weight  # shape: (batch, seq_len, y_dim)
        return logits.squeeze(1)  # shape: (batch, action_dim)

# ---------------------------------------
# Sudoku Environment (4x4)
# ---------------------------------------
class SudokuEnv:
    def __init__(self, size=4):
        self.size = size
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        return torch.tensor(self.grid.flatten(), dtype=torch.float32)

    def _is_valid(self, row, col, val):
        if val in self.grid[row, :]: return False
        if val in self.grid[:, col]: return False
        box_size = int(np.sqrt(self.size))
        r0, c0 = row - row % box_size, col - col % box_size
        if val in self.grid[r0:r0+box_size, c0:c0+box_size]: return False
        return True

    def step(self, action):
        # Action encoding: pick (row,col,value)
        row = (action // (self.size * self.size)) % self.size
        col = (action // self.size) % self.size
        val = (action % self.size) + 1
        reward = -0.1
        done = False
        if self.grid[row, col] == 0 and self._is_valid(row, col, val):
            self.grid[row, col] = val
            reward = 1.0
        else:
            reward = -1.0
        self.steps += 1
        if np.all(self.grid > 0):
            done = True
            reward += 5.0
        if self.steps > 30:
            done = True
        return self._get_state(), reward, done

# ---------------------------------------
# Maze Environment (5x5)
# ---------------------------------------
class MazeEnv:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.pos = [0, 0]
        self.goal = [self.size - 1, self.size - 1]
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        grid = np.zeros((self.size, self.size))
        grid[self.pos[0], self.pos[1]] = 1
        grid[self.goal[0], self.goal[1]] = 0.5
        return torch.tensor(grid.flatten(), dtype=torch.float32)

    def step(self, action):
        moves = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}
        r, c = self.pos
        dr, dc = moves[action]
        r, c = r+dr, c+dc
        reward = -0.1
        done = False
        if 0 <= r < self.size and 0 <= c < self.size:
            self.pos = [r, c]
            if self.pos == self.goal:
                reward = 10.0
                done = True
        else:
            reward = -1.0
        self.steps += 1
        if self.steps > 50:
            done = True
        return self._get_state(), reward, done

# ---------------------------------------
# RL Training Loop (REINFORCE) - Improved
# ---------------------------------------
def train_rl(env, policy, n_episodes=300, gamma=0.99, lr=1e-3):
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    best_reward = -float('inf')
    success_count = 0
    recent_rewards = []

    for ep in range(n_episodes):
        state = env.reset()
        log_probs, rewards = [], []
        prev_action = None
        done = False

        while not done:
            logits = policy(state.unsqueeze(0), prev_action)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            next_state, reward, done = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
            prev_action = action.unsqueeze(0).unsqueeze(0)
            state = next_state

        # Compute returns (discounted cumulative rewards)
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns for stability (but only if we have variance)
        if len(returns) > 1 and returns.std() > 1e-6:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        # Policy gradient loss
        log_probs_tensor = torch.stack(log_probs)
        loss = -(log_probs_tensor * returns).mean()  # Use mean instead of sum

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        # Track statistics
        total_reward = sum(rewards)
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 20:
            recent_rewards.pop(0)

        if total_reward > best_reward:
            best_reward = total_reward

        # Check if reached goal (for MazeEnv)
        if isinstance(env, MazeEnv) and total_reward > 5:
            success_count += 1

        # Print progress
        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {ep+1}: reward={total_reward:.2f} | "
                  f"avg_20={avg_reward:.2f} | best={best_reward:.2f} | "
                  f"steps={len(rewards)} | success_rate={success_count/(ep+1)*100:.1f}%")
        elif ep < 5:  # Show first few episodes
            print(f"Episode {ep+1}: reward={total_reward:.2f} | steps={len(rewards)}")

# ---------------------------------------
# Example Usage
# ---------------------------------------
def run_experiment(env_type='maze', n_episodes=500):
    """
    Run TRM training on specified environment

    Args:
        env_type: 'maze' or 'sudoku'
        n_episodes: number of training episodes
    """
    print("=" * 70)
    print(f"TRM REINFORCEMENT LEARNING - {env_type.upper()}")
    print("=" * 70)

    # Create environment based on type
    if env_type.lower() == 'maze':
        env = MazeEnv(size=5)
        print("\nEnvironment: MazeEnv (5x5)")
        print("Goal: Navigate from (0,0) to (4,4)")
    elif env_type.lower() == 'sudoku':
        env = SudokuEnv(size=4)
        print("\nEnvironment: SudokuEnv (4x4)")
        print("Goal: Fill the grid following Sudoku rules")
    else:
        raise ValueError(f"Unknown environment type: {env_type}. Choose 'maze' or 'sudoku'")

    state_dim = env._get_state().numel()
    action_dim = 4 if isinstance(env, MazeEnv) else env.size**3

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Create policy with smaller architecture for faster learning
    policy = TRMPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        z_dim=32,      # Reduced from 64
        y_dim=16,      # Reduced from 32
        inner_steps=2, # Reduced from 3
        sup_steps=2    # Keep at 2
    )

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {num_params:,}")

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    # Train with better hyperparameters
    train_rl(
        env,
        policy,
        n_episodes=n_episodes,
        gamma=0.95,    # Slightly lower discount for shorter-term rewards
        lr=3e-3        # Higher learning rate for faster learning
    )

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE - {env_type.upper()}")
    print("=" * 70)

if __name__ == "__main__":
    import sys

    # Default to maze, but allow command-line override
    env_type = 'maze'
    n_episodes = 500

    if len(sys.argv) > 1:
        env_type = sys.argv[1].lower()
    if len(sys.argv) > 2:
        n_episodes = int(sys.argv[2])

    print("\nUsage: python trm_rl_evns.py [maze|sudoku|both] [n_episodes]")
    print("Examples:")
    print("  python trm_rl_evns.py maze 500")
    print("  python trm_rl_evns.py sudoku 500")
    print("  python trm_rl_evns.py both 300")
    print()

    # Run single or both environments
    if env_type == 'both':
        print("\n" + "=" * 70)
        print("RUNNING BOTH ENVIRONMENTS")
        print("=" * 70)

        run_experiment('maze', n_episodes)
        print("\n\n")
        run_experiment('sudoku', n_episodes)
    else:
        run_experiment(env_type, n_episodes)
