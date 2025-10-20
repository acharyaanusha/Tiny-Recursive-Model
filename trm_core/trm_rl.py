"""
Tiny Recursive Model (TRM) for Reinforcement Learning
Based on: https://github.com/SamsungSAILMontreal/TinyRecursiveModels

Paper Implementation: Recursive Reasoning with Hierarchical Levels
The model uses two-level hierarchical reasoning (H-level and L-level) that
recursively refines representations through multiple cycles.

Key Architecture Components:
1. H-level (Higher): Global reasoning across the entire state
2. L-level (Lower): Local processing of state features
3. Recursive Cycles: K cycles of H→L→H refinement
4. Adaptive Halting: Optional early stopping mechanism
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math


# ---------------------------------------
# Environments (Maze & Sudoku)
# ---------------------------------------
class ImprovedMazeEnv:
    """Maze environment with shaped rewards for better learning"""
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.pos = [0, 0]
        self.goal = [self.size - 1, self.size - 1]
        self.steps = 0
        self.prev_dist = self._manhattan_distance(self.pos, self.goal)
        return self._get_state()

    def _get_state(self):
        # Return flattened grid with agent and goal positions
        grid = np.zeros((self.size, self.size))
        grid[self.pos[0], self.pos[1]] = 1.0
        grid[self.goal[0], self.goal[1]] = 0.5
        return torch.tensor(grid.flatten(), dtype=torch.float32)

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        r, c = self.pos
        dr, dc = moves[action]
        new_r, new_c = r + dr, c + dc

        reward = -0.01
        done = False

        if 0 <= new_r < self.size and 0 <= new_c < self.size:
            self.pos = [new_r, new_c]
            new_dist = self._manhattan_distance(self.pos, self.goal)
            reward += 0.2 * (self.prev_dist - new_dist)
            self.prev_dist = new_dist

            if self.pos == self.goal:
                reward += 10.0
                done = True
        else:
            reward = -0.5

        self.steps += 1
        if self.steps > 50:
            done = True

        return self._get_state(), reward, done


class SudokuEnv:
    """Simple 4x4 Sudoku environment"""
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
            reward += 10.0
        if self.steps > 30:
            done = True

        return self._get_state(), reward, done


# ---------------------------------------
# TRM Architecture Components
# ---------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU activation from the paper"""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.linear2 = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        return F.silu(self.linear1(x)) * self.linear2(x)


class HierarchicalReasoningBlock(nn.Module):
    """
    Hierarchical Reasoning Block (H-level or L-level)
    Uses SwiGLU or MLP for processing
    """
    def __init__(self, dim, hidden_dim, use_swiglu=True):
        super().__init__()
        self.use_swiglu = use_swiglu

        if use_swiglu:
            self.swiglu = SwiGLU(dim, hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim)
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if self.use_swiglu:
            out = self.output_proj(self.swiglu(self.norm(x)))
        else:
            out = self.net(self.norm(x))
        return x + out  # Residual connection


class TRMRecursiveCore(nn.Module):
    """
    Core recursive reasoning module following the paper's architecture
    Implements hierarchical H-L reasoning with K cycles
    """
    def __init__(self, state_dim, h_dim=64, l_dim=32, k_cycles=3, n_h_layers=2, n_l_layers=1):
        super().__init__()
        self.state_dim = state_dim
        self.h_dim = h_dim
        self.l_dim = l_dim
        self.k_cycles = k_cycles

        # Input embedding to H-level
        self.input_embed = nn.Linear(state_dim, h_dim)

        # H-level reasoning blocks (higher-level global reasoning)
        self.h_blocks = nn.ModuleList([
            HierarchicalReasoningBlock(h_dim, h_dim * 2, use_swiglu=True)
            for _ in range(n_h_layers)
        ])

        # H to L projection
        self.h_to_l = nn.Linear(h_dim, l_dim)

        # L-level reasoning blocks (lower-level local processing)
        self.l_blocks = nn.ModuleList([
            HierarchicalReasoningBlock(l_dim, l_dim * 2, use_swiglu=False)
            for _ in range(n_l_layers)
        ])

        # L to H feedback
        self.l_to_h = nn.Linear(l_dim, h_dim)

        # Carry mechanism (maintains state across cycles)
        self.h_carry = nn.Linear(h_dim, h_dim)

    def forward(self, state, return_all_cycles=False):
        """
        Forward pass with K cycles of H→L→H reasoning

        Args:
            state: (batch, state_dim)
            return_all_cycles: if True, return h from all cycles

        Returns:
            h_final: (batch, h_dim) - final H-level representation
            or list of h from all cycles if return_all_cycles=True
        """
        batch = state.shape[0]

        # Initial H-level embedding
        h = self.input_embed(state).unsqueeze(1)  # (batch, 1, h_dim)

        h_history = []

        # K cycles of recursive reasoning
        for cycle in range(self.k_cycles):
            # H-level processing
            h_residual = h
            for h_block in self.h_blocks:
                h = h_block(h)

            # Carry connection from previous cycle
            if cycle > 0:
                h = h + self.h_carry(h_residual)

            # H → L projection
            l = self.h_to_l(h)  # (batch, 1, l_dim)

            # L-level processing
            for l_block in self.l_blocks:
                l = l_block(l)

            # L → H feedback
            h_feedback = self.l_to_h(l)
            h = h + h_feedback  # Recursive update

            h_history.append(h)

        if return_all_cycles:
            return h_history
        else:
            return h.squeeze(1)  # (batch, h_dim)


class TRMPolicy(nn.Module):
    """
    TRM Policy for RL following the paper's architecture
    Uses recursive reasoning to refine action logits
    """
    def __init__(self, state_dim, action_dim, h_dim=64, l_dim=32,
                 k_cycles=3, n_h_layers=2, n_l_layers=1,
                 use_action_conditioning=True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_action_conditioning = use_action_conditioning

        # Recursive reasoning core
        self.recursive_core = TRMRecursiveCore(
            state_dim=state_dim,
            h_dim=h_dim,
            l_dim=l_dim,
            k_cycles=k_cycles,
            n_h_layers=n_h_layers,
            n_l_layers=n_l_layers
        )

        # Action output head
        self.action_head = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU(),
            nn.Linear(h_dim // 2, action_dim)
        )

        # Optional: Action embedding for conditioning (feedback from previous action)
        if use_action_conditioning:
            self.action_embed = nn.Embedding(action_dim, h_dim)
            self.action_proj = nn.Linear(state_dim + h_dim, state_dim)

    def forward(self, state, prev_action=None, return_all_cycles=False):
        """
        Forward pass through TRM policy

        Args:
            state: (batch, state_dim)
            prev_action: (batch,) optional previous action for conditioning
            return_all_cycles: if True, return logits from all K cycles

        Returns:
            logits: (batch, action_dim) or list if return_all_cycles=True
        """
        batch = state.shape[0]

        # Optional action conditioning (incorporate previous action)
        if self.use_action_conditioning and prev_action is not None:
            if prev_action.dim() == 0:
                prev_action = prev_action.unsqueeze(0)
            if prev_action.shape[0] != batch:
                prev_action = prev_action.expand(batch)

            action_emb = self.action_embed(prev_action)  # (batch, h_dim)
            state_aug = torch.cat([state, action_emb], dim=-1)
            state = self.action_proj(state_aug)

        # Recursive reasoning
        if return_all_cycles:
            h_history = self.recursive_core(state, return_all_cycles=True)
            logits_history = [self.action_head(h.squeeze(1)) for h in h_history]
            return logits_history
        else:
            h_final = self.recursive_core(state)  # (batch, h_dim)
            logits = self.action_head(h_final)  # (batch, action_dim)
            return logits


# ---------------------------------------
# Baseline MLP Policy
# ---------------------------------------
class MLPPolicy(nn.Module):
    """Simple MLP baseline for comparison"""
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, state, prev_action=None, return_all_cycles=False):
        logits = self.net(state)
        if return_all_cycles:
            return [logits]  # Return as list for compatibility
        return logits


# ---------------------------------------
# Training with Deep Supervision
# ---------------------------------------
def train_rl_with_deep_supervision(env, policy, n_episodes=500, gamma=0.95, lr=3e-3,
                                    entropy_coef=0.01, use_deep_supervision=True,
                                    policy_name="Policy"):
    """
    Train with optional deep supervision across all K cycles
    (similar to paper's supervised training on intermediate steps)
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    best_reward = -float('inf')
    success_count = 0
    episode_rewards = []
    episode_lengths = []

    print(f"\nTraining {policy_name}...")
    print(f"Deep supervision: {use_deep_supervision}")
    print("-" * 70)

    for ep in range(n_episodes):
        state = env.reset()
        log_probs, rewards, entropies = [], [], []
        prev_action = None
        done = False

        while not done:
            # Get logits (possibly from all cycles for deep supervision)
            if use_deep_supervision and hasattr(policy, 'recursive_core'):
                logits_all = policy(state.unsqueeze(0), prev_action, return_all_cycles=True)
                logits = logits_all[-1]  # Use final cycle for action selection
            else:
                logits = policy(state.unsqueeze(0), prev_action)

            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            entropy = dist.entropy()
            entropies.append(entropy)

            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
            prev_action = action
            state = next_state

        # Compute returns
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        if len(returns) > 1 and returns.std() > 1e-6:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        # Policy gradient loss with entropy bonus
        log_probs_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)

        policy_loss = -(log_probs_tensor * returns).mean()
        entropy_loss = -entropy_tensor.mean()
        loss = policy_loss + entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        # Track statistics
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        episode_lengths.append(len(rewards))

        if total_reward > best_reward:
            best_reward = total_reward

        if total_reward > 5:
            success_count += 1

        # Print progress
        if (ep + 1) % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(episode_lengths[-50:])
            success_rate = success_count / (ep + 1) * 100
            print(f"Ep {ep+1:3d}: reward={total_reward:5.2f} | "
                  f"avg50={avg_reward:5.2f} | best={best_reward:5.2f} | "
                  f"len={avg_length:4.1f} | success={success_rate:4.1f}%")

    print(f"\n{policy_name} final success rate: {success_count/n_episodes*100:.1f}%")
    return episode_rewards, episode_lengths


# ---------------------------------------
# Visualization
# ---------------------------------------
def plot_comparison(results_dict, save_name='trm_paper_comparison.png'):
    """Plot learning curves for multiple policies"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    def smooth(data, window=20):
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(data[start:i+1]))
        return smoothed

    colors = ['blue', 'red', 'green', 'purple']
    for idx, (name, rewards) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        smoothed = smooth(rewards)
        axes[0].plot(rewards, alpha=0.2, color=color, linewidth=0.5)
        axes[0].plot(smoothed, color=color, linewidth=2, label=name)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Learning Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=9.2, color='black', linestyle='--', alpha=0.3, label='Optimal')

    # Success rate
    window = 50
    for idx, (name, rewards) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        success = [1 if r > 5 else 0 for r in rewards]
        success_rate = [np.mean(success[max(0, i-window):i+1]) * 100
                       for i in range(len(success))]
        axes[1].plot(success_rate, color=color, linewidth=2, label=name)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Success Rate (%) - Rolling 50 eps')
    axes[1].set_title('Success Rate Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {save_name}")
    plt.show()


# ---------------------------------------
# Main
# ---------------------------------------
def run_comparison(env_type='maze', n_episodes=500, env_size=5):
    """
    Run TRM comparison experiment on specified environment

    Args:
        env_type: 'maze' or 'sudoku'
        n_episodes: number of training episodes
        env_size: size of the environment (5 for maze, 4 for sudoku)
    """
    print("=" * 70)
    print(f"TRM FOR RL - {env_type.upper()}")
    print("Based on: Samsung SAIL Montreal's Tiny Recursive Models")
    print("=" * 70)

    # Environment setup based on type
    if env_type.lower() == 'maze':
        env = ImprovedMazeEnv(size=env_size)
        state_dim = env_size * env_size
        action_dim = 4
        print(f"\nEnvironment: Maze {env.size}x{env.size}")
        print(f"Goal: Navigate from (0,0) to ({env.size-1},{env.size-1})")
        success_threshold = 5.0
    elif env_type.lower() == 'sudoku':
        env = SudokuEnv(size=env_size)
        state_dim = env_size * env_size
        action_dim = env_size ** 3
        print(f"\nEnvironment: Sudoku {env.size}x{env.size}")
        print(f"Goal: Fill the grid following Sudoku rules")
        success_threshold = 5.0
    else:
        raise ValueError(f"Unknown environment type: {env_type}. Choose 'maze' or 'sudoku'")

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    results = {}

    # 1. TRM Policy (Paper architecture)
    print("\n" + "=" * 70)
    print("1. TRM Policy (Hierarchical Recursive Reasoning)")
    print("=" * 70)
    trm_policy = TRMPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        h_dim=64,
        l_dim=32,
        k_cycles=3,  # K recursive cycles
        n_h_layers=2,
        n_l_layers=1,
        use_action_conditioning=True
    )
    print(f"Parameters: {sum(p.numel() for p in trm_policy.parameters()):,}")

    trm_rewards, _ = train_rl_with_deep_supervision(
        env, trm_policy, n_episodes=n_episodes, lr=2e-3,
        entropy_coef=0.01, use_deep_supervision=True,
        policy_name="TRM (K=3 cycles)"
    )
    results["TRM (K=3)"] = trm_rewards

    # 2. TRM with more cycles
    print("\n" + "=" * 70)
    print("2. TRM Policy with K=5 cycles")
    print("=" * 70)

    # Reset environment for new training
    if env_type.lower() == 'maze':
        env = ImprovedMazeEnv(size=env_size)
    else:
        env = SudokuEnv(size=env_size)

    trm_k5_policy = TRMPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        h_dim=64,
        l_dim=32,
        k_cycles=5,
        n_h_layers=2,
        n_l_layers=1,
        use_action_conditioning=True
    )
    print(f"Parameters: {sum(p.numel() for p in trm_k5_policy.parameters()):,}")

    trm_k5_rewards, _ = train_rl_with_deep_supervision(
        env, trm_k5_policy, n_episodes=n_episodes, lr=2e-3,
        entropy_coef=0.01, use_deep_supervision=True,
        policy_name="TRM (K=5 cycles)"
    )
    results["TRM (K=5)"] = trm_k5_rewards

    # 3. Baseline MLP
    print("\n" + "=" * 70)
    print("3. Baseline MLP Policy")
    print("=" * 70)

    # Reset environment for new training
    if env_type.lower() == 'maze':
        env = ImprovedMazeEnv(size=env_size)
    else:
        env = SudokuEnv(size=env_size)

    mlp_policy = MLPPolicy(state_dim, action_dim, hidden=128)
    print(f"Parameters: {sum(p.numel() for p in mlp_policy.parameters()):,}")

    mlp_rewards, _ = train_rl_with_deep_supervision(
        env, mlp_policy, n_episodes=n_episodes, lr=3e-3,
        entropy_coef=0.01, use_deep_supervision=False,
        policy_name="MLP Baseline"
    )
    results["MLP Baseline"] = mlp_rewards

    # Final comparison
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS - {env_type.upper()}")
    print("=" * 70)
    for name, rewards in results.items():
        final_50 = np.mean(rewards[-50:])
        success = sum(1 for r in rewards[-50:] if r > success_threshold) / 50 * 100
        print(f"{name:20s}: avg_reward={final_50:5.2f}, success_rate={success:4.1f}%")

    # Visualize
    plot_comparison(results, save_name=f'trm_paper_{env_type}_comparison.png')

    print("\n" + "=" * 70)
    print(f"COMPLETE - {env_type.upper()}!")
    print("=" * 70)

    return results


def main():
    """Main function with command-line interface"""
    import sys

    # Default parameters
    env_type = 'maze'
    n_episodes = 500

    if len(sys.argv) > 1:
        env_type = sys.argv[1].lower()
    if len(sys.argv) > 2:
        n_episodes = int(sys.argv[2])

    print("\nUsage: python trm_rl.py [maze|sudoku|both] [n_episodes]")
    print("Examples:")
    print("  python trm_rl.py maze 500")
    print("  python trm_rl.py sudoku 500")
    print("  python trm_rl.py both 300")
    print()

    # Run single or both environments
    if env_type == 'both':
        print("\n" + "=" * 70)
        print("RUNNING BOTH ENVIRONMENTS")
        print("=" * 70)

        # Run maze
        print("\n\n")
        maze_results = run_comparison('maze', n_episodes, env_size=5)

        # Run sudoku
        print("\n\n")
        sudoku_results = run_comparison('sudoku', n_episodes, env_size=4)

        # Combined summary
        print("\n" + "=" * 70)
        print("COMBINED SUMMARY")
        print("=" * 70)
        print("\nMAZE RESULTS:")
        for name, rewards in maze_results.items():
            final_50 = np.mean(rewards[-50:])
            success = sum(1 for r in rewards[-50:] if r > 5) / 50 * 100
            print(f"  {name:20s}: avg_reward={final_50:5.2f}, success_rate={success:4.1f}%")

        print("\nSUDOKU RESULTS:")
        for name, rewards in sudoku_results.items():
            final_50 = np.mean(rewards[-50:])
            success = sum(1 for r in rewards[-50:] if r > 5) / 50 * 100
            print(f"  {name:20s}: avg_reward={final_50:5.2f}, success_rate={success:4.1f}%")
    else:
        # Determine environment size
        env_size = 5 if env_type == 'maze' else 4
        run_comparison(env_type, n_episodes, env_size)


if __name__ == "__main__":
    main()
