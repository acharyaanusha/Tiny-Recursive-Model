"""
Improved TRM RL with fixes for better learning

Key improvements:
1. Shaped rewards (distance-based)
2. Entropy bonus for exploration
3. Better hyperparameters
4. Baseline MLP for comparison
5. Visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------
# Improved Maze Environment with Shaped Rewards
# ---------------------------------------
class ImprovedMazeEnv:
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
        grid = np.zeros((self.size, self.size))
        grid[self.pos[0], self.pos[1]] = 1.0  # agent position
        grid[self.goal[0], self.goal[1]] = 0.5  # goal
        return torch.tensor(grid.flatten(), dtype=torch.float32)

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        r, c = self.pos
        dr, dc = moves[action]
        new_r, new_c = r + dr, c + dc

        reward = -0.01  # small step penalty
        done = False

        # Check if valid move
        if 0 <= new_r < self.size and 0 <= new_c < self.size:
            self.pos = [new_r, new_c]

            # Distance-based reward shaping
            new_dist = self._manhattan_distance(self.pos, self.goal)
            reward += 0.2 * (self.prev_dist - new_dist)  # reward for getting closer
            self.prev_dist = new_dist

            # Goal reached
            if self.pos == self.goal:
                reward += 10.0
                done = True
        else:
            reward = -0.5  # penalty for hitting wall

        self.steps += 1
        if self.steps > 50:
            done = True

        return self._get_state(), reward, done


# ---------------------------------------
# TRM Policy
# ---------------------------------------
class TinyNetwork(nn.Module):
    def __init__(self, z_dim=32, x_dim=64, y_dim=16, hidden=64):
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
    def __init__(self, z_dim=32, y_dim=16, n_actions=4, hidden=32):
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
    def __init__(self, state_dim, action_dim, z_dim=32, y_dim=16, inner_steps=2, sup_steps=2):
        super().__init__()
        self.tiny = TinyNetwork(z_dim, state_dim, y_dim, hidden=64)
        self.head = OutputHead(z_dim, y_dim, action_dim, hidden=32)
        self.y_embed = nn.Embedding(action_dim, y_dim)
        self.z0 = nn.Parameter(torch.randn(1, 1, z_dim) * 0.1)
        self.inner_steps = inner_steps
        self.sup_steps = sup_steps

    def forward(self, state, prev_action=None):
        batch = state.shape[0]
        seq_len = 1
        z = self.z0.expand(batch, seq_len, -1)

        if prev_action is None:
            prev_action = torch.zeros(batch, seq_len, dtype=torch.long, device=state.device)

        if prev_action.dim() == 2 and prev_action.shape[1] == seq_len:
            y_emb = self.y_embed(prev_action)
        else:
            prev_action = prev_action.view(batch, seq_len)
            y_emb = self.y_embed(prev_action)

        if state.dim() == 2:
            x = state.unsqueeze(1)
        else:
            x = state

        for _ in range(self.sup_steps):
            for _ in range(self.inner_steps):
                z = self.tiny(z, x, y_emb)
            logits = self.head(z, y_emb)
            probs = torch.softmax(logits, dim=-1)
            y_emb = probs @ self.y_embed.weight

        return logits.squeeze(1)


# ---------------------------------------
# Baseline MLP Policy (for comparison)
# ---------------------------------------
class MLPPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, state, prev_action=None):
        # Ignore prev_action for simplicity
        return self.net(state)


# ---------------------------------------
# Improved Training with Entropy Bonus
# ---------------------------------------
def train_rl_improved(env, policy, n_episodes=500, gamma=0.95, lr=3e-3,
                      entropy_coef=0.01, policy_name="Policy"):
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    best_reward = -float('inf')
    success_count = 0
    episode_rewards = []
    episode_lengths = []

    print(f"\nTraining {policy_name}...")
    print("-" * 70)

    for ep in range(n_episodes):
        state = env.reset()
        log_probs, rewards, entropies = [], [], []
        prev_action = None
        done = False

        while not done:
            logits = policy(state.unsqueeze(0), prev_action)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            # Calculate entropy for exploration
            entropy = dist.entropy()
            entropies.append(entropy)

            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
            prev_action = action.unsqueeze(0).unsqueeze(0)
            state = next_state

        # Compute discounted returns
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
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

        if total_reward > 5:  # Reached goal
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
def plot_comparison(trm_rewards, mlp_rewards):
    """Plot learning curves for comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Smooth rewards with moving average
    def smooth(data, window=20):
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(data[start:i+1]))
        return smoothed

    trm_smooth = smooth(trm_rewards)
    mlp_smooth = smooth(mlp_rewards)

    # Plot raw rewards
    axes[0].plot(trm_rewards, alpha=0.3, color='blue', linewidth=0.5)
    axes[0].plot(mlp_rewards, alpha=0.3, color='red', linewidth=0.5)
    axes[0].plot(trm_smooth, color='blue', linewidth=2, label='TRM')
    axes[0].plot(mlp_smooth, color='red', linewidth=2, label='MLP')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Learning Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=9.2, color='green', linestyle='--', alpha=0.5, label='Optimal (~9.2)')

    # Plot success rate over time
    window = 50
    trm_success = [1 if r > 5 else 0 for r in trm_rewards]
    mlp_success = [1 if r > 5 else 0 for r in mlp_rewards]

    trm_success_rate = [np.mean(trm_success[max(0, i-window):i+1]) * 100
                        for i in range(len(trm_success))]
    mlp_success_rate = [np.mean(mlp_success[max(0, i-window):i+1]) * 100
                        for i in range(len(mlp_success))]

    axes[1].plot(trm_success_rate, color='blue', linewidth=2, label='TRM')
    axes[1].plot(mlp_success_rate, color='red', linewidth=2, label='MLP')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Success Rate (%) - Rolling 50 eps')
    axes[1].set_title('Success Rate Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trm_vs_mlp_learning.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: trm_vs_mlp_learning.png")
    plt.show()


# ---------------------------------------
# Main Comparison
# ---------------------------------------
def main():
    print("=" * 70)
    print("IMPROVED TRM RL - WITH FIXES")
    print("=" * 70)
    print("\nImprovements:")
    print("  ✓ Shaped rewards (distance-based)")
    print("  ✓ Entropy bonus for exploration")
    print("  ✓ Better hyperparameters")
    print("  ✓ Comparison with baseline MLP")
    print("=" * 70)

    # Setup
    env = ImprovedMazeEnv(size=5)
    state_dim = 25  # 5x5 grid flattened
    action_dim = 4

    # Train TRM
    print("\n1. Training TRM Policy")
    trm_policy = TRMPolicy(state_dim, action_dim, z_dim=32, y_dim=16,
                          inner_steps=2, sup_steps=2)
    trm_rewards, trm_lengths = train_rl_improved(
        env, trm_policy, n_episodes=500, lr=3e-3,
        entropy_coef=0.01, policy_name="TRM"
    )

    # Train MLP baseline
    print("\n2. Training MLP Baseline")
    mlp_policy = MLPPolicy(state_dim, action_dim, hidden=64)
    mlp_rewards, mlp_lengths = train_rl_improved(
        env, mlp_policy, n_episodes=500, lr=3e-3,
        entropy_coef=0.01, policy_name="MLP"
    )

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\nTRM - Final 50 episodes avg reward: {np.mean(trm_rewards[-50:]):.2f}")
    print(f"MLP - Final 50 episodes avg reward: {np.mean(mlp_rewards[-50:]):.2f}")

    trm_params = sum(p.numel() for p in trm_policy.parameters())
    mlp_params = sum(p.numel() for p in mlp_policy.parameters())
    print(f"\nTRM parameters: {trm_params:,}")
    print(f"MLP parameters: {mlp_params:,}")

    # Visualize
    plot_comparison(trm_rewards, mlp_rewards)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
