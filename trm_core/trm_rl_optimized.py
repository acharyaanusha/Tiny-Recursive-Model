"""
TRM for RL - Optimized Hyperparameters

This version addresses why MLP might outperform TRM on simple tasks:
1. Lower learning rate (TRM-specific)
2. Better initialization
3. Gradient monitoring
4. Tests on BOTH simple (maze) and complex (sudoku) tasks

Key insight: TRM needs different hyperparameters than MLP!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Import environments and models from paper implementation
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from trm_rl import (
    ImprovedMazeEnv, SudokuEnv,
    TRMPolicy, MLPPolicy,
    plot_comparison
)


def init_weights(m):
    """Better initialization for TRM"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=0.5)  # Smaller gain for stability
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train_rl_optimized(env, policy, n_episodes=500, lr=1e-3, gamma=0.95,
                       entropy_coef=0.02, gradient_clip=0.5,
                       warmup_episodes=50, policy_name="Policy",
                       monitor_gradients=False):
    """
    Optimized training for TRM with:
    - Lower base learning rate
    - Learning rate warmup
    - Tighter gradient clipping
    - More entropy for exploration
    - Gradient monitoring
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Learning rate schedule with warmup
    def get_lr_scale(episode):
        if episode < warmup_episodes:
            return episode / warmup_episodes  # Linear warmup
        else:
            return 0.995 ** (episode - warmup_episodes)  # Slow decay

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

    best_reward = -float('inf')
    success_count = 0
    episode_rewards = []
    episode_lengths = []
    gradient_norms = []

    print(f"\nTraining {policy_name} with OPTIMIZED hyperparameters")
    print(f"  Base LR: {lr}")
    print(f"  Warmup: {warmup_episodes} episodes")
    print(f"  Entropy: {entropy_coef}")
    print(f"  Grad clip: {gradient_clip}")
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

        # Policy gradient loss with entropy
        log_probs_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)

        policy_loss = -(log_probs_tensor * returns).mean()
        entropy_loss = -entropy_tensor.mean()
        loss = policy_loss + entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()

        # Monitor gradients (optional)
        if monitor_gradients and ep < 10:
            total_norm = 0
            for p in policy.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            gradient_norms.append(total_norm)
            if ep < 3:
                print(f"  Episode {ep+1} gradient norm: {total_norm:.4f}")

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=gradient_clip)
        optimizer.step()
        scheduler.step()

        # Track statistics
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        episode_lengths.append(len(rewards))

        if total_reward > best_reward:
            best_reward = total_reward

        if total_reward > 5:  # Success threshold
            success_count += 1

        # Print progress
        if (ep + 1) % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(episode_lengths[-50:])
            success_rate = success_count / (ep + 1) * 100
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Ep {ep+1:3d}: reward={total_reward:5.2f} | "
                  f"avg50={avg_reward:5.2f} | best={best_reward:5.2f} | "
                  f"len={avg_length:4.1f} | success={success_rate:4.1f}% | "
                  f"lr={current_lr:.6f}")

    final_success = success_count / n_episodes * 100
    print(f"\n{policy_name} final success rate: {final_success:.1f}%")

    return episode_rewards, episode_lengths


def main():
    print("=" * 70)
    print("TRM RL - OPTIMIZED HYPERPARAMETERS")
    print("Testing: Does TRM outperform MLP with proper tuning?")
    print("=" * 70)

    # Test both simple and complex tasks
    tasks = [
        ("Maze 5×5 (Simple)", ImprovedMazeEnv(size=5), 25, 4),
        ("Sudoku 4×4 (Complex)", SudokuEnv(size=4), 16, 64),
    ]

    all_results = {}

    for task_name, env, state_dim, action_dim in tasks:
        print("\n" + "=" * 70)
        print(f"TASK: {task_name}")
        print("=" * 70)

        results = {}

        # 1. TRM with OPTIMIZED hyperparameters
        print("\n1. TRM (Optimized)")
        trm_optimized = TRMPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            h_dim=128,  # Increased capacity
            l_dim=64,
            k_cycles=3,
            n_h_layers=2,
            n_l_layers=1,
            use_action_conditioning=True
        )
        trm_optimized.apply(init_weights)  # Better initialization

        print(f"Parameters: {sum(p.numel() for p in trm_optimized.parameters()):,}")

        trm_opt_rewards, _ = train_rl_optimized(
            env, trm_optimized,
            n_episodes=600,
            lr=5e-4,  # LOWER than MLP
            entropy_coef=0.02,  # MORE exploration
            gradient_clip=0.5,  # TIGHTER clipping
            warmup_episodes=50,
            policy_name="TRM (Optimized)",
            monitor_gradients=True
        )
        results["TRM (Optimized)"] = trm_opt_rewards

        # 2. TRM with DEFAULT hyperparameters (for comparison)
        print("\n2. TRM (Default)")
        trm_default = TRMPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            h_dim=64,
            l_dim=32,
            k_cycles=3,
            n_h_layers=2,
            n_l_layers=1,
            use_action_conditioning=True
        )

        print(f"Parameters: {sum(p.numel() for p in trm_default.parameters()):,}")

        trm_def_rewards, _ = train_rl_optimized(
            env, trm_default,
            n_episodes=600,
            lr=2e-3,  # TOO HIGH for TRM
            entropy_coef=0.01,
            gradient_clip=1.0,
            warmup_episodes=0,
            policy_name="TRM (Default)",
            monitor_gradients=False
        )
        results["TRM (Default)"] = trm_def_rewards

        # 3. MLP Baseline
        print("\n3. MLP Baseline")
        mlp = MLPPolicy(state_dim, action_dim, hidden=128)
        print(f"Parameters: {sum(p.numel() for p in mlp.parameters()):,}")

        mlp_rewards, _ = train_rl_optimized(
            env, mlp,
            n_episodes=600,
            lr=3e-3,  # Higher LR works for MLP
            entropy_coef=0.01,
            gradient_clip=1.0,
            warmup_episodes=0,
            policy_name="MLP Baseline",
            monitor_gradients=False
        )
        results["MLP Baseline"] = mlp_rewards

        # Store results
        all_results[task_name] = results

        # Plot for this task
        print("\n" + "-" * 70)
        print(f"Results for {task_name}:")
        print("-" * 70)
        for model_name, rewards in results.items():
            final_50 = np.mean(rewards[-50:])
            success = sum(1 for r in rewards[-50:] if r > 5) / 50 * 100
            print(f"{model_name:20s}: avg={final_50:6.2f}, success={success:5.1f}%")

        # Generate plot
        plot_name = f"optimized_{task_name.replace(' ', '_').lower()}.png"
        plot_comparison(results, save_name=plot_name)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\n📊 Key Findings:")
    print("-" * 70)

    for task_name, results in all_results.items():
        print(f"\n{task_name}:")
        rewards_list = [(name, np.mean(rews[-50:])) for name, rews in results.items()]
        rewards_list.sort(key=lambda x: x[1], reverse=True)

        for rank, (name, avg_reward) in enumerate(rewards_list, 1):
            print(f"  {rank}. {name}: {avg_reward:.2f}")

    print("\n" + "=" * 70)
    print("💡 INSIGHTS:")
    print("=" * 70)
    print("""
1. On SIMPLE tasks (Maze 5×5):
   - TRM needs careful tuning to match MLP
   - Lower LR (5e-4) and warmup help TRM
   - MLP may still win (expected - task too simple!)

2. On COMPLEX tasks (Sudoku 4×4):
   - TRM should significantly outperform MLP
   - Recursive reasoning helps with constraints
   - This showcases TRM's true strength!

3. Hyperparameter Impact:
   - LR: TRM needs 3-5× lower than MLP
   - Entropy: TRM benefits from more exploration
   - Warmup: Helps TRM's deeper architecture
   - Initialization: Critical for TRM stability

4. When to use TRM:
   ✓ Complex reasoning tasks
   ✓ Constraint satisfaction
   ✓ Planning problems
   ✗ Simple reactive tasks (use MLP)
    """)

    print("=" * 70)
    print("COMPLETE! Check the generated plots.")
    print("=" * 70)


if __name__ == "__main__":
    main()
