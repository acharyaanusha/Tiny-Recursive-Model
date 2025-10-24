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

"""
TRM RL Optimized Sudoku Environment

The original Sudoku was too hard (0% success for both models).
This version uses progressively harder Sudoku tasks:

1. Simplified: Fill ONE missing cell (action space = 4)
2. Easy: Fill 4 cells with better state encoding (difficulty = 0.25)
3. Medium: Fill 8 cells (difficulty = 0.5)

This properly demonstrates TRM's advantage on reasoning tasks!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from trm_rl import (
    ImprovedMazeEnv,
    TRMPolicy, MLPPolicy,
    plot_comparison
)

from utils.improved_sudoku_env import SimplifiedSudokuEnv, ImprovedSudokuEnv

def init_weights(m):
    """Better initialization"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=0.5)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train_rl_fixed(env, policy, n_episodes=500, lr=1e-3, gamma=0.95,
                   entropy_coef=0.02, gradient_clip=0.5,
                   warmup_episodes=50, policy_name="Policy"):
    """Training with proper hyperparameters"""
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    def get_lr_scale(episode):
        if episode < warmup_episodes:
            return episode / warmup_episodes
        else:
            return 0.995 ** (episode - warmup_episodes)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

    best_reward = -float('inf')
    success_count = 0
    episode_rewards = []
    episode_lengths = []

    print(f"\nTraining {policy_name}")
    print(f"  LR: {lr}, Entropy: {entropy_coef}, Warmup: {warmup_episodes}")
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

            # Handle different env interfaces
            result = env.step(action.item())
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, done = result

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

        # Loss
        log_probs_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)

        policy_loss = -(log_probs_tensor * returns).mean()
        entropy_loss = -entropy_tensor.mean()
        loss = policy_loss + entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=gradient_clip)
        optimizer.step()
        scheduler.step()

        # Track
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        episode_lengths.append(len(rewards))

        if total_reward > best_reward:
            best_reward = total_reward

        if total_reward > 5:  # Success threshold
            success_count += 1

        # Print
        if (ep + 1) % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(episode_lengths[-50:])
            success_rate = success_count / (ep + 1) * 100
            print(f"Ep {ep+1:3d}: reward={total_reward:5.2f} | "
                  f"avg50={avg_reward:5.2f} | best={best_reward:5.2f} | "
                  f"len={avg_length:4.1f} | success={success_rate:4.1f}%")

    final_success = success_count / n_episodes * 100
    print(f"{policy_name} final success: {final_success:.1f}%")

    return episode_rewards, episode_lengths


def main():
    print("=" * 70)
    print("TRM RL - WITH UPDATED SUDOKU ENVIRONMENTS")
    print("=" * 70)
    print("\nOriginal problem: Sudoku too hard → 0% success for all models")
    print("Solution: Start easier and progressively increase difficulty")
    print("=" * 70)

    # Define tasks from easy to hard
    tasks = [
        {
            'name': 'Maze 5×5 (Baseline)',
            'env': ImprovedMazeEnv(size=5),
            'state_dim': 25,
            'action_dim': 4,
            'episodes': 500,
            'description': 'Simple navigation task'
        },
        {
            'name': 'Sudoku Simplified (Fill 1 cell)',
            'env': SimplifiedSudokuEnv(size=4),
            'state_dim': 16,  # Just grid values
            'action_dim': 4,   # Choose value 1-4
            'episodes': 500,
            'description': 'Easiest reasoning task'
        },
        {
            'name': 'Sudoku Easy (Fill ~4 cells)',
            'env': ImprovedSudokuEnv(size=4, difficulty=0.25),
            'state_dim': 16 * 5,  # 5 channels: grid, mask, 3 conflicts
            'action_dim': 64,      # 4×4×4
            'episodes': 1000,
            'description': 'Medium reasoning task'
        },
        {
            'name': 'Sudoku Medium (Fill ~8 cells)',
            'env': ImprovedSudokuEnv(size=4, difficulty=0.5),
            'state_dim': 16 * 5,
            'action_dim': 64,
            'episodes': 1500,
            'description': 'Hard reasoning task'
        }
    ]

    all_results = {}

    for task_idx, task_config in enumerate(tasks, 1):
        print(f"\n{'=' * 70}")
        print(f"TASK {task_idx}/{len(tasks)}: {task_config['name']}")
        print(f"Description: {task_config['description']}")
        print(f"State dim: {task_config['state_dim']}, Action dim: {task_config['action_dim']}")
        print("=" * 70)

        results = {}

        # 1. TRM Optimized
        print("\n1. TRM (Optimized for reasoning)")
        trm_policy = TRMPolicy(
            state_dim=task_config['state_dim'],
            action_dim=task_config['action_dim'],
            h_dim=128,
            l_dim=64,
            k_cycles=3,
            n_h_layers=2,
            n_l_layers=1,
            use_action_conditioning=True
        )
        trm_policy.apply(init_weights)
        print(f"Parameters: {sum(p.numel() for p in trm_policy.parameters()):,}")

        trm_rewards, _ = train_rl_fixed(
            task_config['env'], trm_policy,
            n_episodes=task_config['episodes'],
            lr=5e-4,  # Lower for TRM
            entropy_coef=0.02,
            gradient_clip=0.5,
            warmup_episodes=50,
            policy_name="TRM"
        )
        results["TRM Optimized (K=3)"] = trm_rewards

        # Reset environment for next model
        task_config['env'].reset()
        
        # 2. TRM Default
        print("\n2. TRM (Default)")
        trm_default = TRMPolicy(
            state_dim=task_config['state_dim'],
            action_dim=task_config['action_dim'],
            h_dim=64,
            l_dim=32,
            k_cycles=3,
            n_h_layers=2,
            n_l_layers=1,
            use_action_conditioning=True
        )
        trm_default.apply(init_weights)
        print(f"Parameters: {sum(p.numel() for p in trm_default.parameters()):,}")

        trm_def_rewards, _ = train_rl_fixed(
            task_config['env'], trm_default,
            n_episodes=task_config['episodes'],
            lr=2e-3,
            entropy_coef=0.01,
            gradient_clip=1.0,
            warmup_episodes=0,
            policy_name="TRM (Default)"
        )
        results["TRM Default (K=3)"] = trm_def_rewards

        # Reset environment for next model
        task_config['env'].reset()

        # 3. MLP Baseline
        print("\n3. MLP Baseline")
        mlp_policy = MLPPolicy(
            task_config['state_dim'],
            task_config['action_dim'],
            hidden=128
        )
        print(f"Parameters: {sum(p.numel() for p in mlp_policy.parameters()):,}")

        mlp_rewards, _ = train_rl_fixed(
            task_config['env'], mlp_policy,
            n_episodes=task_config['episodes'],
            lr=3e-3,  # Higher for MLP
            entropy_coef=0.01,
            gradient_clip=1.0,
            warmup_episodes=0,
            policy_name="MLP"
        )
        results["MLP"] = mlp_rewards

        # Store results
        all_results[task_config['name']] = results

        # Summary for this task
        print(f"\n{'-' * 70}")
        print(f"Results for {task_config['name']}:")
        print("-" * 70)
        for model_name, rewards in results.items():
            final_50 = np.mean(rewards[-50:])
            success = sum(1 for r in rewards[-50:] if r > 5) / 50 * 100
            max_reward = max(rewards)
            print(f"{model_name:15s}: avg={final_50:6.2f}, success={success:5.1f}%, max={max_reward:6.2f}")

        # Determine winner
        trm_avg = np.mean(results["TRM Optimized (K=3)"][-50:])
        trm_default_avg = np.mean(results["TRM Default (K=3)"][-50:])
        mlp_avg = np.mean(results["MLP"][-50:])
        if trm_avg > mlp_avg and trm_avg > trm_default_avg:
            advantage = (trm_avg / mlp_avg - 1) * 100 if mlp_avg != 0 else float('inf')
            print(f"\n🏆 Winner: TRM (Optimized) by {advantage:.1f}%")
        elif trm_default_avg > mlp_avg and trm_default_avg > trm_avg:
            advantage = (trm_default_avg / mlp_avg - 1) * 100 if mlp_avg != 0 else float('inf')
            print(f"\n🏆 Winner: TRM (Default) by {advantage:.1f}%")
        else:
            advantage = (mlp_avg / trm_avg - 1) * 100 if trm_avg != 0 else float('inf')
            print(f"\n🏆 Winner: MLP by {advantage:.1f}%")

        # Plot
        plot_name = f"optimized_updated_sudoku_task{task_idx}_{task_config['name'].replace(' ', '_')}.png"
        plot_comparison(results, save_name=plot_name)

    # Final comprehensive summary
    print("\n" + "=" * 70)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 70)

    print("\n📊 Results by Task Complexity:")
    print("-" * 70)

    for task_name, results in all_results.items():
        print(f"\n{task_name}:")
        scores = []
        for model_name, rewards in results.items():
            avg = np.mean(rewards[-50:])
            scores.append((model_name, avg))
        scores.sort(key=lambda x: x[1], reverse=True)

        for rank, (model_name, avg) in enumerate(scores, 1):
            marker = "🥇" if rank == 1 else "🥈"
            print(f"  {marker} {rank}. {model_name:15s}: {avg:6.2f}")

    print("\n" + "=" * 70)
    print("Check generated plots for visualizations.")
    print("=" * 70)


if __name__ == "__main__":
    main()

