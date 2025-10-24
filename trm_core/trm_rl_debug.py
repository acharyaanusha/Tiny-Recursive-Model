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
Debug script to understand why TRM RL isn't learning well
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '/Users/anushaacharya/Projects/New_project/Tiny Recursive Model (TRM)')
from trm_rl_evns import TRMPolicy, MazeEnv

print("=" * 70)
print("TRM RL DEBUGGING - Understanding Poor Performance")
print("=" * 70)

# Create environment
env = MazeEnv(size=5)
state_dim = env._get_state().numel()
action_dim = 4

print(f"\nMaze Environment:")
print(f"  Size: 5x5")
print(f"  Start: [0, 0]")
print(f"  Goal: [4, 4]")
print(f"  State dim: {state_dim}")
print(f"  Action dim: {action_dim} (0=up, 1=down, 2=left, 3=right)")

# Test random policy performance
print("\n" + "-" * 70)
print("BASELINE: Random Policy Performance")
print("-" * 70)

random_rewards = []
for ep in range(10):
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done and steps < 50:
        action = np.random.randint(4)
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1

    random_rewards.append(total_reward)
    if ep < 3:
        print(f"Episode {ep+1}: reward={total_reward:.2f}, steps={steps}")

print(f"\nRandom policy avg reward: {np.mean(random_rewards):.2f}")
print(f"This is your baseline - TRM should do better than this!")

# Optimal path analysis
print("\n" + "-" * 70)
print("OPTIMAL PATH ANALYSIS")
print("-" * 70)

optimal_steps = 8  # Manhattan distance from [0,0] to [4,4]
optimal_reward = optimal_steps * (-0.1) + 10.0  # step penalty + goal reward
print(f"Optimal steps: {optimal_steps}")
print(f"Optimal reward: {optimal_reward:.2f}")

# Test untrained TRM policy
print("\n" + "-" * 70)
print("UNTRAINED TRM Policy")
print("-" * 70)

policy = TRMPolicy(state_dim=state_dim, action_dim=action_dim, z_dim=32, y_dim=16)

# Check initial output distribution
state = env.reset()
with torch.no_grad():
    logits = policy(state.unsqueeze(0), prev_action=None)
    probs = torch.softmax(logits, dim=-1)

print(f"\nInitial action probabilities: {probs[0].numpy()}")
print("(Should be roughly uniform ~0.25 each for random initialization)")

# Test a few episodes
untrained_rewards = []
for ep in range(5):
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False
    prev_action = None
    actions_taken = []

    while not done and steps < 50:
        with torch.no_grad():
            logits = policy(state.unsqueeze(0), prev_action)
            probs = torch.softmax(logits, dim=-1)
            action = torch.distributions.Categorical(probs).sample()

        state, reward, done = env.step(action.item())
        total_reward += reward
        actions_taken.append(action.item())
        prev_action = action.unsqueeze(0).unsqueeze(0)
        steps += 1

    untrained_rewards.append(total_reward)
    if ep < 3:
        print(f"Episode {ep+1}: reward={total_reward:.2f}, steps={steps}, actions={actions_taken[:10]}")

print(f"\nUntrained TRM avg reward: {np.mean(untrained_rewards):.2f}")

# Common issues analysis
print("\n" + "=" * 70)
print("COMMON ISSUES & WHY LEARNING IS HARD")
print("=" * 70)

print("""
1. SPARSE REWARDS
   - Agent only gets +10 when reaching goal
   - Everything else is negative (-0.1 per step, -1 for hitting walls)
   - In a 5x5 maze, reaching goal by chance is VERY rare
   - Solution: Add shaped rewards (distance-based)

2. LARGE ACTION SPACE EXPLORATION
   - With 4 actions and 50 steps, there are 4^50 possible trajectories
   - Random exploration is extremely inefficient
   - Solution: Use better exploration (epsilon-greedy, entropy bonus)

3. CREDIT ASSIGNMENT
   - Hard to know which actions led to success
   - REINFORCE has high variance
   - Solution: Use baseline, or switch to Actor-Critic

4. MODEL CAPACITY vs SAMPLE EFFICIENCY
   - TRM is complex and may need more samples to learn
   - Simpler policies (MLP) might learn faster initially
   - Solution: Reduce model complexity or increase training episodes

5. POOR STATE REPRESENTATION
   - Flattened grid may not capture spatial structure well
   - Solution: Use CNN or better feature extraction
""")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

print("""
QUICK FIXES TO TRY:

1. Add shaped rewards (distance-based):
   In MazeEnv.step():
   ```python
   old_dist = abs(old_pos[0] - goal[0]) + abs(old_pos[1] - goal[1])
   new_dist = abs(new_pos[0] - goal[0]) + abs(new_pos[1] - goal[1])
   reward += 0.1 * (old_dist - new_dist)  # reward for getting closer
   ```

2. Increase learning rate: lr=5e-3 or lr=1e-2

3. Add entropy bonus for exploration:
   ```python
   entropy = -(probs * probs.log()).sum()
   loss = policy_loss - 0.01 * entropy  # encourage exploration
   ```

4. Use simpler baseline first (MLP policy):
   - Compare TRM vs simple 2-layer MLP
   - If MLP learns but TRM doesn't, it's a model issue

5. Increase episodes: Try 1000-2000 episodes

6. Smaller maze: Start with 3x3 for faster learning
""")

print("\n" + "=" * 70)
print("Would you like me to implement these fixes? (y/n)")
print("=" * 70)

