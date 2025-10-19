"""
Simple Maze Comparison: Baseline CNN vs TRM

A simplified comparison focusing on the core architectural differences.
This script trains a simple CNN baseline and TRM on a maze pathfinding task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ----------------------------
# Simple Maze Dataset
# ----------------------------

class SimpleMazeDataset(Dataset):
    """Simple maze dataset with random obstacles"""

    def __init__(self, n_samples=200, size=8):
        self.n_samples = n_samples
        self.size = size
        self.data = []

        print(f"Generating {n_samples} simple mazes...")
        for _ in range(n_samples):
            maze, path = self._generate_maze()
            self.data.append((maze, path))

    def _generate_maze(self):
        """Generate a simple maze with random obstacles"""
        maze = np.zeros((self.size, self.size), dtype=np.float32)

        # Add random obstacles (but not too many)
        n_obstacles = np.random.randint(5, 15)
        for _ in range(n_obstacles):
            y, x = np.random.randint(1, self.size-1, 2)
            maze[y, x] = 1  # obstacle

        # Define start and goal
        start = (0, 0)
        goal = (self.size - 1, self.size - 1)
        maze[start] = 0
        maze[goal] = 0

        # Find path with BFS
        path = self._find_path(maze, start, goal)

        # Create input (3 channels: maze, start marker, goal marker)
        maze_input = np.zeros((3, self.size, self.size), dtype=np.float32)
        maze_input[0] = maze
        maze_input[1, start[0], start[1]] = 1
        maze_input[2, goal[0], goal[1]] = 1

        # Create target (binary: on path or not)
        target = np.zeros((self.size, self.size), dtype=np.int64)
        if path:
            for y, x in path:
                target[y, x] = 1

        return torch.from_numpy(maze_input), torch.from_numpy(target)

    def _find_path(self, maze, start, goal):
        """BFS pathfinding"""
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            (y, x), path = queue.popleft()
            if (y, x) == goal:
                return path

            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.size and 0 <= nx < self.size and
                    maze[ny, nx] == 0 and (ny, nx) not in visited):
                    visited.add((ny, nx))
                    queue.append(((ny, nx), path + [(ny, nx)]))

        return [(0, 0), (self.size-1, self.size-1)]  # fallback

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


# ----------------------------
# Baseline: Simple CNN
# ----------------------------

class CNNMazeSolver(nn.Module):
    """Baseline CNN for maze solving"""

    def __init__(self, size=8):
        super().__init__()
        self.size = size

        self.conv_layers = nn.Sequential(
            # 3 input channels (maze, start, goal) -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output: 2 classes per pixel (on_path, not_on_path)
            nn.Conv2d(32, 2, kernel_size=1)
        )

    def forward(self, x):
        """
        x: (batch, 3, H, W)
        returns: (batch, H, W, 2)
        """
        logits = self.conv_layers(x)  # (batch, 2, H, W)
        return logits.permute(0, 2, 3, 1)  # (batch, H, W, 2)


# ----------------------------
# TRM Components (simplified)
# ----------------------------

class TinyRecursiveNet(nn.Module):
    """The core recursive network of TRM"""

    def __init__(self, z_dim=32, x_dim=16, y_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + x_dim + y_dim, 64),
            nn.ReLU(),
            nn.Linear(64, z_dim)
        )

    def forward(self, z, x, y_embed):
        return self.net(torch.cat([z, x, y_embed], dim=-1))


class SimpleTRM(nn.Module):
    """Simplified TRM for maze solving"""

    def __init__(self, size=8, z_dim=32, x_dim=16, y_dim=8,
                 inner_steps=3, sup_steps=5):
        super().__init__()
        self.size = size
        self.z_dim = z_dim

        # Input encoder
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, x_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # TRM components
        self.tiny_net = TinyRecursiveNet(z_dim, x_dim, y_dim)
        self.output_head = nn.Sequential(
            nn.Linear(z_dim + y_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        # Embeddings
        self.y_embedding = nn.Embedding(2, y_dim)
        self.z0 = nn.Parameter(torch.randn(1, 1, z_dim) * 0.1)

        self.inner_steps = inner_steps
        self.sup_steps = sup_steps

    def forward(self, x):
        """
        x: (batch, 3, H, W)
        returns: list of (batch, H, W, 2) for each supervision step
        """
        batch = x.size(0)

        # Encode input
        x_encoded = self.input_conv(x)  # (batch, x_dim, H, W)
        x_flat = x_encoded.view(batch, x_encoded.size(1), -1).permute(0, 2, 1)
        # x_flat: (batch, H*W, x_dim)

        seq_len = x_flat.size(1)

        # Initialize latent
        z = self.z0.expand(batch, seq_len, self.z_dim).clone()

        # Initialize predictions
        y_logits = torch.zeros(batch, seq_len, 2, device=x.device)

        outputs = []

        # Recursive refinement
        for sup_step in range(self.sup_steps):
            # Inner recursion: refine latent z
            for inner_step in range(self.inner_steps):
                # Get soft embedding of current prediction
                y_prob = torch.softmax(y_logits.detach(), dim=-1)
                y_embed = torch.matmul(y_prob, self.y_embedding.weight)

                # Update latent
                z_new = self.tiny_net(z, x_flat, y_embed)

                # Detach for stability (except last inner step)
                if inner_step < self.inner_steps - 1:
                    z = z_new.detach()
                else:
                    z = z_new

            # Outer update: generate new predictions
            y_prob = torch.softmax(y_logits, dim=-1)
            y_embed = torch.matmul(y_prob, self.y_embedding.weight)
            y_logits = self.output_head(torch.cat([z, y_embed], dim=-1))

            # Reshape to grid and store
            logits_grid = y_logits.view(batch, self.size, self.size, 2)
            outputs.append(logits_grid)

        return outputs


# ----------------------------
# Training and Visualization
# ----------------------------

def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs=20, is_trm=False, model_name="Model"):
    """Train a model and return history"""
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_acc': []}

    print(f"\nTraining {model_name}...")
    print("-" * 50)

    for epoch in range(1, n_epochs + 1):
        # Training
        model.train()
        total_loss = 0

        for x, y_true in train_loader:
            x, y_true = x.to(device), y_true.to(device)
            optimizer.zero_grad()

            if is_trm:
                # TRM: multiple outputs
                outputs = model(x)
                loss = sum(
                    criterion(out.permute(0, 3, 1, 2), y_true) * (1 + 0.1 * i)
                    for i, out in enumerate(outputs)
                ) / len(outputs)
            else:
                # CNN: single output
                output = model(x)
                loss = criterion(output.permute(0, 3, 1, 2), y_true)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y_true in val_loader:
                x, y_true = x.to(device), y_true.to(device)

                if is_trm:
                    outputs = model(x)
                    pred = outputs[-1].argmax(dim=-1)
                else:
                    pred = model(x).argmax(dim=-1)

                correct += (pred == y_true).sum().item()
                total += y_true.numel()

        val_acc = correct / total

        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}: loss={avg_loss:.4f}, val_acc={val_acc*100:.2f}%")

    return history


def visualize_results(cnn_model, trm_model, test_dataset, device):
    """Visualize predictions from both models"""
    cnn_model.eval()
    trm_model.eval()

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))

    for idx in range(3):
        x, y_true = test_dataset[idx]
        x_batch = x.unsqueeze(0).to(device)

        with torch.no_grad():
            cnn_pred = cnn_model(x_batch)[0].argmax(dim=-1).cpu()
            trm_outputs = trm_model(x_batch)
            trm_pred = trm_outputs[-1][0].argmax(dim=-1).cpu()

        # Extract components
        maze = x[0].numpy()
        start = x[1].numpy()
        goal = x[2].numpy()

        # Create display
        display = np.ones((*maze.shape, 3))
        display[maze == 0] = [1, 1, 1]  # white
        display[maze == 1] = [0.3, 0.3, 0.3]  # gray obstacles
        display[start == 1] = [0, 1, 0]  # green start
        display[goal == 1] = [1, 0, 0]  # red goal

        # Plot
        axes[idx, 0].imshow(display)
        axes[idx, 0].set_title('Input Maze' if idx == 0 else '')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(y_true.numpy(), cmap='Blues', vmin=0, vmax=1)
        axes[idx, 1].set_title('Ground Truth' if idx == 0 else '')
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(cnn_pred.numpy(), cmap='Blues', vmin=0, vmax=1)
        axes[idx, 2].set_title('CNN Prediction' if idx == 0 else '')
        axes[idx, 2].axis('off')

        axes[idx, 3].imshow(trm_pred.numpy(), cmap='Blues', vmin=0, vmax=1)
        axes[idx, 3].set_title('TRM Prediction' if idx == 0 else '')
        axes[idx, 3].axis('off')

    plt.tight_layout()
    plt.savefig('simple_maze_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to simple_maze_comparison.png")
    plt.show()


def plot_training_curves(cnn_history, trm_history, n_epochs):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, n_epochs + 1)

    # Loss
    axes[0].plot(epochs, cnn_history['train_loss'], 'b-', label='CNN', linewidth=2)
    axes[0].plot(epochs, trm_history['train_loss'], 'r-', label='TRM', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [a*100 for a in cnn_history['val_acc']], 'b-', label='CNN', linewidth=2)
    axes[1].plot(epochs, [a*100 for a in trm_history['val_acc']], 'r-', label='TRM', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy (%)')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simple_training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved to simple_training_curves.png")
    plt.show()


# ----------------------------
# Main
# ----------------------------

def main():
    print("=" * 60)
    print("SIMPLE MAZE COMPARISON: CNN vs TRM")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Hyperparameters
    SIZE = 8
    BATCH_SIZE = 16
    N_EPOCHS = 20

    # Create datasets
    print("\n" + "=" * 60)
    print("Creating datasets...")
    train_dataset = SimpleMazeDataset(n_samples=300, size=SIZE)
    val_dataset = SimpleMazeDataset(n_samples=60, size=SIZE)
    test_dataset = SimpleMazeDataset(n_samples=30, size=SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize models
    print("\n" + "=" * 60)
    print("Initializing models...")

    cnn_model = CNNMazeSolver(size=SIZE).to(device)
    trm_model = SimpleTRM(size=SIZE, z_dim=32, x_dim=16, y_dim=8,
                         inner_steps=3, sup_steps=5).to(device)

    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    trm_params = sum(p.numel() for p in trm_model.parameters())

    print(f"\nCNN parameters: {cnn_params:,}")
    print(f"TRM parameters: {trm_params:,}")

    # Train models
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    cnn_opt = optim.Adam(cnn_model.parameters(), lr=1e-3)
    trm_opt = optim.Adam(trm_model.parameters(), lr=5e-4)

    cnn_history = train_model(cnn_model, train_loader, val_loader, cnn_opt,
                             device, n_epochs=N_EPOCHS, is_trm=False,
                             model_name="CNN Baseline")

    trm_history = train_model(trm_model, train_loader, val_loader, trm_opt,
                             device, n_epochs=N_EPOCHS, is_trm=True,
                             model_name="TRM")

    # Results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nCNN - Final val accuracy: {cnn_history['val_acc'][-1]*100:.2f}%")
    print(f"TRM - Final val accuracy: {trm_history['val_acc'][-1]*100:.2f}%")

    # Visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    visualize_results(cnn_model, trm_model, test_dataset, device)
    plot_training_curves(cnn_history, trm_history, N_EPOCHS)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
