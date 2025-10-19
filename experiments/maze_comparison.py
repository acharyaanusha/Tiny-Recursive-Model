"""
Maze Solving Comparison: Transformer vs TRM

This script compares a baseline Transformer model with the Tiny Recursive Model (TRM)
on a maze-solving task. The maze is represented as a grid where the model must find
a path from start to goal while avoiding walls.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import time

# ----------------------------
# Maze Generation and Dataset
# ----------------------------

class MazeGenerator:
    """Generate random solvable mazes using DFS algorithm"""

    def __init__(self, height=10, width=10):
        self.height = height
        self.width = width

    def generate(self):
        """Generate a maze and return grid, start, goal"""
        # Initialize grid with walls (1 = wall, 0 = path)
        maze = np.ones((self.height, self.width), dtype=np.int32)

        # Start DFS from random position
        start_y, start_x = 1, 1
        maze[start_y, start_x] = 0

        # DFS to carve paths
        stack = [(start_y, start_x)]
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]

        while stack:
            y, x = stack[-1]
            neighbors = []

            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if (0 < ny < self.height - 1 and 0 < nx < self.width - 1 and
                    maze[ny, nx] == 1):
                    neighbors.append((ny, nx, dy // 2, dx // 2))

            if neighbors:
                ny, nx, dy, dx = neighbors[np.random.randint(len(neighbors))]
                maze[y + dy, x + dx] = 0  # carve path
                maze[ny, nx] = 0
                stack.append((ny, nx))
            else:
                stack.pop()

        # Set start and goal
        start = (1, 1)
        goal = (self.height - 2, self.width - 2)
        maze[goal[0], goal[1]] = 0  # ensure goal is reachable

        return maze, start, goal

    def solve_bfs(self, maze, start, goal):
        """Find shortest path using BFS"""
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            (y, x), path = queue.popleft()

            if (y, x) == goal:
                return path

            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.height and 0 <= nx < self.width and
                    maze[ny, nx] == 0 and (ny, nx) not in visited):
                    visited.add((ny, nx))
                    queue.append(((ny, nx), path + [(ny, nx)]))

        return None  # no solution


class MazeDataset(Dataset):
    """Dataset of mazes with solution paths"""

    def __init__(self, n_samples=1000, height=10, width=10):
        self.n_samples = n_samples
        self.height = height
        self.width = width
        self.data = []

        generator = MazeGenerator(height, width)

        print(f"Generating {n_samples} mazes...")
        for i in range(n_samples):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{n_samples}")

            # Generate solvable maze
            maze, start, goal = generator.generate()
            solution = generator.solve_bfs(maze, start, goal)

            # Create input: maze + start/goal markers
            # Channels: [walls, start, goal]
            maze_input = np.zeros((3, height, width), dtype=np.float32)
            maze_input[0] = maze  # walls
            maze_input[1, start[0], start[1]] = 1  # start
            maze_input[2, goal[0], goal[1]] = 1  # goal

            # Create target: path markers (1 = on path, 0 = not on path)
            target = np.zeros((height, width), dtype=np.int64)
            if solution:
                for y, x in solution:
                    target[y, x] = 1

            self.data.append((torch.from_numpy(maze_input),
                            torch.from_numpy(target),
                            solution))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]


# ----------------------------
# Baseline: Transformer Model
# ----------------------------

class TransformerMazeSolver(nn.Module):
    """Baseline model using standard Transformer encoder"""

    def __init__(self, height=10, width=10, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.height = height
        self.width = width
        self.d_model = d_model

        # Input projection: 3 channels (walls, start, goal) -> d_model
        self.input_proj = nn.Linear(3, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, height * width, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: binary classification per cell
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)  # 2 classes: on_path, not_on_path
        )

    def forward(self, x):
        """
        x: (batch, 3, height, width)
        returns: logits (batch, height, width, 2)
        """
        batch = x.size(0)

        # Reshape: (batch, 3, H, W) -> (batch, H*W, 3)
        x = x.view(batch, 3, -1).transpose(1, 2)

        # Project and add position encoding
        x = self.input_proj(x)  # (batch, H*W, d_model)
        x = x + self.pos_encoding

        # Transformer encoding
        x = self.transformer(x)  # (batch, H*W, d_model)

        # Output projection
        logits = self.output_head(x)  # (batch, H*W, 2)

        # Reshape back to grid
        logits = logits.view(batch, self.height, self.width, 2)

        return logits


# ----------------------------
# TRM Model for Maze Solving
# ----------------------------

class TinyNetwork(nn.Module):
    """Tiny recursive network for TRM"""

    def __init__(self, z_dim=64, x_dim=32, y_dim=16, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + x_dim + y_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z, x, y_embed):
        inp = torch.cat([z, x, y_embed], dim=-1)
        return self.net(inp)


class OutputHead(nn.Module):
    """Output head for TRM"""

    def __init__(self, z_dim=64, y_dim=16, n_classes=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, z, y_embed):
        inp = torch.cat([z, y_embed], dim=-1)
        return self.net(inp)


class TRMMazeSolver(nn.Module):
    """TRM model adapted for maze solving"""

    def __init__(self, height=10, width=10, x_dim=32, z_dim=64, y_dim=16,
                 inner_steps=4, sup_steps=8):
        super().__init__()
        self.height = height
        self.width = width
        self.seq_len = height * width
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_classes = 2  # binary: on_path or not

        # Input projection: 3 channels -> x_dim
        self.input_proj = nn.Linear(3, x_dim)

        # TRM components
        self.tiny = TinyNetwork(z_dim=z_dim, x_dim=x_dim, y_dim=y_dim)
        self.head = OutputHead(z_dim=z_dim, y_dim=y_dim, n_classes=2)
        self.y_token_embedding = nn.Embedding(2, y_dim)
        self.z0 = nn.Parameter(torch.randn(1, 1, z_dim) * 0.1)

        self.inner_steps = inner_steps
        self.sup_steps = sup_steps

    def forward(self, x, y_init_logits=None):
        """
        x: (batch, 3, height, width)
        returns: list of logits for each supervision step
        """
        batch = x.size(0)

        # Reshape: (batch, 3, H, W) -> (batch, H*W, 3)
        x = x.view(batch, 3, -1).transpose(1, 2)
        x = self.input_proj(x)  # (batch, H*W, x_dim)

        seq_len = x.size(1)

        # Initialize latent z
        z = self.z0.expand(batch, seq_len, self.z_dim).clone()

        # Initialize y logits
        if y_init_logits is None:
            y_logits = torch.zeros(batch, seq_len, self.n_classes, device=x.device)
        else:
            y_logits = y_init_logits

        supervised_logits = []

        # Recursive refinement
        for s in range(self.sup_steps):
            # Inner recursion steps
            for t in range(self.inner_steps):
                with torch.no_grad():
                    y_prob = torch.softmax(y_logits, dim=-1)
                y_embed = torch.matmul(y_prob, self.y_token_embedding.weight)

                z_new = self.tiny(z, x, y_embed)
                z = z_new if t == self.inner_steps - 1 else z_new.detach()

            # Outer update: produce new predictions
            y_prob = torch.softmax(y_logits, dim=-1)
            y_embed = torch.matmul(y_prob, self.y_token_embedding.weight)
            y_logits = self.head(z, y_embed)

            supervised_logits.append(y_logits)

        # Reshape logits back to grid format
        output_logits = []
        for logits in supervised_logits:
            logits_grid = logits.view(batch, self.height, self.width, self.n_classes)
            output_logits.append(logits_grid)

        return output_logits


# ----------------------------
# Training and Evaluation
# ----------------------------

def train_epoch(model, dataloader, optimizer, criterion, device, is_trm=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y_true in dataloader:
        x = x.to(device)
        y_true = y_true.to(device)

        optimizer.zero_grad()

        if is_trm:
            # TRM: multiple supervision steps
            sup_logits = model(x)
            loss = 0.0
            for i, logits in enumerate(sup_logits):
                weight = 1.0 + 0.1 * i
                loss_i = criterion(logits.permute(0, 3, 1, 2), y_true)
                loss = loss + weight * loss_i
            loss = loss / len(sup_logits)

            # Accuracy from final step
            final_logits = sup_logits[-1]
            preds = final_logits.argmax(dim=-1)
        else:
            # Transformer: single forward pass
            logits = model(x)
            loss = criterion(logits.permute(0, 3, 1, 2), y_true)
            preds = logits.argmax(dim=-1)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (preds == y_true).sum().item()
        total += y_true.numel()

    return total_loss / len(dataloader.dataset), correct / total


def evaluate(model, dataloader, device, is_trm=False):
    """Evaluate model"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y_true in dataloader:
            x = x.to(device)
            y_true = y_true.to(device)

            if is_trm:
                sup_logits = model(x)
                logits = sup_logits[-1]
            else:
                logits = model(x)

            preds = logits.argmax(dim=-1)
            correct += (preds == y_true).sum().item()
            total += y_true.numel()

    return correct / total


def visualize_comparison(maze_input, ground_truth, transformer_pred, trm_pred, save_path=None):
    """Visualize maze and predictions side by side"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Extract maze components
    walls = maze_input[0].cpu().numpy()
    start = maze_input[1].cpu().numpy()
    goal = maze_input[2].cpu().numpy()

    # Create display maze
    display_maze = np.ones((*walls.shape, 3))
    display_maze[walls == 0] = [1, 1, 1]  # white = path
    display_maze[walls == 1] = [0, 0, 0]  # black = wall
    display_maze[start == 1] = [0, 1, 0]  # green = start
    display_maze[goal == 1] = [1, 0, 0]   # red = goal

    # Plot maze
    axes[0].imshow(display_maze)
    axes[0].set_title('Maze\n(Green=Start, Red=Goal)', fontsize=10)
    axes[0].axis('off')

    # Plot ground truth
    gt_display = walls.copy()
    gt_display[ground_truth.cpu().numpy() == 1] = 0.5
    axes[1].imshow(gt_display, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth Path', fontsize=10)
    axes[1].axis('off')

    # Plot transformer prediction
    trans_display = walls.copy()
    trans_display[transformer_pred.cpu().numpy() == 1] = 0.5
    axes[2].imshow(trans_display, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Transformer Prediction', fontsize=10)
    axes[2].axis('off')

    # Plot TRM prediction
    trm_display = walls.copy()
    trm_display[trm_pred.cpu().numpy() == 1] = 0.5
    axes[3].imshow(trm_display, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('TRM Prediction', fontsize=10)
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


# ----------------------------
# Main Comparison
# ----------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Hyperparameters
    HEIGHT, WIDTH = 10, 10
    BATCH_SIZE = 16
    N_EPOCHS = 15

    # Create datasets
    print("=" * 60)
    print("CREATING DATASETS")
    print("=" * 60)
    train_dataset = MazeDataset(n_samples=500, height=HEIGHT, width=WIDTH)
    val_dataset = MazeDataset(n_samples=100, height=HEIGHT, width=WIDTH)
    test_dataset = MazeDataset(n_samples=50, height=HEIGHT, width=WIDTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize models
    print("\n" + "=" * 60)
    print("INITIALIZING MODELS")
    print("=" * 60)

    transformer_model = TransformerMazeSolver(
        height=HEIGHT, width=WIDTH, d_model=128, nhead=4, num_layers=3
    ).to(device)

    trm_model = TRMMazeSolver(
        height=HEIGHT, width=WIDTH, x_dim=32, z_dim=64, y_dim=16,
        inner_steps=4, sup_steps=6
    ).to(device)

    print(f"\nTransformer parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")
    print(f"TRM parameters: {sum(p.numel() for p in trm_model.parameters()):,}")

    # Optimizers
    transformer_opt = optim.Adam(transformer_model.parameters(), lr=1e-3, weight_decay=1e-5)
    trm_opt = optim.Adam(trm_model.parameters(), lr=5e-4, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()

    # Training
    print("\n" + "=" * 60)
    print("TRAINING COMPARISON")
    print("=" * 60)

    transformer_history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    trm_history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, N_EPOCHS + 1):
        print(f"\n{'Epoch ' + str(epoch):-^60}")

        # Train Transformer
        start_time = time.time()
        trans_loss, trans_train_acc = train_epoch(
            transformer_model, train_loader, transformer_opt, criterion, device, is_trm=False
        )
        trans_time = time.time() - start_time
        trans_val_acc = evaluate(transformer_model, val_loader, device, is_trm=False)

        transformer_history['train_loss'].append(trans_loss)
        transformer_history['train_acc'].append(trans_train_acc)
        transformer_history['val_acc'].append(trans_val_acc)

        # Train TRM
        start_time = time.time()
        trm_loss, trm_train_acc = train_epoch(
            trm_model, train_loader, trm_opt, criterion, device, is_trm=True
        )
        trm_time = time.time() - start_time
        trm_val_acc = evaluate(trm_model, val_loader, device, is_trm=True)

        trm_history['train_loss'].append(trm_loss)
        trm_history['train_acc'].append(trm_train_acc)
        trm_history['val_acc'].append(trm_val_acc)

        # Print comparison
        print(f"\nTransformer: loss={trans_loss:.4f} | train_acc={trans_train_acc*100:.2f}% | "
              f"val_acc={trans_val_acc*100:.2f}% | time={trans_time:.2f}s")
        print(f"TRM:         loss={trm_loss:.4f} | train_acc={trm_train_acc*100:.2f}% | "
              f"val_acc={trm_val_acc*100:.2f}% | time={trm_time:.2f}s")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    trans_test_acc = evaluate(transformer_model, test_loader, device, is_trm=False)
    trm_test_acc = evaluate(trm_model, test_loader, device, is_trm=True)

    print(f"\nTransformer Test Accuracy: {trans_test_acc*100:.2f}%")
    print(f"TRM Test Accuracy:         {trm_test_acc*100:.2f}%")

    # Visualize sample predictions
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    transformer_model.eval()
    trm_model.eval()

    with torch.no_grad():
        # Get a few test samples
        for i, (x, y_true) in enumerate(test_loader):
            if i >= 3:  # Show 3 examples
                break

            x = x.to(device)
            y_true = y_true.to(device)

            # Get predictions
            trans_logits = transformer_model(x)
            trans_pred = trans_logits.argmax(dim=-1)

            trm_logits = trm_model(x)
            trm_pred = trm_logits[-1].argmax(dim=-1)

            # Visualize first sample in batch
            visualize_comparison(
                x[0], y_true[0], trans_pred[0], trm_pred[0],
                save_path=f'maze_comparison_{i+1}.png'
            )

    # Plot training curves
    print("\n" + "=" * 60)
    print("TRAINING CURVES")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, N_EPOCHS + 1)

    # Loss plot
    axes[0].plot(epochs, transformer_history['train_loss'], 'b-', label='Transformer', linewidth=2)
    axes[0].plot(epochs, trm_history['train_loss'], 'r-', label='TRM', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, [acc * 100 for acc in transformer_history['val_acc']],
                'b-', label='Transformer', linewidth=2)
    axes[1].plot(epochs, [acc * 100 for acc in trm_history['val_acc']],
                'r-', label='TRM', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy (%)')
    axes[1].set_title('Validation Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved to training_curves.png")
    plt.show()

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
