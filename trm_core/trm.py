# trm_example.py
# Requires: torch (tested on torch >=1.13)
# Usage: python trm_example.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random

# ----------------------------
# Simple synthetic dataset
# ----------------------------
class SimplePuzzleDataset(Dataset):
    """
    A toy dataset: input x is a vector; target y_true is a discrete label per position.
    We simulate a per-position classification problem (like grid cells in Sudoku / maze output).
    Replace this with real dataset loader for real tasks.
    """
    def __init__(self, n_samples=1000, seq_len=16, n_classes=10):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_classes = n_classes
        # random mapping: label = (sum(x_slice) * some factor) mod n_classes
        self.data = []
        for _ in range(n_samples):
            x = torch.randn(seq_len, 32)  # per-position input embedding (32-d)
            # create a target which is a deterministic function of x (toy)
            target = ( (x.sum(dim=1).round().long() + torch.arange(seq_len)) % n_classes )
            self.data.append((x, target))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]

# ----------------------------
# TRM components
# ----------------------------
class TinyNetwork(nn.Module):
    """
    The tiny recursive network f(z, x, y): a small 2-layer MLP that
    consumes (z, x, y_prev) and returns updated z.
    z_dim: latent dimension
    x_dim: per-position input embedding dimension
    y_dim: per-position soft/hard embedding dimension (we will use a small embedding)
    """
    def __init__(self, z_dim=64, x_dim=32, y_dim=16, hidden=128):
        super().__init__()
        self.in_dim = z_dim + x_dim + y_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z, x, y_embed):
        # z, x, y_embed shapes:
        # z: (batch, seq_len, z_dim)
        # x: (batch, seq_len, x_dim)
        # y_embed: (batch, seq_len, y_dim)
        inp = torch.cat([z, x, y_embed], dim=-1)
        return self.net(inp)  # new z (batch, seq_len, z_dim)

class OutputHead(nn.Module):
    """
    g(y_prev, z) -> logits for new y. This maps (z, optionally y_prev) to per-position logits.
    """
    def __init__(self, z_dim=64, y_dim=16, n_classes=10, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, z, y_embed):
        inp = torch.cat([z, y_embed], dim=-1)
        return self.net(inp)  # logits (batch, seq_len, n_classes)

class TRM(nn.Module):
    """
    Tiny Recursive Model implementation.
    - z is the latent: initialized learnable or zeros
    - y is current predicted logits/class; we represent y in two forms:
        - y_logits: unnormalized logits (to compute loss and to derive y_embed)
        - y_embed: small embedding of current discrete prediction or soft distribution
    """
    def __init__(self,
                 x_dim=32,
                 z_dim=64,
                 y_dim=16,
                 n_classes=10,
                 inner_steps=4,    # number of recursion steps per supervision step
                 sup_steps=8,      # number of supervised improvement steps (N_sup)
                 detach_inner_steps=True):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_classes = n_classes

        self.tiny = TinyNetwork(z_dim=z_dim, x_dim=x_dim, y_dim=y_dim)
        self.head = OutputHead(z_dim=z_dim, y_dim=y_dim, n_classes=n_classes)

        # embedding for y (to feed into tiny network); we will use an embedding layer for discrete labels
        self.y_token_embedding = nn.Embedding(n_classes, y_dim)

        # initial learnable latent z0 (shared across sequences, broadcasted)
        self.z0 = nn.Parameter(torch.randn(1, 1, z_dim) * 0.1)

        self.inner_steps = inner_steps
        self.sup_steps = sup_steps
        self.detach_inner_steps = detach_inner_steps

    def forward(self, x, y_init_logits=None):
        """
        x: (batch, seq_len, x_dim)
        y_init_logits: optionally initial logits (batch, seq_len, n_classes)
        returns:
          - list of logits at each supervised step (length sup_steps)
        """
        batch, seq_len, _ = x.shape

        # initialize z by expanding z0 to batch, seq_len
        z = self.z0.expand(batch, seq_len, self.z_dim).clone()

        # init y logits: if provided, use; else uniform zeros
        if y_init_logits is None:
            y_logits = torch.zeros(batch, seq_len, self.n_classes, device=x.device)
        else:
            y_logits = y_init_logits

        # collect supervised outputs
        supervised_logits = []

        for s in range(self.sup_steps):
            # inner recursion: refine z multiple times given current y and x
            for t in range(self.inner_steps):
                # get y embedding: use argmax->embedding (hard) or soft embedding via probabilities
                with torch.no_grad():
                    # using soft distribution to compute embedding better preserves gradient paths to head
                    y_prob = torch.softmax(y_logits, dim=-1)
                # get y embedding: (batch, seq_len, y_dim)
                y_embed = torch.matmul(y_prob, self.y_token_embedding.weight)  # soft embedding

                # update latent z via tiny network
                z_new = self.tiny(z, x, y_embed)

                # Optionally detach gradients from earlier inner steps to mimic 1-step gradient trick:
                if self.detach_inner_steps and t < (self.inner_steps - 1):
                    z = z_new.detach()
                else:
                    z = z_new

            # outer update: produce new logits y <- g(z, y_embed)
            # recompute y_embed (from current y_logits)
            y_prob = torch.softmax(y_logits, dim=-1)
            y_embed = torch.matmul(y_prob, self.y_token_embedding.weight)
            y_logits = self.head(z, y_embed)

            # store this supervised prediction (logits)
            supervised_logits.append(y_logits)

        # supervised_logits: list length sup_steps of tensors (batch, seq_len, n_classes)
        return supervised_logits

# ----------------------------
# Training loop (toy)
# ----------------------------
def train_one_epoch(model, dataloader, optim, device, criterion):
    model.train()
    total_loss = 0.0
    for x, y_true in dataloader:
        x = x.to(device)
        y_true = y_true.to(device)  # (batch, seq_len) long labels
        optim.zero_grad()
        # forward -> get logits for each supervised step
        sup_logits = model(x)  # list of tensors
        # apply loss on each supervised step (deep supervision)
        loss = 0.0
        # weight recent supervision higher (optional)
        for i, logits in enumerate(sup_logits):
            # reshape for CrossEntropyLoss: (batch*seq_len, n_classes)
            b, L, C = logits.shape
            loss_i = criterion(logits.view(-1, C), y_true.view(-1))
            # optionally weight later steps more strongly
            weight = 1.0 + 0.1 * i
            loss = loss + weight * loss_i
        loss = loss / len(sup_logits)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y_true in dataloader:
            x = x.to(device)
            y_true = y_true.to(device)
            sup_logits = model(x)
            # evaluate final supervised step
            final_logits = sup_logits[-1]
            preds = final_logits.argmax(dim=-1)
            correct += (preds == y_true).sum().item()
            total += y_true.numel()
    return correct / total

# ----------------------------
# Main: set up and run a short training
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # hyperparams (tune to match paper or task)
    x_dim = 32
    z_dim = 64
    y_dim = 16
    n_classes = 10
    inner_steps = 4
    sup_steps = 6

    model = TRM(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, n_classes=n_classes,
                inner_steps=inner_steps, sup_steps=sup_steps, detach_inner_steps=True)
    model.to(device)

    # dataset
    ds = SimplePuzzleDataset(n_samples=400, seq_len=16, n_classes=n_classes)
    train_loader = DataLoader(ds, batch_size=16, shuffle=True)
    val_ds = SimplePuzzleDataset(n_samples=100, seq_len=16, n_classes=n_classes)
    val_loader = DataLoader(val_ds, batch_size=16)

    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 6
    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} - train_loss: {tr_loss:.4f}  val_acc: {val_acc*100:.2f}%")

if __name__ == "__main__":
    main()
