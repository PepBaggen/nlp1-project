import json

import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader

from part1 import smlp

# Our third architecture: the tutorial's SparseMLP with a second hidden layer


class SparseMLP2(nn.Module):
    """Two-hidden-layer MLP that accepts sparse input tensors."""

    def __init__(self, input_dim, hidden_dim, second_hidden_dim, num_classes, dropout=0.2):
        super().__init__()

        # First-layer weights stay dense; inputs arriving here are sparse.
        self.weight = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Everything after the first layer is dense.
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden_dim, num_classes),
            nn.LogSoftmax(dim=-1),
        )

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = torch.sparse.mm(x, self.weight) + self.bias
        x = self.dropout(self.activation(x))
        return self.output(x)


def _make_loader(pairs, encoder, batch_size, shuffle):
    """Encode (text, label) pairs and wrap them in a sparse dataloader"""
    vectors = csr_matrix([encoder(text) for text, label in pairs])
    labels = [int(label) for text, label in pairs]
    return DataLoader(
        smlp.SparseDataset(vectors, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=smlp.sparse_collate,
    )


def train_mlp2(training_set, validation_set, num_classes, encoder, save_to,
        hidden_dim=128, second_hidden_dim=64, dropout=0.1, batch_size=64,
        learning_rate=1e-3, weight_decay=1e-4, num_epochs=20, patience=None,
        device="cpu"):
    """
    Same contract as smlp.train_neural_network_model: returns the metrics per epoch,
    the best epoch (lowest validation loss) and that epoch's validation predictions.
    patience: stop after this many epochs without a better validation loss (None = never)
    """
    device = torch.device(device)
    train_loader = _make_loader(training_set, encoder, batch_size, shuffle=True)
    val_loader = _make_loader(validation_set, encoder, batch_size, shuffle=False)

    args = {
        "input_dim": len(encoder),
        "hidden_dim": hidden_dim,
        "second_hidden_dim": second_hidden_dim,
        "num_classes": num_classes,
        "dropout": dropout,
    }
    with open(f"{save_to}.json", "w", encoding="utf-8") as f:
        json.dump(args, f, indent=2)

    model = SparseMLP2(**args).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    metrics = []
    best_val_loss, best_epoch, predictions = float("inf"), None, None
    for epoch in range(num_epochs):
        train_metrics = smlp.train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, preds = smlp.evaluate(model, val_loader, criterion, device)
        metrics.append({"train_metrics": train_metrics, "val_metrics": val_metrics})

        if val_metrics["loss"] <= best_val_loss:
            best_val_loss, best_epoch, predictions = val_metrics["loss"], epoch, preds
            torch.save(model.state_dict(), f"{save_to}.pt")
        elif patience is not None and epoch - best_epoch >= patience:
            break

    return metrics, best_epoch, predictions
