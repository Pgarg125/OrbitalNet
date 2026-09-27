"""
model.py
The feedforward MLP architecture, used IDENTICALLY for both the PINN and the
baseline model so the comparative evaluation is fair -- exactly as specified
in the synopsis:

    5 inputs -> 4 x [64 neurons, Tanh] -> 4 outputs (x, y, vx, vy)

Only the training procedure differs between the PINN and the baseline (see
losses.py / train.py): the PINN adds a physics-residual term to the loss,
the baseline trains on data loss alone. The network itself is the same class.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OrbitalMLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 5,
        hidden_dim: int = 64,
        n_hidden_layers: int = 4,
        out_dim: int = 4,
    ):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
