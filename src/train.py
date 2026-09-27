"""
train.py
Training loop shared by both the PINN and the baseline MLP, so the two are
trained identically apart from the loss function -- matching the synopsis's
"identical architecture used for both the PINN and the baseline model, to
keep the comparison fair."

PINN:      use_physics=True  -> L_total = L_data + lambda * L_physics
Baseline:  use_physics=False -> L_total = L_data only

Defaults (epochs=10000, batch_size=256, lr via Adam, lambda=0.1) follow the
synopsis's stated methodology section 5.3.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import data_loss, total_pinn_loss
from .model import OrbitalMLP


def normalize_X(X: np.ndarray, stats: dict | None = None):
    """
    Standardise the INPUT features (X = [x0, y0, vx0, vy0, t]) to zero mean,
    unit std. Pass `stats` computed from the training set when normalising
    val/test data, to avoid any leakage from val/test statistics.
    """
    if stats is None:
        stats = {
            "X_mean": X.mean(axis=0),
            "X_std": X.std(axis=0) + 1e-8,
        }
    Xn = (X - stats["X_mean"]) / stats["X_std"]
    return Xn, stats


def train_model(
    splits: dict,
    use_physics: bool = True,
    lam: float = 0.1,
    epochs: int = 10000,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    log_every: int = 200,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Trains an OrbitalMLP on the given train/val splits.

    Returns:
        model:   trained OrbitalMLP (BEST validation-loss checkpoint, not
                 necessarily the final epoch -- see below)
        stats:   dict of input normalisation stats (needed at inference time)
        history: dict of per-logged-epoch metrics (for the loss-convergence plot)
    """
    torch.manual_seed(seed)

    X_train, Y_train = splits["train"]["X"], splits["train"]["Y"]
    X_val, Y_val = splits["val"]["X"], splits["val"]["Y"]

    Xn_train, stats = normalize_X(X_train)
    Xn_val, _ = normalize_X(X_val, stats)

    X_train_t = torch.tensor(Xn_train, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(Xn_val, dtype=torch.float32, device=device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32, device=device)

    t_std = torch.tensor(float(stats["X_std"][4]), dtype=torch.float32, device=device)

    dataset = TensorDataset(X_train_t, Y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = OrbitalMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "data_loss": [], "physics_loss": []}

    start = time.time()
    last_loss, last_l_data, last_l_phys = 0.0, 0.0, 0.0

    # Best-checkpoint tracking: long PINN runs can have unstable epochs where
    # val_loss briefly spikes (the physics and data terms fighting each other
    # mid-training) before recovering. Rather than blindly keeping whatever
    # the LAST epoch happens to look like -- which could be a spike -- we
    # track the best validation loss seen so far and restore those weights
    # at the end. Standard "best checkpoint" practice; applies identically to
    # the PINN and the baseline since both call this same function.
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            if use_physics:
                loss, l_data, l_phys = total_pinn_loss(model, xb, yb, xb.clone(), t_std, lam=lam)
            else:
                pred = model(xb)
                loss = data_loss(pred, yb)
                l_data, l_phys = loss.item(), 0.0
            loss.backward()
            optimizer.step()
            last_loss, last_l_data, last_l_phys = loss.item(), l_data, l_phys

        if epoch % log_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = data_loss(val_pred, Y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            history["epoch"].append(epoch)
            history["train_loss"].append(last_loss)
            history["val_loss"].append(val_loss)
            history["data_loss"].append(last_l_data)
            history["physics_loss"].append(last_l_phys)

            if verbose:
                elapsed = time.time() - start
                tag = "PINN" if use_physics else "Baseline"
                print(
                    f"[{tag}] epoch {epoch:5d} | train_loss {last_loss:.6f} "
                    f"| val_loss {val_loss:.6f} | data {last_l_data:.6f} "
                    f"| physics {last_l_phys:.6f} | {elapsed:.1f}s"
                )

    if best_state is not None:
        model.load_state_dict(best_state)
        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val_loss
        if verbose:
            tag = "PINN" if use_physics else "Baseline"
            print(
                f"[{tag}] Restored BEST checkpoint: epoch {best_epoch} "
                f"(val_loss {best_val_loss:.6f}) -- this is what gets saved, "
                f"not necessarily the final epoch."
            )

    return model, stats, history