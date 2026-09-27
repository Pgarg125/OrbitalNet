"""
evaluate.py
Comparative evaluation of PINN vs baseline, per the synopsis's evaluation
plan (section 5.4):
    - MSE / MAE on held-out test data
    - Noise robustness: Gaussian noise at sigma = 0.01, 0.05, 0.1
    - Long-horizon extrapolation accuracy: error beyond the training time domain

Noise-robustness convention
-----------------------------
Gaussian noise is added to the INITIAL-CONDITION inputs (x0, y0, vx0, vy0)
only -- simulating noisy real-world measurement of a spacecraft/satellite's
initial state -- while the query time t is left clean. Predictions from the
noisy input are compared against the TRUE (noiseless) trajectory, so the
reported MSE/MAE directly reflects how much initial-condition sensor noise
degrades trajectory accuracy. This choice is documented here and in the
README for the viva.
"""

from __future__ import annotations

import numpy as np
import torch

from .physics import simulate_trajectory


def predict(model: torch.nn.Module, X: np.ndarray, stats: dict, device: str = "cpu") -> np.ndarray:
    """Runs the model on raw (unnormalised) input X, returns raw (physical-unit) predictions."""
    model.eval()
    Xn = (X - stats["X_mean"]) / stats["X_std"]
    with torch.no_grad():
        pred = model(torch.tensor(Xn, dtype=torch.float32, device=device)).cpu().numpy()
    return pred


def mse_mae(pred: np.ndarray, target: np.ndarray):
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    return mse, mae


def evaluate_basic(model, splits: dict, stats: dict, split_name: str = "test", device: str = "cpu") -> dict:
    X = splits[split_name]["X"]
    Y = splits[split_name]["Y"]
    pred = predict(model, X, stats, device=device)
    mse, mae = mse_mae(pred, Y)
    return {"mse": mse, "mae": mae}


def evaluate_noise_robustness(
    model,
    splits: dict,
    stats: dict,
    noise_levels=(0.01, 0.05, 0.1),
    split_name: str = "test",
    device: str = "cpu",
    seed: int = 123,
) -> dict:
    """
    Adds Gaussian noise (sigma in canonical units) to the initial-condition
    inputs only, keeps t clean, and measures degradation vs. the TRUE
    (noiseless) trajectory.
    """
    rng = np.random.default_rng(seed)
    X = splits[split_name]["X"]
    Y = splits[split_name]["Y"]

    results = {}
    for sigma in noise_levels:
        X_noisy = X.copy()
        noise = rng.normal(0.0, sigma, size=X_noisy[:, 0:4].shape)
        X_noisy[:, 0:4] += noise
        pred = predict(model, X_noisy, stats, device=device)
        mse, mae = mse_mae(pred, Y)
        results[sigma] = {"mse": mse, "mae": mae}
    return results


def evaluate_long_horizon(
    model,
    splits: dict,
    stats: dict,
    dataset: dict,
    extension_factor: float = 2.0,
    n_points: int = 100,
    n_test_traj: int = 20,
    device: str = "cpu",
    seed: int = 7,
    t_max_multiplier: float = 1.15,
) -> dict:
    """
    For a sample of test trajectories, re-integrates the TRUE trajectory out
    to `extension_factor` x its original training-domain max time (i.e.
    beyond what the model was trained on), and compares against the model's
    prediction at those extrapolated times.
    """
    rng = np.random.default_rng(seed)
    test_traj_ids = np.unique(splits["test"]["traj_id"])
    if len(test_traj_ids) > n_test_traj:
        test_traj_ids = rng.choice(test_traj_ids, size=n_test_traj, replace=False)

    all_mse, all_mae = [], []
    for tid in test_traj_ids:
        state0 = dataset["init_states"][tid]
        period = dataset["periods"][tid]
        t_max_original = t_max_multiplier * period
        t_max_extended = extension_factor * t_max_original
        t_eval = np.linspace(t_max_original, t_max_extended, n_points)

        true_traj = simulate_trajectory(state0, t_eval)

        X_query = np.zeros((n_points, 5))
        X_query[:, 0:4] = state0
        X_query[:, 4] = t_eval

        pred = predict(model, X_query, stats, device=device)
        mse, mae = mse_mae(pred, true_traj)
        all_mse.append(mse)
        all_mae.append(mae)

    return {"mse_mean": float(np.mean(all_mse)), "mae_mean": float(np.mean(all_mae))}
