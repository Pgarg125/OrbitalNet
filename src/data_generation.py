"""
data_generation.py
Builds the OrbitalNet dataset by simulating two-body trajectories with
SciPy's solve_ivp, per the project synopsis:
    - 500+ orbital trajectories, circular and elliptical, varied initial
      conditions
    - 200 time steps per trajectory -> 100,000 training samples
    - Input features per sample: [x0, y0, vx0, vy0, t]
    - Target per sample: [x(t), y(t), vx(t), vy(t)]
"""

from __future__ import annotations

import os

import numpy as np

from .physics import GM, sample_initial_condition, simulate_trajectory

N_TRAJECTORIES_DEFAULT = 500
N_TIMESTEPS_DEFAULT = 200
# Sample slightly beyond one full period so the network sees a tiny bit of
# "wrap-around" behaviour, not just a single monotonic arc.
T_MAX_MULTIPLIER = 1.15


def generate_dataset(
    n_trajectories: int = N_TRAJECTORIES_DEFAULT,
    n_timesteps: int = N_TIMESTEPS_DEFAULT,
    seed: int = 42,
) -> dict:
    """
    Generates the full OrbitalNet dataset.

    Returns a dict with:
        X:           (n_trajectories * n_timesteps, 5) -> [x0, y0, vx0, vy0, t]
        Y:           (n_trajectories * n_timesteps, 4) -> [x, y, vx, vy] (ground truth)
        traj_id:     (n_trajectories * n_timesteps,)   -> which trajectory each row belongs to
        init_states: (n_trajectories, 4)               -> [x0, y0, vx0, vy0] per trajectory
        periods:     (n_trajectories,)                 -> orbital period per trajectory
        GM:          scalar, the fixed canonical gravitational parameter
    """
    rng = np.random.default_rng(seed)

    n_samples = n_trajectories * n_timesteps
    all_X = np.zeros((n_samples, 5), dtype=np.float64)
    all_Y = np.zeros((n_samples, 4), dtype=np.float64)
    all_traj_id = np.zeros(n_samples, dtype=np.int64)

    init_states = np.zeros((n_trajectories, 4), dtype=np.float64)
    periods = np.zeros(n_trajectories, dtype=np.float64)

    idx = 0
    for i in range(n_trajectories):
        state0, period = sample_initial_condition(rng)
        t_max = T_MAX_MULTIPLIER * period
        t_eval = np.linspace(0.0, t_max, n_timesteps)

        traj = simulate_trajectory(state0, t_eval)  # (n_timesteps, 4)

        all_X[idx : idx + n_timesteps, 0:4] = state0
        all_X[idx : idx + n_timesteps, 4] = t_eval
        all_Y[idx : idx + n_timesteps, :] = traj
        all_traj_id[idx : idx + n_timesteps] = i

        init_states[i] = state0
        periods[i] = period

        idx += n_timesteps

    return {
        "X": all_X,
        "Y": all_Y,
        "traj_id": all_traj_id,
        "init_states": init_states,
        "periods": periods,
        "GM": np.array(GM, dtype=np.float64),
    }


def train_val_test_split(dataset: dict, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42) -> dict:
    """
    Splits the dataset by TRAJECTORY (not by individual time-sample), so an
    entire orbit's 200 time-points stay together in one split. This avoids
    leaking time-points of the same trajectory across train/val/test, which
    would make evaluation misleadingly easy.
    """
    n_traj = dataset["init_states"].shape[0]
    rng = np.random.default_rng(seed)
    traj_indices = rng.permutation(n_traj)

    n_test = int(n_traj * test_frac)
    n_val = int(n_traj * val_frac)

    test_traj = traj_indices[:n_test]
    val_traj = traj_indices[n_test : n_test + n_val]
    train_traj = traj_indices[n_test + n_val :]

    traj_id = dataset["traj_id"]

    splits = {}
    for name, traj_set in [("train", train_traj), ("val", val_traj), ("test", test_traj)]:
        mask = np.isin(traj_id, traj_set)
        splits[name] = {
            "X": dataset["X"][mask],
            "Y": dataset["Y"][mask],
            "traj_id": dataset["traj_id"][mask],
        }
    return splits


def save_dataset(path: str, dataset: dict) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(path, **dataset)


def load_dataset(path: str) -> dict:
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}
