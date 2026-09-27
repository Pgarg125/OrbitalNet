"""
quick_test.py
A fast (~1-2 minute) end-to-end sanity check of the ENTIRE OrbitalNet
pipeline: data generation -> PINN training -> baseline training ->
evaluation -> plotting. Uses a tiny dataset and few epochs, so it does NOT
produce a usable trained model -- it only verifies that your environment is
set up correctly and every stage of the pipeline runs without errors before
you commit to the full-scale run (which follows the synopsis's exact
settings: 500 trajectories, 200 timesteps, 10,000 epochs).

Run this FIRST, right after `pip install -r requirements.txt`:

    python quick_test.py

If it prints "ALL CHECKS PASSED" at the end, your environment is ready for
the full pipeline (see README.md "Running the full project").
"""

from __future__ import annotations

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 60)
    print("OrbitalNet quick end-to-end sanity check")
    print("=" * 60)

    t_start = time.time()

    # --- 1. Imports ---
    print("\n[1/6] Checking imports ...")
    import numpy as np
    import torch

    from src.data_generation import generate_dataset, train_val_test_split
    from src.evaluate import evaluate_basic, evaluate_long_horizon, evaluate_noise_robustness
    from src.model import OrbitalMLP
    from src.train import train_model
    from src.visualize import plot_loss_curve, plot_trajectory_matplotlib

    print(f"      torch {torch.__version__}, numpy {np.__version__} -- OK")

    # --- 2. Data generation ---
    print("\n[2/6] Generating a small test dataset (30 trajectories x 30 timesteps) ...")
    dataset = generate_dataset(n_trajectories=30, n_timesteps=30, seed=0)
    assert dataset["X"].shape == (900, 5)
    assert dataset["Y"].shape == (900, 4)
    splits = train_val_test_split(dataset, seed=0)
    print(f"      train={splits['train']['X'].shape[0]} val={splits['val']['X'].shape[0]} "
          f"test={splits['test']['X'].shape[0]} -- OK")

    # --- 3. PINN training (tiny) ---
    print("\n[3/6] Training PINN for 30 epochs (tiny, just to check autograd works) ...")
    pinn_model, pinn_stats, pinn_history = train_model(
        splits, use_physics=True, epochs=30, batch_size=32, log_every=10, verbose=True
    )
    assert isinstance(pinn_model, OrbitalMLP)
    print("      PINN training loop -- OK")

    # --- 4. Baseline training (tiny) ---
    print("\n[4/6] Training baseline for 30 epochs ...")
    baseline_model, baseline_stats, baseline_history = train_model(
        splits, use_physics=False, epochs=30, batch_size=32, log_every=10, verbose=True
    )
    print("      Baseline training loop -- OK")

    # --- 5. Evaluation ---
    print("\n[5/6] Running evaluation functions (MSE/MAE, noise robustness, long-horizon) ...")
    basic = evaluate_basic(pinn_model, splits, pinn_stats)
    noise = evaluate_noise_robustness(pinn_model, splits, pinn_stats)
    long_horizon = evaluate_long_horizon(pinn_model, splits, pinn_stats, dataset, n_test_traj=5)
    assert "mse" in basic and "mae" in basic
    assert len(noise) == 3
    assert "mse_mean" in long_horizon
    print(f"      basic={basic}")
    print("      Evaluation functions -- OK")

    # --- 6. Plotting ---
    print("\n[6/6] Testing plot generation ...")
    os.makedirs("results", exist_ok=True)
    from src.physics import sample_initial_condition, simulate_trajectory

    rng = np.random.default_rng(1)
    state0, period = sample_initial_condition(rng)
    t_eval = np.linspace(0, 1.15 * period, 50)
    true_traj = simulate_trajectory(state0, t_eval)
    X_query = np.zeros((50, 5))
    X_query[:, 0:4] = state0
    X_query[:, 4] = t_eval
    from src.evaluate import predict

    pred_traj = predict(pinn_model, X_query, pinn_stats)
    plot_trajectory_matplotlib(true_traj, pred_traj, save_path="results/_quicktest_orbit.png")
    plot_loss_curve(pinn_history, save_path="results/_quicktest_loss.png")
    assert os.path.exists("results/_quicktest_orbit.png")
    assert os.path.exists("results/_quicktest_loss.png")
    print("      Plots saved to results/_quicktest_*.png -- OK")

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"ALL CHECKS PASSED in {elapsed:.1f}s")
    print("Your environment is ready. See README.md 'Running the full project'")
    print("to generate the real dataset and train the full-scale models.")
    print("=" * 60)


if __name__ == "__main__":
    main()
