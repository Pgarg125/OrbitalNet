"""
Runs the full comparative evaluation (MSE/MAE, noise robustness, long-horizon
extrapolation) for PINN vs baseline, per the synopsis's evaluation plan.
Saves a JSON summary and example orbit + loss-curve plots to results/.

Usage:
    python scripts/run_evaluation.py

Requires both scripts/train_pinn.py and scripts/train_baseline.py to have
been run first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data_generation import load_dataset, train_val_test_split  # noqa: E402
from src.evaluate import (  # noqa: E402
    evaluate_basic,
    evaluate_long_horizon,
    evaluate_noise_robustness,
    predict,
)
from src.model import OrbitalMLP  # noqa: E402
from src.physics import simulate_trajectory  # noqa: E402
from src.visualize import plot_loss_curve, plot_trajectory_matplotlib  # noqa: E402


def load_trained(model_path: str, stats_path: str):
    if not (os.path.exists(model_path) and os.path.exists(stats_path)):
        raise FileNotFoundError(
            f"Missing {model_path} or {stats_path}. "
            f"Run scripts/train_pinn.py and scripts/train_baseline.py first."
        )
    model = OrbitalMLP()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    stats_npz = np.load(stats_path)
    stats = {k: stats_npz[k] for k in stats_npz.files}
    return model, stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate PINN vs baseline for OrbitalNet.")
    parser.add_argument("--data", type=str, default="data/orbitalnet_dataset.npz")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = load_dataset(args.data)
    splits = train_val_test_split(dataset, seed=args.seed)

    pinn_model, pinn_stats = load_trained(
        os.path.join(args.models_dir, "pinn_model.pt"),
        os.path.join(args.models_dir, "pinn_stats.npz"),
    )
    baseline_model, baseline_stats = load_trained(
        os.path.join(args.models_dir, "baseline_model.pt"),
        os.path.join(args.models_dir, "baseline_stats.npz"),
    )

    summary = {}
    for name, model, stats in [("PINN", pinn_model, pinn_stats), ("Baseline", baseline_model, baseline_stats)]:
        basic = evaluate_basic(model, splits, stats)
        noise = evaluate_noise_robustness(model, splits, stats)
        long_horizon = evaluate_long_horizon(model, splits, stats, dataset)

        summary[name] = {
            "test_mse": basic["mse"],
            "test_mae": basic["mae"],
            "noise_robustness": {str(k): v for k, v in noise.items()},
            "long_horizon": long_horizon,
        }

        print(f"\n=== {name} ===")
        print(f"Test MSE: {basic['mse']:.6f} | Test MAE: {basic['mae']:.6f}")
        for sigma, res in noise.items():
            print(f"  Noise sigma={sigma}: MSE={res['mse']:.6f} MAE={res['mae']:.6f}")
        print(f"  Long-horizon (2x training horizon): MSE={long_horizon['mse_mean']:.6f} MAE={long_horizon['mae_mean']:.6f}")

    with open(os.path.join(args.out_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- Plot one example test trajectory: true vs PINN vs baseline ---
    test_traj_ids = np.unique(splits["test"]["traj_id"])
    tid = test_traj_ids[0]
    state0 = dataset["init_states"][tid]
    period = dataset["periods"][tid]
    t_eval = np.linspace(0, 1.15 * period, 200)
    true_traj = simulate_trajectory(state0, t_eval)

    X_query = np.zeros((200, 5))
    X_query[:, 0:4] = state0
    X_query[:, 4] = t_eval

    pinn_pred = predict(pinn_model, X_query, pinn_stats)
    baseline_pred = predict(baseline_model, X_query, baseline_stats)

    plot_trajectory_matplotlib(
        true_traj, pinn_pred, title="PINN: predicted vs true orbit",
        save_path=os.path.join(args.out_dir, "pinn_vs_true_orbit.png"),
    )
    plot_trajectory_matplotlib(
        true_traj, baseline_pred, title="Baseline MLP: predicted vs true orbit",
        save_path=os.path.join(args.out_dir, "baseline_vs_true_orbit.png"),
    )

    # --- Plot loss convergence curves, if history files are present ---
    for name in ["pinn", "baseline"]:
        hist_path = os.path.join(args.models_dir, f"{name}_history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
            plot_loss_curve(
                history, title=f"{name.upper()} training loss convergence",
                save_path=os.path.join(args.out_dir, f"{name}_loss_curve.png"),
            )

    print(f"\nSaved evaluation summary + plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
