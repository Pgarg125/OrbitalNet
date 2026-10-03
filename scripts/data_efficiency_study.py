"""
scripts/data_efficiency_study.py

Synopsis objective (section 4): "Data Efficiency Study demonstrating that
physics constraints let the PINN reach high accuracy with 20-60% less
training data than a plain neural network."

What this does: trains the PINN and the baseline at several different
TRAINING-set sizes (fractions of the full training split, subsampled by
WHOLE TRAJECTORY so no orbit is split across the kept/discarded portion),
then evaluates every one of those models on the SAME untouched, full-size
test set. Plotting test MSE against the training-data fraction for both
models shows how much each one's accuracy degrades as less data is
available -- directly the comparison the synopsis describes.

Design note on epoch count -- EQUAL-COMPUTE SCALING: a fixed epoch count at
every fraction is a hidden confound, because fewer training trajectories
means fewer batches per epoch, so "same epochs" silently means FEWER total
gradient updates at smaller fractions (in testing, 1000 epochs meant 68,000
total steps at 100% data but only 13,000 steps at 20% data -- a >5x gap).
That gap, not the data-efficiency effect itself, can easily dominate the
result. To remove it, --epochs sets the epoch count ONLY for the LARGEST
fraction in the list; every other fraction's epoch count is then scaled up
so it gets approximately the SAME TOTAL NUMBER OF GRADIENT STEPS as that
reference fraction (fewer samples per epoch -> more epochs to compensate).
This keeps total compute comparable across fractions, isolating the effect
of data quantity itself. The computed epoch count for each fraction is
printed before training starts.

This script saves its results-so-far to disk after EVERY fraction finishes
(not just at the very end), and automatically skips any fraction it finds
already saved from a previous run -- so if the terminal closes, the power
goes out, or anything else interrupts a run partway through, simply running
the exact same command again picks up where it left off instead of starting
all 8 training runs over from scratch. Use --force to ignore any existing
saved results and redo every fraction regardless.

Usage:
    python scripts/data_efficiency_study.py
    python scripts/data_efficiency_study.py --fractions 1.0 0.6 0.4 0.2 --epochs 1000
    python scripts/data_efficiency_study.py --force   (ignore saved progress, redo everything)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data_generation import load_dataset, train_val_test_split  # noqa: E402
from src.evaluate import evaluate_basic  # noqa: E402
from src.train import train_model  # noqa: E402
from src.visualize import plot_data_efficiency  # noqa: E402


def subsample_train_by_trajectory(train_split: dict, fraction: float, seed: int = 0):
    """
    Keeps a random `fraction` of the WHOLE TRAJECTORIES in train_split (never
    a partial trajectory), so each reduced training set is still made of
    complete, physically valid orbits -- just fewer of them.
    """
    traj_ids = np.unique(train_split["traj_id"])
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(len(traj_ids) * fraction)))
    keep_ids = rng.choice(traj_ids, size=n_keep, replace=False)
    mask = np.isin(train_split["traj_id"], keep_ids)
    subsampled = {
        "X": train_split["X"][mask],
        "Y": train_split["Y"][mask],
        "traj_id": train_split["traj_id"][mask],
    }
    return subsampled, n_keep, len(traj_ids)


def main():
    parser = argparse.ArgumentParser(description="OrbitalNet data-efficiency study (PINN vs baseline).")
    parser.add_argument("--data", type=str, default="data/orbitalnet_dataset.npz")
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[1.0, 0.6, 0.4, 0.2],
        help="Fractions of the training trajectories to use (default matches the synopsis's 20-60%% range).",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Per-run epoch count (see module docstring).")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument(
        "--device", type=str, default=None,
        help="cuda or cpu. Leave unset to auto-detect.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore any existing saved results and redo every fraction from scratch.",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found at {args.data}. Run scripts/generate_data.py first.")

    dataset = load_dataset(args.data)
    full_splits = train_val_test_split(dataset, seed=args.seed)
    n_total_traj = len(np.unique(full_splits["train"]["traj_id"]))

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "data_efficiency_summary.json")
    plot_path = os.path.join(args.out_dir, "data_efficiency_curve.png")

    # --- Equal-compute epoch scaling ---
    # Reference = the LARGEST fraction; args.epochs applies there. Every other
    # fraction's epoch count is scaled so steps_per_epoch * epochs is
    # approximately constant across all fractions (same total gradient steps).
    reference_fraction = max(args.fractions)
    ref_sub, _, _ = subsample_train_by_trajectory(full_splits["train"], reference_fraction, seed=args.seed)
    ref_batch_size = min(args.batch_size, max(1, len(ref_sub["X"]) // 2))
    ref_steps_per_epoch = max(1, len(ref_sub["X"]) // ref_batch_size)
    target_total_steps = ref_steps_per_epoch * args.epochs
    print(f"Equal-compute target: {target_total_steps} total gradient steps per model "
          f"(= {args.epochs} epochs x {ref_steps_per_epoch} steps/epoch at the reference "
          f"fraction {reference_fraction:.2f}).")

    results = {}  # fraction (str) -> {"PINN": {...}, "Baseline": {...}, "n_trajectories": int}
    if not args.force and os.path.exists(summary_path):
        with open(summary_path) as f:
            results = json.load(f)
        if results:
            print(f"Found existing results for fraction(s) {', '.join(results.keys())} in {summary_path} "
                  f"-- these will be SKIPPED (not retrained). Use --force to redo everything.")

    def save_progress():
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        if len(results) >= 2:  # a 1-point plot isn't meaningful; wait for at least 2
            fractions_sorted = sorted(results.keys(), key=float)
            pinn_mse = [results[k]["PINN"]["mse"] for k in fractions_sorted]
            baseline_mse = [results[k]["Baseline"]["mse"] for k in fractions_sorted]
            plot_data_efficiency([float(k) for k in fractions_sorted], pinn_mse, baseline_mse, save_path=plot_path)

    t_start = time.time()

    for fraction in args.fractions:
        key = f"{fraction:.2f}"
        if key in results:
            print(f"\n=== Fraction {fraction:.2f} -- already done, skipping ===")
            continue

        train_sub, n_keep, n_total = subsample_train_by_trajectory(full_splits["train"], fraction, seed=args.seed)
        batch_size = min(args.batch_size, max(1, len(train_sub["X"]) // 2))  # never exceed the subsampled set
        steps_per_epoch = max(1, len(train_sub["X"]) // batch_size)
        epochs_for_fraction = max(1, round(target_total_steps / steps_per_epoch))
        actual_total_steps = epochs_for_fraction * steps_per_epoch
        splits = {"train": train_sub, "val": full_splits["val"], "test": full_splits["test"]}

        print(f"\n=== Fraction {fraction:.2f}  ({n_keep}/{n_total} training trajectories, "
              f"{len(train_sub['X'])} samples, batch_size={batch_size}) ===")
        print(f"    equal-compute: {epochs_for_fraction} epochs x {steps_per_epoch} steps/epoch "
              f"= {actual_total_steps} total steps (target was {target_total_steps})")

        fraction_results = {"n_trajectories": n_keep, "n_total_trajectories": n_total, "epochs_used": epochs_for_fraction}

        for name, use_physics in [("PINN", True), ("Baseline", False)]:
            print(f"-- training {name} --")
            model, stats, _ = train_model(
                splits, use_physics=use_physics, lam=args.lam, epochs=epochs_for_fraction,
                batch_size=batch_size, log_every=args.log_every, seed=args.seed,
                device=device, verbose=True,
            )
            basic = evaluate_basic(model, splits, stats, device=device)
            fraction_results[name] = basic
            print(f"{name} @ fraction {fraction:.2f}: Test MSE={basic['mse']:.6f} MAE={basic['mae']:.6f}")

        results[key] = fraction_results
        save_progress()
        print(f"-- progress saved to {summary_path} ({len(results)}/{len(args.fractions)} fractions done) --")

    elapsed = time.time() - t_start
    print(f"\nAll fractions complete in {elapsed/60:.1f} minutes.")
    save_progress()
    print(f"Saved {summary_path} and {plot_path}")


if __name__ == "__main__":
    main()