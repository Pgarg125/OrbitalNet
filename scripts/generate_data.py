"""
Generates the OrbitalNet dataset and saves it to data/orbitalnet_dataset.npz

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --n_trajectories 50 --n_timesteps 50   (quick test)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation import generate_dataset, save_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate the OrbitalNet dataset via SciPy solve_ivp.")
    parser.add_argument("--n_trajectories", type=int, default=500, help="Number of orbital trajectories (synopsis default: 500)")
    parser.add_argument("--n_timesteps", type=int, default=200, help="Time steps per trajectory (synopsis default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/orbitalnet_dataset.npz")
    args = parser.parse_args()

    n_samples = args.n_trajectories * args.n_timesteps
    print(f"Generating {args.n_trajectories} trajectories x {args.n_timesteps} timesteps = {n_samples} samples ...")

    t0 = time.time()
    dataset = generate_dataset(n_trajectories=args.n_trajectories, n_timesteps=args.n_timesteps, seed=args.seed)
    save_dataset(args.out, dataset)
    elapsed = time.time() - t0

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"Done in {elapsed:.1f}s. Saved to {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
