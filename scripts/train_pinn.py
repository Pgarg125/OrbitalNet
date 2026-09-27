"""
Trains the PINN (physics-informed) model and saves it to models/pinn_model.pt

Usage:
    python scripts/train_pinn.py
    python scripts/train_pinn.py --epochs 200   (quick test)
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
from src.train import train_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train the OrbitalNet PINN.")
    parser.add_argument("--data", type=str, default="data/orbitalnet_dataset.npz")
    parser.add_argument("--epochs", type=int, default=10000, help="synopsis default: 10000")
    parser.add_argument("--batch_size", type=int, default=256, help="synopsis default: 256")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lam", type=float, default=0.1, help="physics loss weight lambda (synopsis default: 0.1)")
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="models")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found at {args.data}. Run scripts/generate_data.py first.")

    dataset = load_dataset(args.data)
    splits = train_val_test_split(dataset, seed=args.seed)
    print(f"Loaded dataset. Train/val/test sizes: "
          f"{splits['train']['X'].shape[0]}/{splits['val']['X'].shape[0]}/{splits['test']['X'].shape[0]}")

    model, stats, history = train_model(
        splits,
        use_physics=True,
        lam=args.lam,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every=args.log_every,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "pinn_model.pt"))
    np.savez(os.path.join(args.out_dir, "pinn_stats.npz"), **stats)
    with open(os.path.join(args.out_dir, "pinn_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"PINN training complete. Model + stats + history saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
