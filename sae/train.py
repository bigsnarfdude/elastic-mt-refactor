"""Train an SAE on MT trajectory snapshots.

Usage:
    python -m sae.train <data_dir> <output_path> [--n_features 64] [--epochs 200]

<data_dir> contains one or more ``orderp_seed*_*.pickle`` files (anywhere
nested). They get loaded into a single (N, n_bins) snapshot matrix.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

from .snapshot import load_trajectory, N_BINS
from .model import SAE


def collect_snapshots(data_dir: Path) -> np.ndarray:
    """Find all orderp_*.pickle files under data_dir and stack their snapshots."""
    pkls = list(data_dir.rglob('orderp_*.pickle'))
    if not pkls:
        raise FileNotFoundError(f"no orderp_*.pickle files under {data_dir}")
    print(f"found {len(pkls)} pickle files", flush=True)
    all_snaps = []
    for p in pkls:
        try:
            snaps = load_trajectory(p)
            if snaps.shape[0] > 0:
                all_snaps.append(snaps)
        except Exception as e:
            print(f"  skip {p.name}: {e}", flush=True)
    X = np.concatenate(all_snaps, axis=0)
    print(f"total snapshots: {X.shape[0]}, dim={X.shape[1]}", flush=True)
    return X


def train(X: np.ndarray, n_features: int, epochs: int,
          batch_size: int = 64, lr: float = 1e-3,
          sparsity_coef: float = 0.04, seed: int = 42) -> SAE:
    """Train an SAE on the snapshot matrix. Returns trained model."""
    sae = SAE(input_dim=X.shape[1], n_features=n_features,
              sparsity_coef=sparsity_coef, seed=seed)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    print(f"training {n_features}-feature SAE on {n} snapshots for {epochs} epochs",
          flush=True)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        losses, sparses, actives = [], [], []
        for i in range(0, n, batch_size):
            batch = X[perm[i:i+batch_size]]
            metrics = sae.step(batch, lr=lr)
            losses.append(metrics['recon'])
            sparses.append(metrics['sparse'])
            actives.append(metrics['active_frac'])
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  recon={np.mean(losses):.5f}  "
                  f"sparse={np.mean(sparses):.5f}  active={np.mean(actives):.3f}",
                  flush=True)
    return sae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir', type=Path)
    ap.add_argument('output_path', type=Path)
    ap.add_argument('--n_features', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--sparsity_coef', type=float, default=0.04)
    args = ap.parse_args()

    X = collect_snapshots(args.data_dir)
    sae = train(X, args.n_features, args.epochs,
                sparsity_coef=args.sparsity_coef)
    sae.save(args.output_path)
    # Also save the training matrix for later probing
    np.savez(str(args.output_path).replace('.npz', '_data.npz'),
             X=X)
    print(f"saved SAE to {args.output_path}")


if __name__ == '__main__':
    main()
