"""For each learned SAE feature, report what it correlates with.

We want to know: does feature 7 correspond to S₂? Does feature 23 fire when
there's a defect? The way to answer that is to compute a known quantity
(S₂, dominant angle Ω, MT density) for each snapshot in the training set,
then check which SAE feature most strongly tracks each quantity.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from math import pi

from .model import SAE


def s2_from_snapshot(snap: np.ndarray) -> float:
    """Nematic order parameter from a length-weighted angle histogram.

    The histogram covers [0, π) at N_BINS resolution. S₂ = sqrt((Σ p_i cos 2θ_i)² + (Σ p_i sin 2θ_i)²)
    where p_i is the histogram bin probability and θ_i is the bin center.
    """
    n_bins = snap.shape[0]
    bin_centers = (np.arange(n_bins) + 0.5) * (pi / n_bins)
    c = np.sum(snap * np.cos(2 * bin_centers))
    s = np.sum(snap * np.sin(2 * bin_centers))
    return float(np.sqrt(c * c + s * s))


def omega_from_snapshot(snap: np.ndarray) -> float:
    """Dominant angle Ω, in [0, π/2]. arg max of length-weighted distribution."""
    n_bins = snap.shape[0]
    bin_centers = (np.arange(n_bins) + 0.5) * (pi / n_bins)
    c = np.sum(snap * np.cos(2 * bin_centers))
    s = np.sum(snap * np.sin(2 * bin_centers))
    return float(np.arctan2(s, c) / 2 % pi)


def entropy_from_snapshot(snap: np.ndarray) -> float:
    """Shannon entropy of the histogram (high = isotropic, low = aligned)."""
    p = snap + 1e-12
    return float(-np.sum(p * np.log(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sae_path', type=Path,
                    help='*.npz file saved by sae.train')
    ap.add_argument('--top_k', type=int, default=5)
    args = ap.parse_args()

    sae = SAE.load(args.sae_path)
    data_path = str(args.sae_path).replace('.npz', '_data.npz')
    X = np.load(data_path)['X']
    print(f"loaded {X.shape[0]} snapshots from {data_path}")

    # Compute known scalar metrics per snapshot
    s2 = np.array([s2_from_snapshot(x) for x in X])
    omega = np.array([omega_from_snapshot(x) for x in X])
    entropy = np.array([entropy_from_snapshot(x) for x in X])

    # Compute SAE activations
    H = sae.encode(X)   # shape (N, n_features)
    print(f"feature activations shape: {H.shape}")

    print("\nFeature activation statistics:")
    print(f"  mean active fraction: {(H > 0).mean():.3f}")
    print(f"  features that ever activate: "
          f"{(H.sum(axis=0) > 0).sum()}/{H.shape[1]}")

    # For each known metric, find the SAE features most correlated with it
    for metric_name, metric in [('S₂', s2), ('Ω', omega), ('entropy', entropy)]:
        corrs = []
        for f in range(H.shape[1]):
            h = H[:, f]
            if h.std() < 1e-8:
                corrs.append(0.0)
                continue
            corrs.append(float(np.corrcoef(h, metric)[0, 1]))
        corrs = np.array(corrs)
        order = np.argsort(-np.abs(corrs))[:args.top_k]
        print(f"\n  Top {args.top_k} features for {metric_name}:")
        for f in order:
            print(f"    feature {f:3d}: corr = {corrs[f]:+.3f}  "
                  f"(activated in {100*(H[:,f]>0).mean():.1f}% of snapshots)")


if __name__ == '__main__':
    main()
