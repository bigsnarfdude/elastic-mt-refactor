#!/usr/bin/env python3
"""Retrain SAE + MLP on multi-seed data.

Pulls per-region snapshots from all 30 jobs (3 conditions × 10 seeds), pools
them, and retrains both models. Compares to the n=1 versions to check
whether feature interpretations hold up at n>1.

Usage:
    python -m sae.retrain_multiseed <data_dir>

<data_dir> = directory containing condition_seed*/orderp_seed*_*.pickle files.
"""
import argparse
import sys
import os
from pathlib import Path
import numpy as np

# allow run from repo root
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from sae.snapshot import load_per_region, N_BINS
from sae.model import SAE
from sae.learned_collision import (
    generate_dataset, CollisionMLP, encode_input,
    OUTCOME_TO_IDX, IDX_TO_OUTCOME
)
from sae.probe import s2_from_snapshot, omega_from_snapshot


def collect_per_region(data_dir: Path, min_mt_len: float = 0.5):
    """Walk data_dir, find every orderp_*.pickle, accumulate per-region snapshots
    + labels (condition name) + metadata (seed, time, region_idx, mt_len)."""
    all_snaps, all_labels, all_meta = [], [], []
    seeds_per_cond = {}
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir() or '_seed' not in sub.name:
            continue
        cond, sd = sub.name.rsplit('_seed', 1)
        seed = int(sd)
        seeds_per_cond.setdefault(cond, set()).add(seed)
        for pkl in sub.glob(f'orderp_seed{seed}_*.pickle'):
            try:
                snaps, meta = load_per_region(pkl, min_mt_len=min_mt_len)
                if snaps.shape[0] == 0:
                    continue
                all_snaps.append(snaps)
                all_labels.extend([cond] * snaps.shape[0])
                for m in meta:
                    m['seed'] = seed
                    m['condition'] = cond
                all_meta.extend(meta)
            except Exception as e:
                print(f"  skip {pkl.name}: {e}", flush=True)
    X = np.concatenate(all_snaps, axis=0).astype(np.float32) if all_snaps else np.zeros((0, N_BINS), dtype=np.float32)
    labels = np.array(all_labels)
    print(f"Collected {X.shape[0]} per-region snapshots from {sum(len(s) for s in seeds_per_cond.values())} runs:", flush=True)
    for c, s in seeds_per_cond.items():
        print(f"  {c}: {len(s)} seeds", flush=True)
    return X, labels, all_meta


def train_sae(X, n_features=32, sparsity_coef=0.02, epochs=100,
              batch_size=128, lr=3e-3, seed=0):
    sae = SAE(input_dim=X.shape[1], n_features=n_features,
              sparsity_coef=sparsity_coef, seed=seed)
    rng = np.random.default_rng(seed)
    print(f"Training {n_features}-feature SAE on {X.shape[0]} snapshots for {epochs} epochs...",
          flush=True)
    for epoch in range(epochs):
        perm = rng.permutation(X.shape[0])
        for i in range(0, X.shape[0], batch_size):
            sae.step(X[perm[i:i+batch_size]], lr=lr)
        if epoch % 20 == 0 or epoch == epochs - 1:
            h = sae.encode(X)
            recon = ((sae.decode(h) - X) ** 2).mean()
            print(f"  epoch {epoch:3d}: recon={recon:.5f}  active_frac={(h > 0).mean():.3f}  "
                  f"dead={(h.sum(axis=0)==0).sum()}/{n_features}", flush=True)
    return sae


def probe_orientation_tuning(sae, X, labels):
    """For each feature, find its preferred angle and correlation with S2/Ω."""
    H = sae.encode(X)
    pref_bins = sae.W_d.argmax(axis=0)
    pref_angles_deg = (pref_bins + 0.5) * 180 / N_BINS

    s2 = np.array([s2_from_snapshot(x) for x in X])
    omega = np.array([omega_from_snapshot(x) for x in X])
    n_features = H.shape[1]
    alive = (H.sum(axis=0) > 0)
    corrs_s2 = np.zeros(n_features)
    corrs_omega = np.zeros(n_features)
    for f in range(n_features):
        if H[:, f].std() < 1e-8:
            continue
        corrs_s2[f] = np.corrcoef(H[:, f], s2)[0, 1]
        corrs_omega[f] = np.corrcoef(H[:, f], np.cos(2 * omega))[0, 1]
    return dict(
        H=H, pref_angles_deg=pref_angles_deg,
        corrs_s2=corrs_s2, corrs_omega=corrs_omega,
        alive=alive,
        usage=(H > 0).mean(axis=0),
    )


def retrain_mlp(n_samples=50_000, seed=42, epochs=40, batch_size=256, lr=3e-3):
    """Retrain MLP — same recipe as the n=1 version, just confirms reproducibility."""
    print(f"\nGenerating {n_samples} MLP training samples (seed={seed})...", flush=True)
    X, y_out, y_ang = generate_dataset(n_samples, seed=seed)
    Xt, y_out_t, y_ang_t = generate_dataset(5000, seed=seed + 1000)
    mlp = CollisionMLP(seed=0)
    rng = np.random.default_rng(0)
    print(f"Training MLP for {epochs} epochs...", flush=True)
    for epoch in range(epochs):
        perm = rng.permutation(X.shape[0])
        for i in range(0, X.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            mlp.step(X[idx], y_out[idx], y_ang[idx], lr=lr)
        if epoch % 10 == 0 or epoch == epochs - 1:
            pred_out, _ = mlp.predict(Xt)
            acc = (pred_out == y_out_t).mean()
            print(f"  epoch {epoch:3d}  test_acc={100*acc:.1f}%", flush=True)
    return mlp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir', type=Path)
    ap.add_argument('--out_dir', type=Path, default=Path('models'))
    ap.add_argument('--n_features', type=int, default=32)
    ap.add_argument('--sparsity_coef', type=float, default=0.02)
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. SAE on multi-seed per-region data
    X, labels, meta = collect_per_region(args.data_dir)
    sae = train_sae(X, n_features=args.n_features,
                    sparsity_coef=args.sparsity_coef, epochs=args.epochs)
    sae.save(args.out_dir / 'sae_multiseed.npz')
    np.savez(args.out_dir / 'sae_data_multiseed.npz', X=X, labels=labels)
    print(f"\nSaved SAE → {args.out_dir / 'sae_multiseed.npz'}", flush=True)

    # 2. Orientation probe
    probe = probe_orientation_tuning(sae, X, labels)
    print(f"\nSAE feature stats:", flush=True)
    print(f"  alive: {probe['alive'].sum()}/{probe['H'].shape[1]}", flush=True)
    if probe['alive'].any():
        print(f"  max |corr(feature, cos 2Ω)|: {np.max(np.abs(probe['corrs_omega'][probe['alive']])):.3f}", flush=True)
        print(f"  max |corr(feature, S₂)|:      {np.max(np.abs(probe['corrs_s2'][probe['alive']])):.3f}", flush=True)

    # 3. MLP — independent retraining for reproducibility
    mlp = retrain_mlp()
    mlp.save(args.out_dir / 'learned_zipcat_multiseed.npz')
    print(f"\nSaved MLP → {args.out_dir / 'learned_zipcat_multiseed.npz'}", flush=True)


if __name__ == '__main__':
    main()
