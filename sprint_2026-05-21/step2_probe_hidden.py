#!/usr/bin/env python3
"""Sprint Step 2 — probe the trained MLP's hidden layer.

Q1: Is incident_angle linearly decodable from h2 (last hidden layer)?
    -> Linear regression h2 -> incident_angle.  Report R².

Q2: What features does a small SAE find on h2 (across 200k collisions)?
    -> Train 16-feature SAE with L1 sparsity.
    -> Report each feature's tuning curve vs incident_angle, vs outcome.
    -> Flag any feature that fires sharply near TH_CRIT.

Outputs:
  sprint_2026-05-21/step2_metrics.json
  sprint_2026-05-21/step2_probe.png       (h2 PCA + linear-probe scatter)
  sprint_2026-05-21/step2_sae_features.png (16 tuning curves)
"""
import sys, json
from pathlib import Path
import numpy as np
from math import pi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from collision.api import zip_cat_clean
from collision.decision import incident_angle, TH_CRIT
sys.path.insert(0, str(REPO / "sprint_2026-05-21"))
from step1_retrain_100 import MLP, encode, OUTCOME_TO_IDX, generate_dataset


def load_model():
    """Reconstruct MLP from saved v2 weights."""
    npz = np.load(REPO / "models" / "learned_zipcat_v2.npz")
    m = MLP(hidden=npz['W1'].shape[1], seed=0)
    for k in m._params:
        setattr(m, k, npz[k])
    return m


def hidden_h2(model, X, batch=8192):
    """Forward and return h2 (last hidden) — the layer we probe."""
    out = []
    for s in range(0, X.shape[0], batch):
        _, _, (_, _, _, h2) = model.forward(X[s:s+batch])
        out.append(h2)
    return np.concatenate(out, axis=0)


def linear_probe(H, y, l2=1e-3):
    """Closed-form ridge regression.  Returns (w, b, R²)."""
    Hc = np.concatenate([H, np.ones((H.shape[0], 1), dtype=H.dtype)], axis=1)
    A = Hc.T @ Hc + l2 * np.eye(Hc.shape[1], dtype=H.dtype)
    A[-1, -1] = 0  # don't regularize bias
    w = np.linalg.solve(A, Hc.T @ y)
    yhat = Hc @ w
    ss_res = ((y - yhat)**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1 - ss_res/ss_tot
    return w[:-1], w[-1], float(r2)


class SAE:
    """Anthropic-style SAE: pre-center data, decoder rows = unit norm,
    encoder tied init, dead-feature resample every 5 epochs.

    forward:  f = ReLU(W_enc @ (x - b_dec) + b_enc)
              recon = W_dec @ f + b_dec
    """
    def __init__(self, d_in, d_feat=16, sparsity=0.001, seed=0):
        rng = np.random.default_rng(seed)
        # decoder rows: random unit vectors
        W_dec = rng.normal(0, 1.0, (d_feat, d_in)).astype(np.float32)
        W_dec /= (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-9)
        self.W_dec = W_dec
        # encoder = decoder transpose at init (tied init)
        self.W_enc = W_dec.T.copy()
        self.b_enc = np.zeros(d_feat, dtype=np.float32)
        self.b_dec = np.zeros(d_in, dtype=np.float32)  # will be set to data mean
        self.sparsity = sparsity
        # Track activation frequency for dead-feature resampling
        self._activation_count = np.zeros(d_feat, dtype=np.int64)
        self._examples_seen = 0

    def encode(self, X):
        return np.maximum(0, (X - self.b_dec) @ self.W_enc + self.b_enc)

    def forward(self, X):
        f = self.encode(X)
        recon = f @ self.W_dec + self.b_dec
        return recon, f

    def _resample_dead(self, X, threshold_frac=1e-4):
        """Reinit dead features (those that activated < threshold_frac of the time)."""
        if self._examples_seen == 0:
            return 0
        freq = self._activation_count / self._examples_seen
        dead_mask = freq < threshold_frac
        n_dead = int(dead_mask.sum())
        if n_dead == 0:
            return 0
        # Resample dead features from random data examples
        rng = np.random.default_rng()
        idx = rng.choice(X.shape[0], n_dead, replace=False)
        # Decoder rows: random data examples (centered) normalized
        new_dec = X[idx] - self.b_dec
        new_dec /= (np.linalg.norm(new_dec, axis=1, keepdims=True) + 1e-9)
        self.W_dec[dead_mask] = new_dec
        # Encoder rows: same as decoder (tied)
        self.W_enc[:, dead_mask] = new_dec.T
        self.b_enc[dead_mask] = 0.0
        # Reset counters for resampled features
        self._activation_count[dead_mask] = 0
        return n_dead

    def train(self, X, n_epochs=80, batch=4096, lr=3e-3, resample_every=5):
        # Initialize b_dec to data mean (so features can subtract the constant)
        self.b_dec = X.mean(axis=0).astype(np.float32)
        rng = np.random.default_rng(0)
        n = X.shape[0]
        history = []
        for epoch in range(n_epochs):
            idx = rng.permutation(n)
            r_total, s_total, nb = 0.0, 0.0, 0
            self._activation_count[:] = 0
            self._examples_seen = 0
            for start in range(0, n, batch):
                sl = idx[start:start+batch]
                xb = X[sl]
                xc = xb - self.b_dec
                pre = xc @ self.W_enc + self.b_enc
                f = np.maximum(0, pre)
                recon = f @ self.W_dec + self.b_dec
                err = recon - xb
                B = xb.shape[0]
                # grads
                gW_dec = f.T @ err / B
                gb_dec = err.mean(axis=0)
                gf = err @ self.W_dec.T + self.sparsity * np.sign(f)
                gf_pre = gf * (f > 0)
                gW_enc = xc.T @ gf_pre / B
                gb_enc = gf_pre.mean(axis=0)
                # SGD
                self.W_enc -= lr * gW_enc
                self.b_enc -= lr * gb_enc
                self.W_dec -= lr * gW_dec
                self.b_dec -= lr * gb_dec
                # Renormalize decoder rows to unit norm
                norms = np.linalg.norm(self.W_dec, axis=1, keepdims=True) + 1e-9
                self.W_dec /= norms
                # Track activations for resampling
                self._activation_count += (f > 0).sum(axis=0)
                self._examples_seen += B
                r_total += (err**2).mean()
                s_total += np.abs(f).mean()
                nb += 1
            n_live = int((self._activation_count > 0).sum())
            history.append(dict(epoch=epoch+1, recon=r_total/nb,
                                sparsity=s_total/nb, live=n_live))
            if (epoch+1) % 10 == 0:
                print(f"  SAE epoch {epoch+1:2d}/{n_epochs}  "
                      f"recon={r_total/nb:.4f}  L1={s_total/nb:.4f}  "
                      f"live={n_live}/{self.W_dec.shape[0]}")
            # Resample dead features every N epochs (skip last 10 epochs)
            if (epoch+1) % resample_every == 0 and epoch < n_epochs - 10:
                n_resampled = self._resample_dead(X[idx[:50000]])
                if n_resampled > 0:
                    print(f"    ↳ resampled {n_resampled} dead features")
        return history


def main():
    out_dir = REPO / "sprint_2026-05-21"
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("STEP 2 — Probe MLP hidden layer")
    print("=" * 60)

    print("\n[1/4] Loading trained MLP...")
    model = load_model()
    print(f"  hidden size = {model.W1.shape[1]}, params = "
          f"{sum(getattr(model, k).size for k in model._params)}")

    print("\n[2/4] Generating 200k probe examples (uniform random)...")
    X, y_out, _ = generate_dataset(200_000, seed=2026, boundary_frac=0.0)
    # Recompute incident_angle for each example (the supervised target for the probe)
    a1 = np.arctan2(X[:, 1], X[:, 0])
    a2 = np.arctan2(X[:, 3], X[:, 2])
    inc = np.array([incident_angle(float(x1), float(x2)) for x1, x2 in zip(a1, a2)],
                   dtype=np.float32)
    is_zip_regime = (inc <= TH_CRIT).astype(np.float32)

    print("\n[3/4] Computing h2 activations + linear probes...")
    H2 = hidden_h2(model, X)
    print(f"  H2 shape: {H2.shape}, mean nonzero per row: {(H2 > 0).sum(axis=1).mean():.1f}")

    _, _, r2_incident = linear_probe(H2, inc)
    _, _, r2_iszip    = linear_probe(H2, is_zip_regime)
    print(f"  linear probe  h2 -> incident_angle    R² = {r2_incident:.4f}")
    print(f"  linear probe  h2 -> is_zip_regime     R² = {r2_iszip:.4f}")

    # Per-unit correlation with incident_angle
    h2_mean = H2.mean(axis=0)
    h2_std  = H2.std(axis=0) + 1e-9
    inc_z   = (inc - inc.mean()) / (inc.std() + 1e-9)
    h2_z    = (H2 - h2_mean) / h2_std
    per_unit_corr = (h2_z * inc_z[:, None]).mean(axis=0)
    print(f"  per-unit |corr(h2_i, incident_angle)| top-3: "
          f"{np.sort(np.abs(per_unit_corr))[-3:][::-1].round(3).tolist()}")

    print("\n[4/4] Training 16-feature SAE on h2 (Anthropic-style + dead-feature resample)...")
    sae = SAE(d_in=H2.shape[1], d_feat=16, sparsity=0.001, seed=0)
    sae_history = sae.train(H2, n_epochs=80, batch=4096, lr=3e-3, resample_every=5)
    # Live-feature count (any nonzero activation across the 200k probe set)
    feats_check = sae.encode(H2)
    n_live = int((feats_check.max(axis=0) > 1e-6).sum())
    n_significant = int(((feats_check > 0).mean(axis=0) > 0.01).sum())  # fires on >1% of examples
    print(f"  live features: {n_live}/16  (significant: {n_significant}/16)")

    # Tuning curves: for each SAE feature, mean activation in 20 incident-angle bins
    features = sae.encode(H2)  # shape (N, 16)
    n_bins = 20
    bins = np.linspace(0, pi/2, n_bins + 1)
    bin_idx = np.digitize(inc, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    tuning = np.zeros((16, n_bins), dtype=np.float32)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            tuning[:, b] = features[mask].mean(axis=0)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    th_crit_bin = float(TH_CRIT)

    # Feature -> outcome correlation: for each feature, mean activation per outcome class
    feat_by_outcome = np.zeros((16, 4), dtype=np.float32)
    for cls in range(4):
        m = y_out == cls
        if m.any():
            feat_by_outcome[:, cls] = features[m].mean(axis=0)

    metrics = dict(
        r2_h2_to_incident_angle = r2_incident,
        r2_h2_to_is_zip_regime  = r2_iszip,
        per_unit_corr_top3_abs  = sorted(np.abs(per_unit_corr).tolist(), reverse=True)[:3],
        sae_n_features = 16,
        sae_n_live_features = n_live,
        sae_sparsity_coef = 0.001,
        sae_final_recon = sae_history[-1]['recon'],
        sae_final_L1    = sae_history[-1]['sparsity'],
        tuning_bin_centers = bin_centers.tolist(),
        tuning_curves      = tuning.tolist(),  # 16 x 20
        feat_by_outcome    = feat_by_outcome.tolist(),  # 16 x 4
        outcome_labels     = ['zipper+', 'zipper-', 'cross', 'catas'],
        th_crit_radians    = th_crit_bin,
    )
    with open(out_dir / "step2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Plot 1: tuning curves
        fig, axes = plt.subplots(4, 4, figsize=(14, 10), sharex=True)
        for fi, ax in enumerate(axes.flat):
            ax.plot(bin_centers * 180/pi, tuning[fi], lw=2, color='#2c5aa0')
            ax.axvline(th_crit_bin * 180/pi, color='#c0392b', ls='--', lw=1, alpha=0.7)
            ax.set_title(f"Feature {fi}", fontsize=9)
            ax.tick_params(labelsize=7)
        fig.text(0.5, 0.02, "incident_angle (degrees) — red line = TH_CRIT (40°)",
                 ha='center', fontsize=11)
        fig.text(0.02, 0.5, "mean feature activation", va='center',
                 rotation='vertical', fontsize=11)
        fig.suptitle(f"SAE-on-h2 tuning curves  (probe R²={r2_incident:.3f} "
                     f"on incident_angle, {r2_iszip:.3f} on is_zip_regime)",
                     fontsize=12)
        plt.tight_layout(rect=[0.04, 0.03, 1, 0.96])
        plt.savefig(out_dir / "step2_sae_features.png", dpi=130)
        plt.close()
        print(f"  ✓ wrote {out_dir / 'step2_sae_features.png'}")

        # Plot 2: feature x outcome heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(feat_by_outcome, aspect='auto', cmap='viridis')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['zipper+', 'zipper-', 'cross', 'catas'])
        ax.set_yticks(range(16))
        ax.set_yticklabels([f"F{i}" for i in range(16)])
        ax.set_xlabel("outcome class")
        ax.set_title("SAE feature × outcome (mean activation)")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(out_dir / "step2_feat_by_outcome.png", dpi=130)
        plt.close()
        print(f"  ✓ wrote {out_dir / 'step2_feat_by_outcome.png'}")
    except ImportError:
        print("  (matplotlib not available — skipping plots)")

    print(f"\n  RESULT:")
    print(f"    h2 -> incident_angle    R² = {r2_incident:.4f}")
    print(f"    h2 -> is_zip_regime     R² = {r2_iszip:.4f}")
    print(f"    metrics → sprint_2026-05-21/step2_metrics.json")


if __name__ == "__main__":
    main()
