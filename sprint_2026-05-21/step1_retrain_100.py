#!/usr/bin/env python3
"""Sprint Step 1 — retrain the collision MLP to label-100% accuracy.

Strategy:
  - 1M training samples
  - 50% uniform (a1, a2) ~ U(0, 2π)
  - 50% boundary-oversampled: incident_angle ~ U(TH_CRIT - 0.15, TH_CRIT + 0.15)
  - Slightly bigger model: 5 -> 64 -> 64 -> {4 logits + 2 sincos}
  - Stop when held-out classification == 100.0%

Outputs:
  models/learned_zipcat_v2.npz                  (model weights)
  sprint_2026-05-21/step1_metrics.json          (final accuracy, confusion, time)
"""
import sys, os, time, json
from pathlib import Path
import numpy as np
from math import pi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from collision.api import zip_cat_clean
from collision.decision import incident_angle, TH_CRIT

OUTCOME_TO_IDX = {'zipper+': 0, 'zipper-': 1, 'cross': 2, 'catas': 3}
IDX_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_IDX.items()}


def encode(a1, a2, r):
    return np.array([np.cos(a1), np.sin(a1), np.cos(a2), np.sin(a2), float(r)],
                    dtype=np.float32)


def generate_dataset(n_samples, seed=42, boundary_frac=0.5, boundary_band=0.15):
    """Half uniform, half oversampled near TH_CRIT decision boundary."""
    rng = np.random.default_rng(seed)
    n_boundary = int(n_samples * boundary_frac)
    n_uniform  = n_samples - n_boundary

    X = np.zeros((n_samples, 5), dtype=np.float32)
    y_out = np.zeros(n_samples, dtype=np.int64)
    y_ang = np.zeros((n_samples, 2), dtype=np.float32)
    pt = [0.5, 0.5]; pt_prev = [0.4, 0.4]

    print(f"  generating {n_uniform} uniform + {n_boundary} boundary samples...")
    t0 = time.time()

    # Uniform half
    for i in range(n_uniform):
        a1 = rng.uniform(0, 2*pi); a2 = rng.uniform(0, 2*pi)
        r = int(rng.integers(0, 2))
        new_angle, _, outcome, _, _ = zip_cat_clean(a1, a2, pt, pt_prev, r)
        X[i] = encode(a1, a2, r)
        y_out[i] = OUTCOME_TO_IDX[outcome]
        y_ang[i] = [np.cos(new_angle), np.sin(new_angle)]

    # Boundary-oversampled half: pick a1 uniform, then a2 so incident_angle ∈ [TH-band, TH+band]
    for i in range(n_uniform, n_samples):
        a1 = rng.uniform(0, 2*pi)
        target_incident = rng.uniform(max(0, TH_CRIT - boundary_band),
                                      min(pi/2, TH_CRIT + boundary_band))
        # a2 = a1 ± target_incident, plus optional +π flip (nematic equivalence)
        sign = 1 if rng.random() < 0.5 else -1
        flip = pi if rng.random() < 0.5 else 0
        a2 = (a1 + sign * target_incident + flip) % (2*pi)
        r = int(rng.integers(0, 2))
        new_angle, _, outcome, _, _ = zip_cat_clean(a1, a2, pt, pt_prev, r)
        X[i] = encode(a1, a2, r)
        y_out[i] = OUTCOME_TO_IDX[outcome]
        y_ang[i] = [np.cos(new_angle), np.sin(new_angle)]

    print(f"  dataset built in {time.time()-t0:.1f}s")
    # Class balance sanity
    counts = np.bincount(y_out, minlength=4)
    print(f"  class counts: zipper+={counts[0]} zipper-={counts[1]} cross={counts[2]} catas={counts[3]}")
    return X, y_out, y_ang


class MLP:
    """5 -> 64 -> 64 -> {4 logits + 2 sincos}.  ~5k params."""
    def __init__(self, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2/5), (5, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2/hidden), (hidden, hidden)).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.Wo = rng.normal(0, np.sqrt(2/hidden), (hidden, 4)).astype(np.float32)
        self.bo = np.zeros(4, dtype=np.float32)
        self.Wa = rng.normal(0, np.sqrt(2/hidden), (hidden, 2)).astype(np.float32)
        self.ba = np.zeros(2, dtype=np.float32)
        self._params = ['W1','b1','W2','b2','Wo','bo','Wa','ba']
        self._m = {k: np.zeros_like(getattr(self, k)) for k in self._params}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._params}
        self._t = 0

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0, z2)
        logits = h2 @ self.Wo + self.bo
        ang = h2 @ self.Wa + self.ba
        return logits, ang, (z1, h1, z2, h2)

    def predict_label(self, X):
        logits, _, _ = self.forward(X)
        return logits.argmax(axis=1)

    def step(self, X, y_out, y_ang, lr=3e-3, ang_w=0.3):
        B = X.shape[0]
        logits, ang, (z1, h1, z2, h2) = self.forward(X)
        # CE
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        oh = np.zeros_like(p); oh[np.arange(B), y_out] = 1.0
        g_logits = (p - oh) / B
        # MSE on angle
        g_ang = 2.0 * (ang - y_ang) / B * ang_w
        # heads
        gWo = h2.T @ g_logits; gbo = g_logits.sum(axis=0)
        gh2 = g_logits @ self.Wo.T
        gWa = h2.T @ g_ang;    gba = g_ang.sum(axis=0)
        gh2 += g_ang @ self.Wa.T
        gz2 = gh2 * (z2 > 0)
        gW2 = h1.T @ gz2; gb2 = gz2.sum(axis=0)
        gh1 = gz2 @ self.W2.T
        gz1 = gh1 * (z1 > 0)
        gW1 = X.T @ gz1;  gb1 = gz1.sum(axis=0)
        # Adam
        self._t += 1; b1_, b2_, eps = 0.9, 0.999, 1e-8
        grads = dict(W1=gW1, b1=gb1, W2=gW2, b2=gb2,
                     Wo=gWo, bo=gbo, Wa=gWa, ba=gba)
        for k, g in grads.items():
            self._m[k] = b1_*self._m[k] + (1-b1_)*g
            self._v[k] = b2_*self._v[k] + (1-b2_)*g*g
            mh = self._m[k] / (1 - b1_**self._t)
            vh = self._v[k] / (1 - b2_**self._t)
            setattr(self, k, getattr(self, k) - lr*mh/(np.sqrt(vh)+eps))
        loss_ce  = -np.log(p[np.arange(B), y_out] + 1e-9).mean()
        loss_ang = ((ang - y_ang)**2).mean()
        return loss_ce, loss_ang

    def save(self, path):
        np.savez(path, **{k: getattr(self, k) for k in self._params})


def main():
    out_dir = REPO / "sprint_2026-05-21"
    out_dir.mkdir(exist_ok=True)
    model_dir = REPO / "models"
    model_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("STEP 1 — Retrain MLP to label-100%")
    print("=" * 60)

    print("\n[1/3] Generating training set (1M samples)...")
    X_train, y_out_train, y_ang_train = generate_dataset(1_000_000, seed=42)

    print("\n[2/3] Generating held-out test set (100k uniform + 100k boundary)...")
    X_test, y_out_test, y_ang_test = generate_dataset(200_000, seed=999)

    print("\n[3/3] Training MLP (hidden=64)...")
    model = MLP(hidden=64, seed=0)
    n_epochs = 30
    batch_size = 4096
    n_train = X_train.shape[0]
    history = []
    best_acc = 0.0
    t_train_start = time.time()

    rng = np.random.default_rng(0)
    for epoch in range(n_epochs):
        # shuffle
        idx = rng.permutation(n_train)
        ce_total, ang_total, nb = 0.0, 0.0, 0
        for start in range(0, n_train, batch_size):
            sl = idx[start:start+batch_size]
            ce, ang = model.step(X_train[sl], y_out_train[sl], y_ang_train[sl])
            ce_total += ce; ang_total += ang; nb += 1

        # eval
        preds = []
        for start in range(0, X_test.shape[0], 8192):
            preds.append(model.predict_label(X_test[start:start+8192]))
        preds = np.concatenate(preds)
        acc = float((preds == y_out_test).mean())

        history.append(dict(epoch=epoch+1, ce=ce_total/nb, ang_mse=ang_total/nb, test_acc=acc))
        print(f"  epoch {epoch+1:2d}/{n_epochs}  CE={ce_total/nb:.4f}  "
              f"ang_MSE={ang_total/nb:.4f}  test_acc={acc*100:.3f}%")
        if acc > best_acc:
            best_acc = acc
            model.save(model_dir / "learned_zipcat_v2.npz")
        if acc >= 0.99999:  # effectively 100%
            print(f"  ✓ hit ~100% at epoch {epoch+1}")
            break

    t_train = time.time() - t_train_start

    # Final confusion + boundary-specific accuracy
    print("\n[final] Computing detailed metrics...")
    preds = []
    for start in range(0, X_test.shape[0], 8192):
        preds.append(model.predict_label(X_test[start:start+8192]))
    preds = np.concatenate(preds)
    final_acc = float((preds == y_out_test).mean())
    conf = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_out_test, preds):
        conf[t, p] += 1

    # Boundary-only accuracy on the boundary half of test (indices >= n_uniform)
    n_boundary_test = X_test.shape[0] // 2
    boundary_preds = preds[X_test.shape[0] - n_boundary_test:]
    boundary_true  = y_out_test[X_test.shape[0] - n_boundary_test:]
    boundary_acc = float((boundary_preds == boundary_true).mean())

    metrics = dict(
        final_test_acc = final_acc,
        final_test_acc_pct = f"{final_acc*100:.4f}%",
        boundary_acc = boundary_acc,
        boundary_acc_pct = f"{boundary_acc*100:.4f}%",
        confusion_matrix = conf.tolist(),
        confusion_labels = ['zipper+', 'zipper-', 'cross', 'catas'],
        n_train = int(n_train),
        n_test  = int(X_test.shape[0]),
        epochs_run = len(history),
        train_seconds = t_train,
        hidden_size = 64,
        n_params = int(sum(getattr(model, k).size for k in model._params)),
        history = history,
        model_path = "models/learned_zipcat_v2.npz",
    )
    with open(out_dir / "step1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  RESULT: test accuracy = {final_acc*100:.4f}%  "
          f"(boundary-only = {boundary_acc*100:.4f}%)")
    print(f"  model saved to {metrics['model_path']}  ({metrics['n_params']} params)")
    print(f"  metrics saved to sprint_2026-05-21/step1_metrics.json")
    print(f"  train time: {t_train:.1f}s")
    return final_acc


if __name__ == "__main__":
    main()
