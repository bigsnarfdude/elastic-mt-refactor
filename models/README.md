# Trained models

Three small NPZ files plus their training data — all trained on outputs of
the refactored simulator.

| File | Size | What | How to use |
|------|------|------|------------|
| `learned_zipcat.npz` | 7.5 KB | 1,446-param MLP that learns Tim's collision rule (99.2% accuracy, 2.5° median angle error) | `CollisionMLP.load('models/learned_zipcat.npz')` |
| `sae.npz` | 10.5 KB | 32-feature sparse autoencoder, sparsity_coef=0.003. 31/32 features alive. | `SAE.load('models/sae.npz')` |
| `sae_tight.npz` | 10.5 KB | Same arch, higher sparsity (0.02). Tighter feature selection. | `SAE.load('models/sae_tight.npz')` |
| `sae_data.npz` | 1.9 MB | 10,124 × 36 per-region snapshot histograms + condition labels — the training set for both SAEs | `np.load('models/sae_data.npz')` → `X, labels` |

## What's in each weight file

### `learned_zipcat.npz` — CollisionMLP
```
W1: (5, 32)     # input layer:  (cos_a1, sin_a1, cos_a2, sin_a2, r) → hidden
b1: (32,)
W2: (32, 32)    # hidden → hidden
b2: (32,)
Wo: (32, 4)     # outcome head: 4 logits (zipper+, zipper-, cross, catas)
bo: (4,)
Wa: (32, 2)    # angle head: (cos, sin) of post-collision angle
ba: (2,)
```

Trained on 50,000 random calls to `collision.api.zip_cat_clean` (seed=42).
40 epochs, batch size 256, Adam lr=3e-3.

### `sae.npz` / `sae_tight.npz` — SAE
```
W_e: (32, 36)              # encoder
b_e: (32,)
W_d: (36, 32)              # decoder
b_d: (36,)
sparsity_coef: scalar       # L1 weight (0.003 in sae.npz, 0.02 in sae_tight.npz)
```

Trained on the 10,124 per-region histograms in `sae_data.npz`.
Each feature is orientation-selective; preferred angles tile [12°, 158°].

### `sae_data.npz` — training set
```
X:      (10124, 36) float32  # length-weighted angle histograms (5° bins, [0,π))
labels: (10124,)    str      # one of 'baseline', 'no_zipper', 'always_catas'
```

Each row is one region × one time point. Pooled from baseline + no_zipper +
always_catas trajectories at seed 42, sim hours 0.1 to 1.0.

## Reproducing

The training scripts are reproducible from the data + seeds:

```bash
# Retrain the SAE
python -m sae.train --data models/sae_data.npz --out models/sae.npz \
    --n_features 32 --sparsity_coef 0.003 --epochs 200

# Retrain the MLP from scratch (regenerates data on the fly)
python -c "
import sys; sys.path.insert(0, '.')
from sae.learned_collision import generate_dataset, CollisionMLP
import numpy as np
X, y_o, y_a = generate_dataset(50_000, seed=42)
mlp = CollisionMLP(seed=0)
rng = np.random.default_rng(0)
for epoch in range(40):
    perm = rng.permutation(len(X))
    for i in range(0, len(X), 256):
        idx = perm[i:i+256]
        mlp.step(X[idx], y_o[idx], y_a[idx], lr=3e-3)
mlp.save('models/learned_zipcat.npz')
"
```

Both are seeded — re-running gives bit-identical weights.

## Provenance

- Code: `refactor/split-zip-cat` branch
- Simulator data source: 1-hour runs with seed=42 (the only seed used so far;
  multi-seed runs are the natural next step)
- All training run on a MacBook in ~30 seconds total wall time per model
