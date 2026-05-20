"""Learn the collision rule as a neural network.

Demonstrates: we can replace Tim's hand-coded collision logic with a learned
function, then run the full simulator with the learned rule. This is the
"neural surrogate of zip_cat" experiment.

Pipeline:
  1. Generate N random (angle1, angle2, r) inputs
  2. Call the (refactored) zip_cat_clean to get ground-truth outputs
  3. Train a small MLP to predict (outcome, new_angle)
  4. Wrap the trained MLP to match the simulator's expected return signature
  5. Plug it in via collision.decision.decide_outcome and run the sim

The trained MLP is tiny (a few KB) and pure numpy.
"""
import numpy as np
from math import pi


OUTCOME_TO_IDX = {'zipper+': 0, 'zipper-': 1, 'cross': 2, 'catas': 3}
IDX_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_IDX.items()}


def encode_input(angle1, angle2, r):
    """Featurize a collision: (cos a1, sin a1, cos a2, sin a2, r).

    Trig encoding because raw angles are 2π-periodic and a NN can't easily
    learn that without help. With cos/sin features, the wraparound is
    handled by construction.
    """
    return np.array([
        np.cos(angle1), np.sin(angle1),
        np.cos(angle2), np.sin(angle2),
        float(r)
    ], dtype=np.float32)


def generate_dataset(n_samples: int, seed: int = 42):
    """Generate (X, y_outcome, y_angle) training data by calling zip_cat_clean."""
    from collision.api import zip_cat_clean
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, 5), dtype=np.float32)
    y_outcome = np.zeros(n_samples, dtype=np.int64)
    y_angle_sincos = np.zeros((n_samples, 2), dtype=np.float32)  # encode angle as (cos, sin)
    pt = [0.5, 0.5]
    pt_prev = [0.4, 0.4]
    for i in range(n_samples):
        a1 = rng.uniform(0, 2 * pi)
        a2 = rng.uniform(0, 2 * pi)
        r = int(rng.integers(0, 2))
        new_angle, _, outcome, _, _ = zip_cat_clean(a1, a2, pt, pt_prev, r)
        X[i] = encode_input(a1, a2, r)
        y_outcome[i] = OUTCOME_TO_IDX[outcome]
        y_angle_sincos[i] = [np.cos(new_angle), np.sin(new_angle)]
    return X, y_outcome, y_angle_sincos


# ── Tiny MLP in numpy ────────────────────────────────────────────────────────

class CollisionMLP:
    """3-layer MLP: 5 → 32 → 32 → (4 logits + 2 angle outputs).

    Joint head: same body predicts both classification and regression.
    """
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He init
        self.W1 = rng.normal(0, np.sqrt(2/5), (5, 32)).astype(np.float32)
        self.b1 = np.zeros(32, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2/32), (32, 32)).astype(np.float32)
        self.b2 = np.zeros(32, dtype=np.float32)
        # Outcome head: 4 classes
        self.Wo = rng.normal(0, np.sqrt(2/32), (32, 4)).astype(np.float32)
        self.bo = np.zeros(4, dtype=np.float32)
        # Angle head: 2 outputs (cos, sin)
        self.Wa = rng.normal(0, np.sqrt(2/32), (32, 2)).astype(np.float32)
        self.ba = np.zeros(2, dtype=np.float32)

        # Adam state
        self._params = ['W1', 'b1', 'W2', 'b2', 'Wo', 'bo', 'Wa', 'ba']
        self._m = {k: np.zeros_like(getattr(self, k)) for k in self._params}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._params}
        self._t = 0

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0, z2)
        outcome_logits = h2 @ self.Wo + self.bo
        angle_out = h2 @ self.Wa + self.ba   # (cos, sin) prediction
        return outcome_logits, angle_out, (z1, h1, z2, h2)

    def predict(self, X):
        logits, angle, _ = self.forward(X)
        outcome_idx = logits.argmax(axis=1)
        # Normalize (cos, sin) to unit vector for angle recovery
        norm = np.linalg.norm(angle, axis=1, keepdims=True) + 1e-9
        cs = angle / norm
        new_angle = np.arctan2(cs[:, 1], cs[:, 0]) % (2 * pi)
        return outcome_idx, new_angle

    def step(self, X, y_outcome, y_angle_sincos, lr: float = 3e-3,
             angle_weight: float = 1.0):
        B = X.shape[0]
        logits, angle_pred, (z1, h1, z2, h2) = self.forward(X)

        # Cross-entropy gradient
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(B), y_outcome] = 1.0
        g_logits = (probs - one_hot) / B

        # MSE on angle (cos, sin)
        g_angle = 2.0 * (angle_pred - y_angle_sincos) / B * angle_weight

        # Backprop through heads
        gWo = h2.T @ g_logits
        gbo = g_logits.sum(axis=0)
        gh2_from_outcome = g_logits @ self.Wo.T

        gWa = h2.T @ g_angle
        gba = g_angle.sum(axis=0)
        gh2_from_angle = g_angle @ self.Wa.T

        gh2 = gh2_from_outcome + gh2_from_angle
        gz2 = gh2 * (z2 > 0)

        gW2 = h1.T @ gz2
        gb2 = gz2.sum(axis=0)
        gh1 = gz2 @ self.W2.T
        gz1 = gh1 * (z1 > 0)

        gW1 = X.T @ gz1
        gb1 = gz1.sum(axis=0)

        # Adam
        self._t += 1
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        for name, g in (('W1', gW1), ('b1', gb1), ('W2', gW2), ('b2', gb2),
                        ('Wo', gWo), ('bo', gbo), ('Wa', gWa), ('ba', gba)):
            self._m[name] = b1_ * self._m[name] + (1 - b1_) * g
            self._v[name] = b2_ * self._v[name] + (1 - b2_) * (g * g)
            m_hat = self._m[name] / (1 - b1_ ** self._t)
            v_hat = self._v[name] / (1 - b2_ ** self._t)
            setattr(self, name, getattr(self, name) - lr * m_hat / (np.sqrt(v_hat) + eps))

        # Losses
        loss_ce = -np.log(probs[np.arange(B), y_outcome] + 1e-9).mean()
        loss_angle = ((angle_pred - y_angle_sincos) ** 2).mean()
        return loss_ce, loss_angle

    def save(self, path):
        np.savez(path, **{k: getattr(self, k) for k in self._params})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls()
        for k in m._params:
            setattr(m, k, d[k])
        return m


# ── Wrapper: make trained MLP look like zip_cat ─────────────────────────────

def make_learned_decision(model: CollisionMLP):
    """Return a function that has the same signature as decide_outcome
    but uses the trained MLP."""
    def learned_decide(angle1, angle2, r):
        X = encode_input(angle1, angle2, r).reshape(1, -1)
        idx, _ = model.predict(X)
        return IDX_TO_OUTCOME[int(idx[0])]
    return learned_decide


def make_learned_geometry(model: CollisionMLP):
    """Return a function that has the same signature as zipper_geometry
    but uses the trained MLP."""
    def learned_geometry(angle1, angle2, outcome=None):
        X = encode_input(angle1, angle2, 0).reshape(1, -1)
        _, new_angle = model.predict(X)
        # Honor explicit outcome label if given
        if outcome in ('zipper+', 'zipper-'):
            return float(new_angle[0]), outcome
        return float(new_angle[0]), 'zipper+'  # placeholder
    return learned_geometry
