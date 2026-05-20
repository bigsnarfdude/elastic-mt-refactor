"""Minimal sparse autoencoder in numpy.

Keeps the dependency surface tiny (no torch) so this runs anywhere the
simulator runs. For training-speed reasons we use a single-layer SAE with
ReLU activation and L1 sparsity penalty, batch SGD with Adam.

  encoder : x → h = ReLU(W_e x + b_e)      h ∈ R^d
  decoder : h → x̂ = W_d h + b_d
  loss    : ||x - x̂||^2 + λ ||h||_1
"""
import numpy as np


class SAE:
    def __init__(self, input_dim: int, n_features: int,
                 sparsity_coef: float = 0.04, seed: int = 42):
        rng = np.random.default_rng(seed)
        # He init for ReLU
        self.W_e = rng.normal(0, np.sqrt(2/input_dim),
                              (n_features, input_dim)).astype(np.float32)
        self.b_e = np.zeros(n_features, dtype=np.float32)
        self.W_d = rng.normal(0, np.sqrt(2/n_features),
                              (input_dim, n_features)).astype(np.float32)
        self.b_d = np.zeros(input_dim, dtype=np.float32)
        self.sparsity_coef = sparsity_coef
        # Adam state
        self._m = {k: np.zeros_like(getattr(self, k))
                   for k in ('W_e', 'b_e', 'W_d', 'b_d')}
        self._v = {k: np.zeros_like(getattr(self, k))
                   for k in ('W_e', 'b_e', 'W_d', 'b_d')}
        self._t_adam = 0

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x @ self.W_e.T + self.b_e)

    def decode(self, h: np.ndarray) -> np.ndarray:
        return h @ self.W_d.T + self.b_d

    def forward(self, x: np.ndarray) -> tuple:
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

    def loss(self, x: np.ndarray) -> tuple:
        x_hat, h = self.forward(x)
        recon = float(((x - x_hat) ** 2).mean())
        sparse = float(self.sparsity_coef * np.abs(h).mean())
        return recon + sparse, recon, sparse

    def step(self, x: np.ndarray, lr: float = 1e-3) -> dict:
        """Single Adam update on batch ``x`` of shape (B, input_dim)."""
        B = x.shape[0]
        h = self.encode(x)
        x_hat = self.decode(h)
        # Gradients
        # dL_recon/dx_hat = 2(x_hat - x) / (B * input_dim)
        grad_xhat = 2.0 * (x_hat - x) / (B * x.shape[1])
        # decoder
        gW_d = grad_xhat.T @ h
        gb_d = grad_xhat.sum(axis=0)
        # back through decoder
        grad_h_recon = grad_xhat @ self.W_d
        # sparsity: dL_sparse/dh = sparsity_coef * sign(h) / (B * n_features)
        grad_h_sparse = self.sparsity_coef * np.sign(h) / (B * h.shape[1])
        grad_h = grad_h_recon + grad_h_sparse
        # back through ReLU
        grad_h = grad_h * (h > 0)
        # encoder
        gW_e = grad_h.T @ x
        gb_e = grad_h.sum(axis=0)

        # Adam
        self._t_adam += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for k, g in (('W_e', gW_e), ('b_e', gb_e),
                     ('W_d', gW_d), ('b_d', gb_d)):
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * (g * g)
            m_hat = self._m[k] / (1 - beta1 ** self._t_adam)
            v_hat = self._v[k] / (1 - beta2 ** self._t_adam)
            setattr(self, k, getattr(self, k) - lr * m_hat / (np.sqrt(v_hat) + eps))

        return dict(
            recon=float(((x - x_hat) ** 2).mean()),
            sparse=float(self.sparsity_coef * np.abs(h).mean()),
            active_frac=float((h > 0).mean()),
        )

    def save(self, path):
        np.savez(path,
                 W_e=self.W_e, b_e=self.b_e, W_d=self.W_d, b_d=self.b_d,
                 sparsity_coef=self.sparsity_coef)

    @classmethod
    def load(cls, path, **kwargs):
        d = np.load(path)
        n_features, input_dim = d['W_e'].shape
        sae = cls(input_dim=input_dim, n_features=n_features,
                  sparsity_coef=float(d['sparsity_coef']), **kwargs)
        sae.W_e = d['W_e']
        sae.b_e = d['b_e']
        sae.W_d = d['W_d']
        sae.b_d = d['b_d']
        return sae
