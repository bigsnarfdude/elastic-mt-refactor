"""Tests for the SAE pipeline.

Three modules, three test groups:
  - sae.snapshot: encoding a single order_hist into a histogram vector
  - sae.model:    the numpy SAE — encode/decode shapes, gradient direction,
                    save/load round-trip
  - sae.probe:    hand-designed metrics computed from snapshot vectors
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pytest
import numpy as np
from math import pi


# ─────────────────────────────────────────────────────────────────────────────
# sae.snapshot
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotVector:
    def test_empty_returns_zero_vec(self):
        from sae.snapshot import snapshot_vector, N_BINS
        v = snapshot_vector(([], [], 0.0, []))
        assert v.shape == (N_BINS,)
        assert v.sum() == 0

    def test_normalizes_to_unit_sum(self):
        from sae.snapshot import snapshot_vector
        rng = np.random.default_rng(0)
        angles = rng.uniform(0, 2*pi, 50)
        lengths = rng.uniform(0.1, 1.0, 50)
        v = snapshot_vector((list(angles), list(lengths), float(sum(lengths)), [0]*50))
        assert abs(v.sum() - 1.0) < 1e-6

    def test_aligned_concentrates_in_one_bin(self):
        """All MTs at angle 0 → first bin (or last via wraparound) should hold mass."""
        from sae.snapshot import snapshot_vector
        # All exactly at 0 — first bin gets everything
        v = snapshot_vector(([0.0]*10, [1.0]*10, 10.0, [0]*10))
        # The mass is concentrated, not spread
        max_bin = v.argmax()
        assert v[max_bin] > 0.9, f"expected near-1 in one bin, got max={v[max_bin]}"

    def test_random_spreads_uniformly(self):
        """Uniformly random angles → roughly uniform histogram (mean bin ~ 1/N)."""
        from sae.snapshot import snapshot_vector, N_BINS
        rng = np.random.default_rng(1)
        angles = rng.uniform(0, 2*pi, 5000)
        v = snapshot_vector((list(angles), [1.0]*5000, 5000.0, [0]*5000))
        # max-min should be small (Poisson noise around 1/N_BINS)
        expected = 1.0 / N_BINS
        # 5000 samples, 36 bins → ~139 per bin. std ~ sqrt(139) ~ 12. ratio ~ 0.085.
        assert v.max() < expected * 2.5, (
            f"max bin {v.max()} too far from uniform {expected}"
        )

    def test_nematic_equivalence(self):
        """Angle θ and angle θ+π land in the same nematic bin."""
        from sae.snapshot import snapshot_vector
        # Half at 0.3, half at 0.3+π. Nematically the same.
        v1 = snapshot_vector(([0.3]*10, [1.0]*10, 10.0, [0]*10))
        v2 = snapshot_vector(([0.3 + pi]*10, [1.0]*10, 10.0, [0]*10))
        assert np.allclose(v1, v2, atol=1e-6), (
            f"nematic equivalence broken: peak at bin {v1.argmax()} vs {v2.argmax()}"
        )

    def test_length_weighting(self):
        """A long MT contributes more than a short one to its bin."""
        from sae.snapshot import snapshot_vector
        # one short MT at 0, one long MT at π/2
        angles = [0.0, pi / 2]
        lengths = [0.1, 10.0]
        v = snapshot_vector((angles, lengths, sum(lengths), [0, 0]))
        # bin for π/2 should have ~10/(10+0.1) ≈ 0.99 of the mass
        from sae.snapshot import N_BINS
        bin_pi2 = int((pi / 2) / pi * N_BINS)
        assert v[bin_pi2] > 0.95


class TestAggregateRegions:
    def test_single_region_matches_snapshot_vector(self):
        from sae.snapshot import aggregate_regions, snapshot_vector
        geom = ([0.1, 0.5], [1.0, 1.0], 2.0, [0, 0])
        order = [[geom]]   # one time point, one region
        order_t = [1.0]
        result = aggregate_regions((order, order_t))
        expected = snapshot_vector(geom)
        assert result.shape == (1, expected.shape[0])
        assert np.allclose(result[0], expected, atol=1e-6)

    def test_multiple_regions_combine_by_length(self):
        """Two regions with different angles combine in proportion to MT length."""
        from sae.snapshot import aggregate_regions
        # region 1: 100 mass at angle 0
        # region 2: 100 mass at angle π/2
        # combined should be 50/50 split between two bins
        g1 = ([0.0]*100, [1.0]*100, 100.0, [0]*100)
        g2 = ([pi/2]*100, [1.0]*100, 100.0, [0]*100)
        order = [[g1, g2]]
        order_t = [1.0]
        result = aggregate_regions((order, order_t))
        from sae.snapshot import N_BINS
        bin_0 = 0
        bin_pi2 = int((pi / 2) / pi * N_BINS)
        # roughly 50/50 (allowing the wraparound near 0 to split)
        assert 0.3 < result[0, bin_pi2] < 0.7

    def test_empty_region_handled(self):
        from sae.snapshot import aggregate_regions
        order = [[None]]    # no MTs
        order_t = [1.0]
        result = aggregate_regions((order, order_t))
        assert result.shape[0] == 1
        # all-zero if no MTs contributed
        assert abs(result[0].sum()) < 1e-6


class TestLoadTrajectory:
    def test_load_from_pickle(self, tmp_path):
        """Write a pickle in the orderp format, load it via load_trajectory."""
        import pickle
        from sae.snapshot import load_trajectory, N_BINS
        # Build an order_pickle: list of regions per time, plus order_t
        geom_t0 = ([0.1, 0.2, 0.3], [1.0, 1.0, 1.0], 3.0, [0, 0, 0])
        geom_t1 = ([pi/2]*5, [1.0]*5, 5.0, [0]*5)
        order = [[geom_t0], [geom_t1]]   # 2 time points, 1 region each
        order_t = [0.5, 1.0]
        path = tmp_path / "orderp_seed42_0idx.pickle"
        with open(path, 'wb') as f:
            pickle.dump((order, order_t), f)
        snaps = load_trajectory(path)
        assert snaps.shape == (2, N_BINS)
        # t0 mass should be in the low bins (angles 0.1-0.3)
        # t1 mass should peak near bin N_BINS/2 (π/2)
        bin_pi2 = int((pi / 2) / pi * N_BINS)
        assert snaps[1, bin_pi2] > 0.9

    def test_load_missing_file_raises(self, tmp_path):
        from sae.snapshot import load_trajectory
        with pytest.raises((FileNotFoundError, OSError)):
            load_trajectory(tmp_path / "does_not_exist.pickle")


# ─────────────────────────────────────────────────────────────────────────────
# sae.model.SAE
# ─────────────────────────────────────────────────────────────────────────────

class TestSAEModel:
    def test_shapes(self):
        from sae.model import SAE
        sae = SAE(input_dim=36, n_features=64, seed=0)
        x = np.random.default_rng(0).normal(0, 1, (10, 36)).astype(np.float32)
        h = sae.encode(x)
        x_hat = sae.decode(h)
        assert h.shape == (10, 64)
        assert x_hat.shape == (10, 36)

    def test_relu_nonneg(self):
        from sae.model import SAE
        sae = SAE(input_dim=36, n_features=64, seed=0)
        x = np.random.default_rng(0).normal(0, 1, (100, 36)).astype(np.float32)
        h = sae.encode(x)
        assert (h >= 0).all()

    def test_loss_decreases_during_training(self):
        """Average loss over an epoch should drop after a few epochs."""
        from sae.model import SAE
        rng = np.random.default_rng(42)
        # 50 simple periodic patterns
        X = np.array([
            np.exp(-((np.arange(36) - i) / 4) ** 2)
            for i in rng.integers(0, 36, 50)
        ], dtype=np.float32)
        X /= X.sum(axis=1, keepdims=True)

        sae = SAE(input_dim=36, n_features=16, sparsity_coef=0.005, seed=0)
        losses = []
        for epoch in range(20):
            ep_losses = []
            perm = rng.permutation(50)
            for i in range(0, 50, 8):
                m = sae.step(X[perm[i:i+8]], lr=3e-3)
                ep_losses.append(m['recon'])
            losses.append(np.mean(ep_losses))
        # last-epoch loss should be at least 30% lower than first
        assert losses[-1] < losses[0] * 0.7, (
            f"loss didn't decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_save_load_round_trip(self, tmp_path):
        from sae.model import SAE
        sae = SAE(input_dim=10, n_features=8, sparsity_coef=0.02, seed=42)
        # train a couple steps so weights aren't init
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (20, 10)).astype(np.float32)
        for _ in range(5):
            sae.step(X, lr=1e-3)

        path = tmp_path / "sae.npz"
        sae.save(str(path))
        sae2 = SAE.load(str(path))

        # outputs should match
        x = rng.normal(0, 1, (5, 10)).astype(np.float32)
        h1, _ = sae.forward(x)
        h2, _ = sae2.forward(x)
        assert np.allclose(h1, h2, atol=1e-7)

    def test_sparsity_kills_features_at_high_coef(self):
        """With high sparsity_coef, most features should silence."""
        from sae.model import SAE
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (100, 36)).astype(np.float32)
        sae = SAE(input_dim=36, n_features=64, sparsity_coef=1.0, seed=0)
        for _ in range(30):
            sae.step(X, lr=3e-3)
        h = sae.encode(X)
        active_frac = (h > 0).mean()
        assert active_frac < 0.5, f"too many active features under heavy sparsity: {active_frac}"


# ─────────────────────────────────────────────────────────────────────────────
# sae.probe — hand-designed metrics from snapshot vectors
# ─────────────────────────────────────────────────────────────────────────────

class TestProbe:
    def test_s2_of_perfectly_aligned_is_one(self):
        """All mass in one bin → S₂ = 1."""
        from sae.probe import s2_from_snapshot
        from sae.snapshot import N_BINS
        v = np.zeros(N_BINS, dtype=np.float32)
        v[0] = 1.0
        assert abs(s2_from_snapshot(v) - 1.0) < 1e-3

    def test_s2_of_uniform_is_zero(self):
        """Uniform histogram → S₂ ≈ 0."""
        from sae.probe import s2_from_snapshot
        from sae.snapshot import N_BINS
        v = np.ones(N_BINS, dtype=np.float32) / N_BINS
        assert abs(s2_from_snapshot(v)) < 1e-6

    def test_s2_of_perpendicular_modes_is_zero(self):
        """50% at θ and 50% at θ+π/2 → cancellation, S₂ = 0."""
        from sae.probe import s2_from_snapshot
        from sae.snapshot import N_BINS
        v = np.zeros(N_BINS, dtype=np.float32)
        v[0] = 0.5
        v[N_BINS // 2] = 0.5
        assert abs(s2_from_snapshot(v)) < 0.05

    def test_omega_in_valid_range(self):
        """Ω is always in [0, π)."""
        from sae.probe import omega_from_snapshot
        from sae.snapshot import N_BINS
        rng = np.random.default_rng(0)
        for _ in range(50):
            v = rng.uniform(0, 1, N_BINS).astype(np.float32)
            v /= v.sum()
            omega = omega_from_snapshot(v)
            assert 0 <= omega < pi

    def test_entropy_of_uniform_is_max(self):
        """Uniform distribution maximizes entropy."""
        from sae.probe import entropy_from_snapshot
        from sae.snapshot import N_BINS
        uniform = np.ones(N_BINS, dtype=np.float32) / N_BINS
        peaked = np.zeros(N_BINS, dtype=np.float32); peaked[0] = 1.0
        assert entropy_from_snapshot(uniform) > entropy_from_snapshot(peaked)
        # max entropy = log(N_BINS)
        import math
        assert abs(entropy_from_snapshot(uniform) - math.log(N_BINS)) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# end-to-end: snapshot → SAE → probe
# ─────────────────────────────────────────────────────────────────────────────

def test_e2e_aligned_vs_random_have_different_sae_features():
    """A trained SAE on mixed (aligned, random) data should have features that
    activate differently on each class.
    """
    from sae.snapshot import snapshot_vector, N_BINS
    from sae.model import SAE
    rng = np.random.default_rng(42)
    snaps = []
    labels = []
    # 50 aligned at random angles, 50 isotropic
    for _ in range(50):
        center = rng.uniform(0, pi)
        angles = (rng.normal(center, 0.05, 200)) % (2*pi)
        snaps.append(snapshot_vector((list(angles), [1.0]*200, 200.0, [0]*200)))
        labels.append('aligned')
    for _ in range(50):
        angles = rng.uniform(0, 2*pi, 200)
        snaps.append(snapshot_vector((list(angles), [1.0]*200, 200.0, [0]*200)))
        labels.append('random')

    X = np.array(snaps, dtype=np.float32)
    sae = SAE(input_dim=N_BINS, n_features=16, sparsity_coef=0.005, seed=0)
    for epoch in range(100):
        perm = rng.permutation(100)
        for i in range(0, 100, 16):
            sae.step(X[perm[i:i+16]], lr=3e-3)

    H = sae.encode(X)
    # mean activation of any feature must differ between aligned and random
    aligned_idx = [i for i, l in enumerate(labels) if l == 'aligned']
    random_idx = [i for i, l in enumerate(labels) if l == 'random']
    diff = H[aligned_idx].mean(0) - H[random_idx].mean(0)
    # at least one feature has |diff| above noise floor
    assert np.max(np.abs(diff)) > 0.01, (
        f"no feature discriminates aligned vs random: max |diff| = {np.max(np.abs(diff))}"
    )


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
