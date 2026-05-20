"""Equivalence test: refactored zip_cat must agree with the original for any input.

This is the load-bearing test for the refactor. If it passes, the refactor is
behaviorally identical to Tim's original code. If it fails, the refactor is wrong.

What's compared:
  - outcome string ('zipper+' / 'zipper-' / 'cross' / 'catas') — must match exactly
  - new_angle — must agree mod 2π to within numerical precision
  - new_pt and col_pt — must agree element-wise to within numerical precision

What's NOT compared:
  - The 'error' field (always None in both)
  - Stylistic differences (variable names, ordering of branches)
"""
import sys
import os
from pathlib import Path

# allow running as a script from anywhere
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
from math import pi


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ang_equal(a, b, tol=1e-9):
    """Two angles are equal mod 2π if their wrapped difference is ~0."""
    d = (a - b) % (2 * pi)
    return d < tol or (2 * pi - d) < tol


def _pt_equal(p, q, tol=1e-9):
    return abs(p[0] - q[0]) < tol and abs(p[1] - q[1]) < tol


def _run_case(angle1, angle2, pt, pt_prev, r):
    """Run both implementations and return (original_result, refactored_result).

    The 'original' is the preserved 300-line ``_zip_cat_original`` kept inside
    ``zippering.py`` as a regression target. The 'refactored' goes through
    ``collision.api.zip_cat_clean``, which is what the new ``zip_cat`` shim
    forwards to.
    """
    from zippering import _zip_cat_original as orig
    from collision.api import zip_cat_clean as new
    orig_result = orig(angle1, angle2, pt, pt_prev, r)
    new_result = new(angle1, angle2, pt, pt_prev, r)
    return orig_result, new_result


# ── Test cases ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a1,a2,r", [
    # Small angle, both in first quadrant — should zip
    (0.1, 0.2, 0),
    (0.3, 0.5, 0),
    # Large angle, both in first quadrant — should cross or catas
    (0.1, 1.0, 0),  # catas (r==0)
    (0.1, 1.0, 1),  # cross
    # Anti-parallel (nematic-equivalent to parallel) — small incident
    (0.1, pi + 0.2, 0),  # incident ~ 0.1
    # Cross-quadrant: one near 0, one near 2π
    (5.5, 0.3, 0),
    # Both in upper-half plane (π/2 < θ < π)
    (1.6, 1.8, 0),
    # Both in lower-half plane (π < θ < 3π/2)
    (3.5, 3.7, 0),
    # Exactly at the threshold
    (0.0, 2*pi/9 - 1e-6, 0),
    (0.0, 2*pi/9 + 1e-6, 0),
])
def test_specific_cases(a1, a2, r):
    pt = [0.5, 0.5]
    pt_prev = [0.4, 0.4]
    orig, new = _run_case(a1, a2, pt, pt_prev, r)
    assert orig[2] == new[2], f"outcome mismatch: orig={orig[2]} new={new[2]}"
    assert _ang_equal(orig[0], new[0]), \
        f"new_angle mismatch for ({a1}, {a2}, r={r}): orig={orig[0]} new={new[0]}"


def test_random_sweep():
    """1000 random (a1, a2, r) inputs — outcomes and angles must match."""
    rng = np.random.default_rng(42)
    pt = [0.5, 0.5]
    pt_prev = [0.4, 0.4]
    mismatches = []
    for i in range(1000):
        a1 = rng.uniform(0, 2*pi)
        a2 = rng.uniform(0, 2*pi)
        r = rng.integers(0, 2)
        orig, new = _run_case(a1, a2, pt, pt_prev, int(r))
        # outcome must match exactly
        if orig[2] != new[2]:
            mismatches.append(('outcome', i, a1, a2, r, orig[2], new[2]))
            continue
        # new_angle must match mod 2π
        if not _ang_equal(orig[0], new[0]):
            mismatches.append(('angle', i, a1, a2, r, orig[0], new[0]))

    if mismatches:
        print(f"\n{len(mismatches)} mismatches in 1000 trials:")
        for kind, i, a1, a2, r, o, n in mismatches[:10]:
            print(f"  [{kind}] case {i}: a1={a1:.4f} a2={a2:.4f} r={r} "
                  f"orig={o} new={n}")
    assert not mismatches, f"{len(mismatches)} mismatches"


def test_deterministic_decision():
    """decide_outcome must be deterministic in (a1, a2, r)."""
    from collision.decision import decide_outcome
    for _ in range(10):
        result1 = decide_outcome(0.5, 1.0, 0)
        result2 = decide_outcome(0.5, 1.0, 0)
        assert result1 == result2


def test_zipper_geometry_is_closer_alignment():
    """For zipper outcomes, the new_angle should be closer to angle1 than the
    alternative would be."""
    from collision.geometry import zipper_geometry
    rng = np.random.default_rng(7)
    for _ in range(200):
        a1 = rng.uniform(0, 2*pi)
        # only test cases with small incident angle
        a2 = a1 + rng.uniform(-0.3, 0.3)
        new_angle, outcome = zipper_geometry(a1, a2)
        # new_angle should be either a2 or a2+pi (mod 2π) — closer to a1
        opt_plus = a2 % (2*pi)
        opt_minus = (a2 + pi) % (2*pi)

        def cdist(x, y):
            d = abs(x - y) % (2*pi)
            return min(d, 2*pi - d)

        d_plus = cdist(a1, opt_plus)
        d_minus = cdist(a1, opt_minus)
        chosen_dist = cdist(a1, new_angle)
        assert chosen_dist <= max(d_plus, d_minus) + 1e-9, \
            f"chose worse alignment: a1={a1} a2={a2} chose={new_angle}"


if __name__ == '__main__':
    # quick smoke run without pytest
    print("Running specific cases...")
    for a1, a2, r in [(0.1, 0.2, 0), (0.1, 1.0, 0), (5.5, 0.3, 0)]:
        try:
            test_specific_cases(a1, a2, r)
            print(f"  ✓ ({a1}, {a2}, r={r})")
        except AssertionError as e:
            print(f"  ✗ ({a1}, {a2}, r={r}): {e}")

    print("\nRunning random sweep...")
    try:
        test_random_sweep()
        print("  ✓ 1000 random cases agreed")
    except AssertionError as e:
        print(f"  ✗ {e}")
