"""When `decide_outcome` is monkey-patched to force a label different from
what the natural (closer-alignment) geometry would pick, `zip_cat_clean` must
return a `new_angle` that is consistent with the forced label — not the angle
the natural geometry would have produced.

Why this matters
----------------
Imagine an ablation that forces every collision to 'zipper+'. The user expects
every MT to align *same-direction* with its barrier. If the geometry silently
picks anti-direction (because that's geometrically closer to angle1), the MT
ends up anti-aligned but the simulator's bundle bookkeeping records it as
zipper+. The simulation is now in a corrupted state — bundles labeled '+' but
containing anti-aligned MTs — and any downstream interpretation of zipper+ vs
zipper- is wrong.

The fix: `api.zip_cat_clean` passes the decided outcome to `zipper_geometry`
as a kwarg, and the geometry honors it.

This test would have failed under the bug (api ignored the label) and passes
under the fix (api propagates the label).
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from math import pi


def _restore():
    """Restore decide_outcome and zipper_geometry to their original module-level."""
    import importlib, collision.decision as d, collision.geometry as g
    importlib.reload(d)
    importlib.reload(g)


def test_forced_zipper_plus_at_antiparallel_pair():
    """At antiparallel angles (a1=0, a2≈π), natural geometry picks 'zipper-'
    (closer = back to 0). When user forces 'zipper+', new_angle must be ≈π
    (same-direction as barrier), not the natural 0.
    """
    import collision.decision
    from collision.api import zip_cat_clean

    a1, a2 = 0.0, pi - 0.05   # incident is small (zipper territory)
    pt = [0.5, 0.5]
    pt_prev = [0.4, 0.4]

    # Sanity: un-patched, natural choice is zipper- (because angle 0 is closer
    # to (a2+π) = -0.05 mod 2π = 6.23 than to a2 = 3.09)
    out_natural = zip_cat_clean(a1, a2, pt, pt_prev, 0)
    assert out_natural[2] == 'zipper-', (
        f"sanity broke — expected natural choice 'zipper-' got {out_natural[2]}"
    )

    # Force zipper+
    collision.decision.decide_outcome = lambda a1, a2, r: 'zipper+'
    try:
        out = zip_cat_clean(a1, a2, pt, pt_prev, 0)
        assert out[2] == 'zipper+', f"outcome should be forced zipper+, got {out[2]}"
        # The angle must be the SAME-DIRECTION alignment with barrier = a2 mod 2π
        expected_angle = a2 % (2 * pi)
        actual = out[0]
        assert abs((actual - expected_angle) % (2*pi)) < 1e-9 or \
               abs(2*pi - (actual - expected_angle) % (2*pi)) < 1e-9, (
            f"LABEL/GEOMETRY MISMATCH: outcome was forced to 'zipper+' (expected "
            f"new_angle = a2 = {expected_angle:.4f}), but new_angle = {actual:.4f}. "
            f"This means api.py is not passing outcome to zipper_geometry."
        )
    finally:
        _restore()


def test_forced_zipper_minus_at_parallel_pair():
    """At parallel angles (a1=0, a2=0.05), natural geometry picks 'zipper+'.
    When user forces 'zipper-', new_angle must be ≈π (anti-direction).
    """
    import collision.decision
    from collision.api import zip_cat_clean

    a1, a2 = 0.0, 0.05
    pt = [0.5, 0.5]; pt_prev = [0.4, 0.4]

    out_natural = zip_cat_clean(a1, a2, pt, pt_prev, 0)
    assert out_natural[2] == 'zipper+', f"sanity: expected natural 'zipper+' got {out_natural[2]}"

    collision.decision.decide_outcome = lambda a1, a2, r: 'zipper-'
    try:
        out = zip_cat_clean(a1, a2, pt, pt_prev, 0)
        assert out[2] == 'zipper-'
        expected_angle = (a2 + pi) % (2 * pi)
        actual = out[0]
        diff = (actual - expected_angle) % (2*pi)
        assert min(diff, 2*pi - diff) < 1e-9, (
            f"new_angle should be (a2 + π) = {expected_angle:.4f}, got {actual:.4f}"
        )
    finally:
        _restore()


def test_unpatched_behavior_unchanged():
    """When no patches are applied, the new label-driven geometry must produce
    exactly the same (new_angle, outcome) as the original natural geometry.

    This is what allows the 100k-case equivalence test to keep passing.
    """
    from collision.api import zip_cat_clean
    import numpy as np
    rng = np.random.default_rng(123)
    for _ in range(1000):
        a1 = rng.uniform(0, 2*pi)
        a2 = rng.uniform(0, 2*pi)
        r = int(rng.integers(0, 2))
        # The fact that zipper outcomes get the closer-direction angle is the
        # contract of un-patched behavior. We verify by checking that the
        # returned new_angle minimizes the angular bend from angle1.
        out = zip_cat_clean(a1, a2, [0.5, 0.5], [0.4, 0.4], r)
        if out[2] in ('zipper+', 'zipper-'):
            opt_plus = a2 % (2*pi)
            opt_minus = (a2 + pi) % (2*pi)
            chosen = out[0]
            def cdist(x, y):
                d = abs(x - y) % (2*pi)
                return min(d, 2*pi - d)
            d_chosen = cdist(a1, chosen)
            d_alt = cdist(a1, opt_minus if chosen == opt_plus else opt_plus)
            assert d_chosen <= d_alt + 1e-9, (
                f"un-patched: chose the further alignment! a1={a1} a2={a2} "
                f"chose={chosen} other={opt_minus if chosen == opt_plus else opt_plus}"
            )


if __name__ == '__main__':
    tests = [
        test_forced_zipper_plus_at_antiparallel_pair,
        test_forced_zipper_minus_at_parallel_pair,
        test_unpatched_behavior_unchanged,
    ]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            print(f"  ✗ {t.__name__}")
            print(f"    {str(e)[:300]}")
            fails += 1
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
