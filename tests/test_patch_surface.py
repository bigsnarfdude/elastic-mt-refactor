"""The ablation surface contract: monkey-patching `collision.decision.decide_outcome`
or `collision.geometry.zipper_geometry` MUST take effect when zip_cat_clean is called.

This is the *whole point* of the refactor. If these tests fail, the refactor
has not actually delivered a clean ablation surface — even if the equivalence
test passes. The bug we're guarding against is using ``from .decision import
decide_outcome`` (early binding) inside api.py, which makes the patch invisible.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pytest


@pytest.fixture(autouse=True)
def reset_modules():
    """Each test gets a clean slate — restore the original functions after."""
    import collision.decision as d
    import collision.geometry as g
    _orig_decide = d.decide_outcome
    _orig_zipgeo = g.zipper_geometry
    _orig_incident = d.incident_angle
    yield
    d.decide_outcome = _orig_decide
    g.zipper_geometry = _orig_zipgeo
    d.incident_angle = _orig_incident


# ── 1. Patching the decision propagates ──────────────────────────────────────

def test_patch_decision_to_always_cross():
    import collision.decision
    from collision.api import zip_cat_clean

    # Before patch: small angle → zipper+
    out_before = zip_cat_clean(0.1, 0.2, [0.5, 0.5], [0.4, 0.4], 0)
    assert out_before[2] == 'zipper+', f"sanity: expected zipper+ got {out_before[2]}"

    # Patch
    collision.decision.decide_outcome = lambda a1, a2, r: 'cross'

    # After patch: every outcome must be 'cross'
    out_after = zip_cat_clean(0.1, 0.2, [0.5, 0.5], [0.4, 0.4], 0)
    assert out_after[2] == 'cross', (
        f"PATCH IGNORED — decision.decide_outcome was monkey-patched to return "
        f"'cross' but zip_cat_clean returned outcome={out_after[2]}. This means "
        f"collision/api.py is doing 'from .decision import decide_outcome' "
        f"(early binding) and the patch surface is broken."
    )


def test_patch_decision_to_always_catas():
    import collision.decision
    from collision.api import zip_cat_clean
    collision.decision.decide_outcome = lambda a1, a2, r: 'catas'
    out = zip_cat_clean(0.1, 0.2, [0.5, 0.5], [0.4, 0.4], 0)
    assert out[2] == 'catas'


def test_patch_decision_to_always_zipper_plus():
    import collision.decision
    from collision.api import zip_cat_clean
    collision.decision.decide_outcome = lambda a1, a2, r: 'zipper+'
    out = zip_cat_clean(0.5, 0.6, [0.5, 0.5], [0.4, 0.4], 0)
    assert out[2] == 'zipper+'
    # new_angle should match the un-patched geometry (still zipper+)
    # since the patch only changed the decision


# ── 2. Patching the geometry propagates ──────────────────────────────────────

def test_patch_zipper_geometry():
    import collision.geometry
    from collision.api import zip_cat_clean

    # Patch geometry to return a fixed angle. Decision (small angle) still says zipper+.
    collision.geometry.zipper_geometry = lambda a1, a2: (3.14159, 'zipper+')

    out = zip_cat_clean(0.1, 0.2, [0.5, 0.5], [0.4, 0.4], 0)
    assert out[2] == 'zipper+', f"decision unchanged, should still be zipper+ got {out[2]}"
    assert abs(out[0] - 3.14159) < 1e-9, (
        f"PATCH IGNORED — geometry.zipper_geometry was monkey-patched to return "
        f"angle=3.14159 but zip_cat_clean returned new_angle={out[0]}. This means "
        f"collision/api.py is using early binding for zipper_geometry."
    )


# ── 3. Sanity: un-patched behavior still works ──────────────────────────────

def test_unpatched_baseline():
    from collision.api import zip_cat_clean
    # Small incident angle → zipper outcome
    out = zip_cat_clean(0.1, 0.2, [0.5, 0.5], [0.4, 0.4], 0)
    assert out[2] in ('zipper+', 'zipper-')
    # Large incident angle, r=0 → catastrophe
    out = zip_cat_clean(0.1, 1.5, [0.5, 0.5], [0.4, 0.4], 0)
    assert out[2] == 'catas'
    # Large incident angle, r=1 → crossover
    out = zip_cat_clean(0.1, 1.5, [0.5, 0.5], [0.4, 0.4], 1)
    assert out[2] == 'cross'


# ── 4. Patching survives across many calls (no caching) ─────────────────────

def test_patch_persists():
    import collision.decision
    from collision.api import zip_cat_clean
    collision.decision.decide_outcome = lambda a1, a2, r: 'catas'
    for i in range(50):
        out = zip_cat_clean(0.1 + i*0.01, 0.2, [0.5, 0.5], [0.4, 0.4], 0)
        assert out[2] == 'catas', f"call {i} returned {out[2]} instead of catas"


# ── 5. The cross-module test: zippering.zip_cat (the shim) honors patches ──

def test_patch_via_zippering_shim():
    """sim_algs imports zippering.zip_cat. Make sure the patch takes effect
    through that import path, not just direct calls to api.zip_cat_clean."""
    import collision.decision
    from zippering import zip_cat
    collision.decision.decide_outcome = lambda a1, a2, r: 'cross'
    out = zip_cat(0.1, 0.2, [0.5, 0.5], [0.4, 0.4], 0)
    assert out[2] == 'cross', (
        f"PATCH INVISIBLE THROUGH SHIM — zippering.zip_cat returned {out[2]} "
        f"after decide_outcome was patched to 'cross'. The shim must forward "
        f"to collision.api.zip_cat_clean which late-binds the decision."
    )


if __name__ == '__main__':
    print("Running patch-surface tests...")
    failures = 0
    tests = [
        test_patch_decision_to_always_cross,
        test_patch_decision_to_always_catas,
        test_patch_decision_to_always_zipper_plus,
        test_patch_zipper_geometry,
        test_unpatched_baseline,
        test_patch_persists,
        test_patch_via_zippering_shim,
    ]
    for t in tests:
        # manual fixture reset
        import collision.decision as d
        import collision.geometry as g
        _od = d.decide_outcome
        _og = g.zipper_geometry
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            print(f"  ✗ {t.__name__}")
            print(f"    {str(e)[:200]}")
            failures += 1
        finally:
            d.decide_outcome = _od
            g.zipper_geometry = _og
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
