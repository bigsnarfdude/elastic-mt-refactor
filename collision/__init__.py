"""Collision logic, refactored.

The original ``zippering.zip_cat`` does three jobs in one 300-line function:

  1. Decide what outcome happens (zipper+/-/cross/catas) based on angles
  2. Compute the post-collision geometry (new MT angle and position)
  3. Implicitly signal bundle bookkeeping via the return tuple

After refactor, each job is its own function:

  - ``collision.decision.decide_outcome(a1, a2, r) -> str``
  - ``collision.geometry.zipper_geometry(a1, a2) -> (new_angle, label)``
  - ``collision.api.zip_cat_clean(...)`` — backward-compatible drop-in

The point of the refactor is **clean ablation**. Any one of the three can be
swapped without breaking the others. See ``tests/test_equivalence.py`` for
the proof that the refactored code reproduces the original behavior.
"""
