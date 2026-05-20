"""Pure geometry: post-collision MT state.

Given an outcome label, compute where and at what angle the incoming MT
continues. No decisions, no bookkeeping.

This is the second interpretability surface — to test the effect of a
different bending rule, swap ``zipper_geometry``.

Semantic contract for ablation users
------------------------------------
``zipper_geometry(a1, a2, outcome=label)`` is **label-driven**:

  - outcome='zipper+' → return ``a2 % 2π`` (incoming aligns same-direction)
  - outcome='zipper-' → return ``(a2 + π) % 2π`` (incoming aligns anti-direction)
  - outcome=None     → pick whichever alignment is geometrically closer

When ``decide_outcome`` is monkey-patched to force a label, the geometry
honors that label by producing the corresponding new_angle. This guarantees
that the bundle bookkeeping ("zipper+ bundle" or "zipper- bundle") and the
MT's actual direction are consistent.

If you want the *natural* (closer) alignment regardless of label — e.g. to
study how the bookkeeping behaves under inconsistent state — call
``zipper_geometry(a1, a2)`` with no outcome and use the returned label.
"""
from math import pi, sin, cos
from ._helpers import circ_dist


def zipper_geometry(angle1: float, angle2: float, outcome: str = None) -> tuple:
    """Compute the post-collision angle when the incoming MT zips onto the barrier.

    Parameters
    ----------
    angle1, angle2 : float
        Angles of incoming (1) and barrier (2), in [0, 2π].
    outcome : 'zipper+' | 'zipper-' | None
        If 'zipper+', return same-direction alignment.
        If 'zipper-', return anti-direction alignment.
        If None, pick whichever is closer to angle1 (natural behavior).

    Returns
    -------
    (new_angle, label) : (float, str)
        ``new_angle`` is in [0, 2π). ``label`` is the outcome name corresponding
        to the chosen direction. When ``outcome`` was passed explicitly, ``label``
        will equal that argument.
    """
    same_dir = angle2 % (2 * pi)
    anti_dir = (angle2 + pi) % (2 * pi)

    if outcome == 'zipper+':
        return same_dir, 'zipper+'
    if outcome == 'zipper-':
        return anti_dir, 'zipper-'

    # outcome is None — pick closer alignment
    if circ_dist(angle1, same_dir) <= circ_dist(angle1, anti_dir):
        return same_dir, 'zipper+'
    return anti_dir, 'zipper-'


def step_back_offset(angle1: float, incident: float, d: float) -> tuple:
    """Backward offset along the incoming MT's direction.

    When the simulator is in 'no bundle ID' mode (``no_bdl_id=True``), zippering
    MTs step back from the collision point by ``d/sin(incident)`` units along
    their original direction. This is used to keep MTs from overlapping at the
    exact zipping point.

    Returns the (dx, dy) offset. Caller adds this to ``pt`` to get ``new_pt``.
    """
    if incident < 1e-12:
        return 0.0, 0.0
    step = d / sin(incident)
    return -step * cos(angle1), -step * sin(angle1)
