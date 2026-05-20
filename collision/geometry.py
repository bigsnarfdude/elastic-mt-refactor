"""Pure geometry: post-collision MT state.

Given an outcome label, compute where and at what angle the incoming MT
continues. No decisions, no bookkeeping.

This is the second interpretability surface — to test the effect of a
different bending rule, swap ``zipper_geometry``. To test the effect of
a different step-back rule, swap ``step_back_offset``.
"""
from math import pi, sin, cos


def zipper_geometry(angle1: float, angle2: float) -> tuple:
    """Compute the post-collision angle when the incoming MT zips onto the barrier.

    The incoming MT (angle1) bends to align with the barrier (angle2). The
    barrier has two orientations — same direction (angle2) and anti-direction
    (angle2 + π). We pick whichever requires the smaller bend.

    Returns
    -------
    (new_angle, label) : (float, str)
        ``new_angle`` is in [0, 2π). ``label`` is 'zipper+' if the MT aligned
        same-direction, 'zipper-' if anti-direction.
    """
    same_dir = angle2 % (2 * pi)
    anti_dir = (angle2 + pi) % (2 * pi)
    d_same = _circ_dist(angle1, same_dir)
    d_anti = _circ_dist(angle1, anti_dir)
    if d_same <= d_anti:
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


def _circ_dist(a: float, b: float) -> float:
    d = abs(a - b) % (2 * pi)
    return min(d, 2 * pi - d)
