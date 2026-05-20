"""Pure decision: which collision outcome happens.

This module has no state, no side effects, and no geometry. Given the angles
of two MTs at a collision and a coin-flip ``r``, return a string label.

This is the *interpretability surface* — to randomize the collision rule for
an ablation, swap this single function:

    >>> import collision.decision
    >>> collision.decision.decide_outcome = lambda a1, a2, r: 'cross'
    # Now every collision becomes a crossover.
"""
from math import pi
from ._helpers import circ_dist

# Tim's critical zippering angle: collisions below this incident angle zip,
# collisions above it either crossover (r==1) or catastrophe (r==0).
TH_CRIT = 2 * pi / 9   # 40°


def incident_angle(angle1: float, angle2: float) -> float:
    """Acute angle between two rods, in [0, π/2].

    Rods are nematic — angle θ and angle θ+π represent the same orientation —
    so the rod-rod angle is always in [0, π/2]. The 13 quadrant cases in the
    original ``zip_cat`` are all computing this same quantity.
    """
    d = abs(angle1 - angle2) % (2 * pi)
    d = min(d, 2 * pi - d)          # angle between rays, in [0, π]
    return min(d, pi - d)            # angle between rods, in [0, π/2]


def decide_outcome(angle1: float, angle2: float, r: int) -> str:
    """Return the collision outcome label.

    Parameters
    ----------
    angle1, angle2 : float
        Angles in [0, 2π] of the incoming MT (1) and barrier MT (2).
    r : int (0 or 1)
        Random bit used to break the tie between catastrophe and crossover
        when the incident angle is large.

    Returns
    -------
    str : one of 'zipper+', 'zipper-', 'cross', 'catas'.
    """
    if incident_angle(angle1, angle2) <= TH_CRIT:
        # zipper sign = whichever alignment direction is closer to incoming
        same_dir_dist = circ_dist(angle1, angle2 % (2 * pi))
        anti_dir_dist = circ_dist(angle1, (angle2 + pi) % (2 * pi))
        return 'zipper+' if same_dir_dist <= anti_dir_dist else 'zipper-'
    return 'catas' if r == 0 else 'cross'
