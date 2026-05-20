"""Shared internal helpers. Not part of the public ablation API."""
from math import pi


def circ_dist(a: float, b: float) -> float:
    """Shortest angular distance between a and b on the circle. In [0, π]."""
    d = abs(a - b) % (2 * pi)
    return min(d, 2 * pi - d)
