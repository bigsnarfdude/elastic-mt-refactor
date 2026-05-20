"""Unit tests for the individual refactored functions.

Most of our coverage runs through ``zip_cat_clean`` — these tests pin down
each piece in isolation, with explicit input/output pairs. If one of these
fails, you know which sub-function broke.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pytest
import numpy as np
from math import pi


# ─────────────────────────────────────────────────────────────────────────────
# _helpers.circ_dist — shortest angular distance on the circle
# ─────────────────────────────────────────────────────────────────────────────

class TestCircDist:
    @staticmethod
    def cdist():
        from collision._helpers import circ_dist
        return circ_dist

    def test_zero_distance(self):
        cdist = self.cdist()
        assert cdist(0, 0) == 0
        assert cdist(1.5, 1.5) == 0
        assert cdist(pi, pi) == 0

    def test_wrap_around(self):
        """2π wraps to 0."""
        cdist = self.cdist()
        assert cdist(0, 2 * pi) < 1e-12
        assert cdist(0.1, 2 * pi + 0.1) < 1e-12

    def test_max_distance_is_pi(self):
        """Two points exactly opposite have distance π."""
        cdist = self.cdist()
        assert abs(cdist(0, pi) - pi) < 1e-12
        assert abs(cdist(pi / 2, 3 * pi / 2) - pi) < 1e-12

    def test_short_way(self):
        """Distance picks the short arc, not the long one."""
        cdist = self.cdist()
        # 0.1 vs 2π - 0.1 → short distance is 0.2 (going through 0)
        assert abs(cdist(0.1, 2 * pi - 0.1) - 0.2) < 1e-12

    def test_symmetric(self):
        cdist = self.cdist()
        rng = np.random.default_rng(0)
        for _ in range(200):
            a, b = rng.uniform(0, 2*pi), rng.uniform(0, 2*pi)
            assert abs(cdist(a, b) - cdist(b, a)) < 1e-12

    def test_in_valid_range(self):
        """Result is always in [0, π]."""
        cdist = self.cdist()
        rng = np.random.default_rng(1)
        for _ in range(200):
            d = cdist(rng.uniform(-10, 10), rng.uniform(-10, 10))
            assert 0 <= d <= pi + 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# decision.incident_angle — acute rod-rod angle in [0, π/2]
# ─────────────────────────────────────────────────────────────────────────────

class TestIncidentAngle:
    @staticmethod
    def ia():
        from collision.decision import incident_angle
        return incident_angle

    def test_parallel_rods(self):
        """Identical angle → zero incident."""
        ia = self.ia()
        assert ia(0, 0) == 0
        assert ia(pi / 3, pi / 3) == 0
        assert ia(5.0, 5.0) == 0

    def test_antiparallel_is_zero_nematic(self):
        """Anti-parallel (θ vs θ+π) is nematically the same → incident = 0."""
        ia = self.ia()
        assert abs(ia(0, pi)) < 1e-12
        assert abs(ia(0.3, 0.3 + pi)) < 1e-12
        assert abs(ia(pi / 4, 5 * pi / 4)) < 1e-12

    def test_perpendicular(self):
        """90° apart → incident = π/2."""
        ia = self.ia()
        assert abs(ia(0, pi / 2) - pi / 2) < 1e-12
        assert abs(ia(0.1, 0.1 + pi / 2) - pi / 2) < 1e-12

    def test_45_degrees(self):
        ia = self.ia()
        assert abs(ia(0, pi / 4) - pi / 4) < 1e-12

    def test_in_valid_range(self):
        """Output is always in [0, π/2]."""
        ia = self.ia()
        rng = np.random.default_rng(2)
        for _ in range(500):
            a, b = rng.uniform(0, 2*pi), rng.uniform(0, 2*pi)
            v = ia(a, b)
            assert 0 <= v <= pi / 2 + 1e-12, f"out of range: ia({a},{b}) = {v}"

    def test_handles_out_of_range_input(self):
        """Negative or > 2π inputs still produce a sensible result."""
        ia = self.ia()
        # negative angles
        assert abs(ia(-0.1, -0.1) - 0) < 1e-12
        # > 2π
        assert abs(ia(3 * pi, 3 * pi) - 0) < 1e-12
        # mix
        assert ia(-1.0, 7.0) < pi / 2 + 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# decision.decide_outcome — pure decision function
# ─────────────────────────────────────────────────────────────────────────────

class TestDecideOutcome:
    @staticmethod
    def d():
        from collision.decision import decide_outcome
        return decide_outcome

    def test_returns_valid_label(self):
        """Output is always one of the 4 valid labels."""
        d = self.d()
        valid = {'zipper+', 'zipper-', 'cross', 'catas'}
        rng = np.random.default_rng(3)
        for _ in range(500):
            a, b = rng.uniform(0, 2*pi), rng.uniform(0, 2*pi)
            r = int(rng.integers(0, 2))
            assert d(a, b, r) in valid

    def test_zip_for_small_incident(self):
        """Below the threshold → must be zipper+ or zipper-."""
        d = self.d()
        # incident = 0.01, well below TH_CRIT = 2π/9
        assert d(0, 0.01, 0) in ('zipper+', 'zipper-')
        assert d(0, 0.01, 1) in ('zipper+', 'zipper-')

    def test_cross_or_catas_for_large_incident(self):
        """Above the threshold → must be cross or catas."""
        d = self.d()
        # incident = π/2, well above TH_CRIT
        assert d(0, pi / 2 - 0.01, 0) == 'catas'  # r=0 → catas
        assert d(0, pi / 2 - 0.01, 1) == 'cross'  # r=1 → cross

    def test_r_only_matters_for_large_angle(self):
        """At small incident, the r bit shouldn't matter — it's always zip."""
        d = self.d()
        for a in [0.0, 1.0, pi, 2.5]:
            for b_offset in [0.01, 0.1, 0.3]:
                b = a + b_offset
                # both r values should give the same (zip) outcome
                assert d(a, b, 0) == d(a, b, 1), (
                    f"r changed outcome at small incident: ({a},{b})"
                )

    def test_deterministic(self):
        """Same input always gives same output."""
        d = self.d()
        rng = np.random.default_rng(4)
        for _ in range(100):
            a, b = rng.uniform(0, 2*pi), rng.uniform(0, 2*pi)
            r = int(rng.integers(0, 2))
            assert d(a, b, r) == d(a, b, r)

    def test_zipper_sign_picks_closer_alignment(self):
        """When zip happens, the sign matches whichever is closer to angle1."""
        from collision._helpers import circ_dist
        d = self.d()
        rng = np.random.default_rng(5)
        for _ in range(200):
            a1 = rng.uniform(0, 2*pi)
            # force small incident
            a2 = a1 + rng.uniform(-0.3, 0.3)
            outcome = d(a1, a2, 0)
            if outcome not in ('zipper+', 'zipper-'):
                continue
            same = a2 % (2*pi)
            anti = (a2 + pi) % (2*pi)
            d_same = circ_dist(a1, same)
            d_anti = circ_dist(a1, anti)
            if outcome == 'zipper+':
                assert d_same <= d_anti + 1e-12
            else:  # zipper-
                assert d_anti <= d_same + 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# geometry.zipper_geometry — post-collision angle
# ─────────────────────────────────────────────────────────────────────────────

class TestZipperGeometry:
    @staticmethod
    def zg():
        from collision.geometry import zipper_geometry
        return zipper_geometry

    def test_natural_choice_picks_closer(self):
        zg = self.zg()
        # incoming at 0.1, barrier at 0.2 → same direction is closer
        new_angle, label = zg(0.1, 0.2)
        assert label == 'zipper+'
        assert abs(new_angle - 0.2) < 1e-12

    def test_natural_choice_picks_anti(self):
        zg = self.zg()
        # incoming at 0.1, barrier at π+0.05 → anti is closer (back to ~0.05)
        new_angle, label = zg(0.1, pi + 0.05)
        assert label == 'zipper-'
        assert abs(new_angle - 0.05) < 1e-12

    def test_forced_plus_returns_same_dir(self):
        zg = self.zg()
        new_angle, label = zg(0.1, pi - 0.05, outcome='zipper+')
        assert label == 'zipper+'
        # same-direction is angle2 mod 2π = π - 0.05
        assert abs(new_angle - (pi - 0.05)) < 1e-12

    def test_forced_minus_returns_anti_dir(self):
        zg = self.zg()
        new_angle, label = zg(0.1, 0.2, outcome='zipper-')
        assert label == 'zipper-'
        # anti-direction = (0.2 + π) mod 2π
        expected = (0.2 + pi) % (2 * pi)
        assert abs(new_angle - expected) < 1e-12

    def test_output_in_valid_range(self):
        """new_angle is always in [0, 2π)."""
        zg = self.zg()
        rng = np.random.default_rng(6)
        for _ in range(200):
            a1, a2 = rng.uniform(0, 2*pi), rng.uniform(0, 2*pi)
            for outcome in [None, 'zipper+', 'zipper-']:
                new_angle, _ = zg(a1, a2, outcome=outcome)
                assert 0 <= new_angle < 2 * pi + 1e-12, (
                    f"out of range: zg({a1},{a2},{outcome}) = {new_angle}"
                )

    def test_label_matches_outcome_when_forced(self):
        """When outcome is explicit, the label always equals the outcome."""
        zg = self.zg()
        rng = np.random.default_rng(7)
        for _ in range(50):
            a1, a2 = rng.uniform(0, 2*pi), rng.uniform(0, 2*pi)
            _, label_plus = zg(a1, a2, outcome='zipper+')
            _, label_minus = zg(a1, a2, outcome='zipper-')
            assert label_plus == 'zipper+'
            assert label_minus == 'zipper-'


# ─────────────────────────────────────────────────────────────────────────────
# geometry.step_back_offset — backward offset along incoming direction
# ─────────────────────────────────────────────────────────────────────────────

class TestStepBackOffset:
    @staticmethod
    def sb():
        from collision.geometry import step_back_offset
        return sb

    def test_zero_at_tiny_incident(self):
        """Avoid div-by-zero when incident is essentially zero."""
        from collision.geometry import step_back_offset as sb
        dx, dy = sb(0.5, 1e-13, 0.01)
        assert dx == 0.0 and dy == 0.0

    def test_perpendicular_step(self):
        """At incident = π/2, sin = 1, so step = d directly."""
        from collision.geometry import step_back_offset as sb
        # angle1 = 0 means step in -x direction
        dx, dy = sb(0.0, pi / 2, 0.05)
        assert abs(dx - (-0.05)) < 1e-12
        assert abs(dy - 0.0) < 1e-12

    def test_step_along_incoming_direction(self):
        """Step is always anti-parallel to angle1 (backward along incoming)."""
        from collision.geometry import step_back_offset as sb
        from math import cos, sin
        rng = np.random.default_rng(8)
        for _ in range(50):
            a1 = rng.uniform(0, 2 * pi)
            incident = rng.uniform(0.01, pi / 2)
            d = rng.uniform(0.001, 0.1)
            dx, dy = sb(a1, incident, d)
            step = d / sin(incident)
            assert abs(dx - (-step * cos(a1))) < 1e-12
            assert abs(dy - (-step * sin(a1))) < 1e-12

    def test_step_scales_with_d(self):
        """Doubling d doubles the step."""
        from collision.geometry import step_back_offset as sb
        dx1, dy1 = sb(0.5, 0.3, 0.01)
        dx2, dy2 = sb(0.5, 0.3, 0.02)
        assert abs(dx2 - 2*dx1) < 1e-12
        assert abs(dy2 - 2*dy1) < 1e-12

    def test_step_inversely_with_sin(self):
        """Step grows as 1/sin(incident) — small incident → large step."""
        from collision.geometry import step_back_offset as sb
        # At very small incident, step should be very large
        _, _ = sb(0.0, 0.001, 0.01)
        from math import sin
        # expected magnitude
        expected = 0.01 / sin(0.001)
        dx, _ = sb(0.0, 0.001, 0.01)
        assert abs(dx) > 0.5  # should be huge


# ─────────────────────────────────────────────────────────────────────────────
# Cross-module: incident_angle and circ_dist should be consistent
# ─────────────────────────────────────────────────────────────────────────────

def test_incident_angle_via_circ_dist():
    """incident_angle(a1, a2) = min(circ_dist(a1, a2), π - circ_dist(a1, a2))."""
    from collision.decision import incident_angle
    from collision._helpers import circ_dist
    rng = np.random.default_rng(9)
    for _ in range(200):
        a, b = rng.uniform(0, 2*pi), rng.uniform(0, 2*pi)
        ia = incident_angle(a, b)
        cd = circ_dist(a, b)
        expected = min(cd, pi - cd)
        assert abs(ia - expected) < 1e-12


if __name__ == '__main__':
    import pytest as _pt
    sys.exit(_pt.main([__file__, '-v']))
