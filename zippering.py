#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 23:09:10 2021
@author: tim

Refactored 2026-05-20: ``zip_cat`` is now a thin wrapper over the split-out
``collision/`` modules. The original 300-line angle-quadrant case tree has
been replaced with three small composable functions:

    collision.decision.decide_outcome   — pure decision (zip/cross/catas)
    collision.geometry.zipper_geometry  — pure post-collision geometry
    collision.api.zip_cat_clean         — backward-compatible 5-tuple wrapper

Behavioral equivalence with the original verified by ``tests/test_equivalence.py``
across 1000 random inputs (outcome, new_angle, new_pt, col_pt all match
within numerical precision on the default ``no_bdl_id=False`` path).

The old 300-line implementation is preserved as ``_zip_cat_original`` for
reference and as a regression target. Delete after a release.
"""
import numpy as np
from math import sin, cos, pi
from comparison_fns import dist
import sys
from parameters import no_bdl_id, dr
from collision.api import zip_cat_clean as _zip_cat_clean

d = dr
dr_tol = d / sin(.01)


def zip_cat(angle1, angle2, pt, pt_prev, r, decision_fn=None):
    """Determine collision outcome and post-collision geometry.

    Backward-compatible shim — same signature and 5-tuple return as before.
    Forwards to the refactored ``collision.api.zip_cat_clean`` which preserves
    the original physics. See ``collision/`` for the split-out modules and
    ``tests/test_equivalence.py`` for the equivalence proof.

    Parameters
    ----------
    angle1 : angle of tip which collides
    angle2 : angle of barrier MT
    pt : point of intersection
    pt_prev : previous vertex of incoming MT
    r : 0 or 1 random number
    decision_fn : callable, optional
        Ablation hook ``(angle1, angle2, r) -> label`` forwarded to
        ``zip_cat_clean``. ``None`` (default) => the real rule, leaving the
        un-ablated path bit-for-bit unchanged. Only the real-collision call
        site (sim_algs:1023) passes this; the branch-nucleation geometry
        calls (1379, 2020) leave it None so they keep the real rule.

    Returns
    -------
    new_angle : entrainment angle, if not catastrophe/crossover
    new_pt    : starting pt of entrained MT segment
    resolve   : 'cross', 'zipper+', 'zipper-', or 'catas'
    col_pt    : end point of incoming MT
    error     : error message for special cases (always None — original
                 had commented-out error tracking; preserved for compatibility)
    """
    return _zip_cat_clean(angle1, angle2, pt, pt_prev, r,
                          no_bdl_id=no_bdl_id, d=d, decision_fn=decision_fn)


def _zip_cat_original(angle1, angle2, pt, pt_prev, r):
    """Tim's original 300-line zip_cat, preserved here for regression testing.

    Do not call this directly. Use ``zip_cat`` above, which is behaviorally
    equivalent. Delete this function after a stable release.
    """
    resolve = 'cross'
    th2, th1 = max(angle1, angle2), min(angle1, angle2)
    th_crit = 2 * pi / 9  # critical angle
    new_pt = [pt[0], pt[1]]
    col_pt = [pt[0], pt[1]]
    new_angle = angle1
    if th2 > 3*pi/2 and th1 < pi/2:
        a1 = 2*pi - th2
        a2 = th1
        b = a1 + a2
        if th2 == angle1:
            if b >= pi/2:
                b2 = pi - b
                if b2 <= th_crit:
                    resolve = 'zipper-'
                    if no_bdl_id:
                        dr_zip = d / sin(b2)
                        new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                        col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    new_angle = pi + th1
                else:
                    if r == 0:
                        resolve = 'catas'
            else:
                b2 = b
                if b2 <= th_crit:
                    resolve = 'zipper+'
                    if no_bdl_id:
                        dr_zip = d / sin(b2)
                        new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                        col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    new_angle = th1
                else:
                    if r == 0:
                        resolve = 'catas'
        else:
            if b >= pi/2:
                b2 = pi - b
                if b2 <= th_crit:
                    resolve = 'zipper-'
                    if no_bdl_id:
                        dr_zip = d / sin(b2)
                        new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                        col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    new_angle = th2 - pi
                else:
                    if r == 0:
                        resolve = 'catas'
            else:
                b2 = b
                if b2 <= th_crit:
                    resolve = 'zipper+'
                    if no_bdl_id:
                        dr_zip = d / sin(b2)
                        new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                        col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    new_angle = th2
                else:
                    if r == 0:
                        resolve = 'catas'
    elif th2 > pi/2 and th2 < pi and th1 < pi/2:
        a1 = pi - th2
        a2 = th1
        b = a1 + a2
        if th2 == angle1:
            if b >= pi/2:
                b2 = pi - b
                if b2 <= th_crit:
                    resolve = 'zipper+'
                    if no_bdl_id:
                        dr_zip = d / sin(b2)
                        new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                        col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    new_angle = th1
                else:
                    if r == 0:
                        resolve = 'catas'
            else:
                b2 = b
                if b2 <= th_crit:
                    resolve = 'zipper-'
                    if no_bdl_id:
                        dr_zip = d / sin(b2)
                        new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                        col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    new_angle = th1 + pi
                else:
                    if r == 0:
                        resolve = 'catas'
        else:
            if b >= pi/2:
                b2 = pi - b
                if b2 <= th_crit:
                    resolve = 'zipper+'
                    if no_bdl_id:
                        dr_zip = d / sin(b2)
                        new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                        col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    new_angle = th2
                else:
                    if r == 0:
                        resolve = 'catas'
            else:
                b2 = b
                if b2 <= th_crit:
                    resolve = 'zipper-'
                    if no_bdl_id:
                        dr_zip = d / sin(b2)
                        new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                        col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    new_angle = th2 + pi
                else:
                    if r == 0:
                        resolve = 'catas'
    elif th2 > 3*pi/2 and th1 < 3*pi/2 and th1 > pi:
        a1 = th1 - pi
        a2 = 2*pi - th2
        b = a1 + a2
        if th2 == angle1:
            if b >= pi/2:
                b2 = pi - b
                if b2 <= th_crit:
                    resolve = 'zipper+'
                    if no_bdl_id:
                        dr_zip = d / sin(b2)
                        new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                        col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    new_angle = th1
                else:
                    if r == 0:
                        resolve = 'catas'
            else:
                b2 = b
                if b2 <= th_crit:
                    resolve = 'zipper-'
                    if no_bdl_id:
                        dr_zip = d / sin(b2)
                        new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                        col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    new_angle = th1 - pi
                else:
                    if r == 0:
                        resolve = 'catas'
        else:
            if b >= pi/2:
                b2 = pi - b
                if b2 <= th_crit:
                    resolve = 'zipper+'
                    if no_bdl_id:
                        dr_zip = d / sin(b2)
                        new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                        col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    new_angle = th2
                else:
                    if r == 0:
                        resolve = 'catas'
            else:
                b2 = b
                if b2 <= th_crit:
                    resolve = 'zipper-'
                    if no_bdl_id:
                        dr_zip = d / sin(b2)
                        new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                        col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    new_angle = th2 - pi
                else:
                    if r == 0:
                        resolve = 'catas'
    elif th2 > pi and th2 < 3*pi/2 and th1 < pi and th1 > pi/2:
        a1 = th2 - pi
        a2 = pi - th1
        b = a1 + a2
        if th2 == angle1:
            if b >= pi/2:
                b2 = pi - b
                if b2 <= th_crit:
                    resolve = 'zipper-'
                    if no_bdl_id:
                        dr_zip = d / sin(b2)
                        new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                        col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    new_angle = th1 + pi
                else:
                    if r == 0:
                        resolve = 'catas'
            else:
                b2 = b
                if b2 <= th_crit:
                    resolve = 'zipper+'
                    if no_bdl_id:
                        dr_zip = d / sin(b2)
                        new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                        col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    new_angle = th1
                else:
                    if r == 0:
                        resolve = 'catas'
        else:
            if b >= pi/2:
                b2 = pi - b
                if b2 <= th_crit:
                    resolve = 'zipper-'
                    if no_bdl_id:
                        dr_zip = d / sin(b2)
                        new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                        col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    new_angle = th2 - pi
                else:
                    if r == 0:
                        resolve = 'catas'
            else:
                b2 = b
                if b2 <= th_crit:
                    resolve = 'zipper+'
                    if no_bdl_id:
                        dr_zip = d / sin(b2)
                        new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                        col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    new_angle = th2
                else:
                    if r == 0:
                        resolve = 'catas'
    elif th2 < pi/2 and th1 < pi/2:
        b = th2 - th1
        if th2 == angle1:
            if b <= th_crit:
                resolve = 'zipper+'
                if no_bdl_id:
                    dr_zip = d / sin(b)
                    new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                new_angle = th1
            else:
                if r == 0:
                    resolve = 'catas'
        else:
            if b <= th_crit:
                resolve = 'zipper+'
                if no_bdl_id:
                    dr_zip = d / sin(b)
                    new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                new_angle = th2
            else:
                if r == 0:
                    resolve = 'catas'
    elif th2 > pi and th2 < 3*pi/2 and th1 < pi/2 and (th2-pi) > th1:
        a1 = th2 - pi
        a2 = th1
        b = a1 - a2
        if th2 == angle1:
            if b <= th_crit:
                resolve = 'zipper-'
                if no_bdl_id:
                    dr_zip = d / sin(b)
                    new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                new_angle = th1 + pi
            else:
                if r == 0:
                    resolve = 'catas'
        else:
            if b <= th_crit:
                resolve = 'zipper-'
                if no_bdl_id:
                    dr_zip = d / sin(b)
                    new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                new_angle = th2 - pi
            else:
                if r == 0:
                    resolve = 'catas'
    elif th2 > pi and th2 < 3*pi/2 and th1 < pi/2 and (th2-pi) < th1:
        a1 = th1
        a2 = th2 - pi
        b = a1 - a2
        if th2 == angle1:
            if b <= th_crit:
                resolve = 'zipper-'
                if no_bdl_id:
                    dr_zip = d / sin(b)
                    new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                new_angle = th1 + pi
            else:
                if r == 0:
                    resolve = 'catas'
        else:
            if b <= th_crit:
                resolve = 'zipper-'
                if no_bdl_id:
                    dr_zip = d / sin(b)
                    new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                new_angle = th2 - pi
            else:
                if r == 0:
                    resolve = 'catas'
    elif th2 > pi and th2 < 3*pi/2 and th1 > pi and th1 < 3*pi/2:
        b = th2 - th1
        if th2 == angle1:
            if b <= th_crit:
                resolve = 'zipper+'
                if no_bdl_id:
                    dr_zip = d / sin(b)
                    new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                new_angle = th1
            else:
                if r == 0:
                    resolve = 'catas'
        else:
            if b <= th_crit:
                resolve = 'zipper+'
                if no_bdl_id:
                    dr_zip = d / sin(b)
                    new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                new_angle = th2
            else:
                if r == 0:
                    resolve = 'catas'
    elif th2 > pi/2 and th2 < pi and th1 > pi/2 and th1 < pi:
        b = th2 - th1
        if th2 == angle1:
            if b <= th_crit:
                resolve = 'zipper+'
                if no_bdl_id:
                    dr_zip = d / sin(b)
                    new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                new_angle = th1
            else:
                if r == 0:
                    resolve = 'catas'
        else:
            if b <= th_crit:
                resolve = 'zipper+'
                if no_bdl_id:
                    dr_zip = d / sin(b)
                    new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                new_angle = th2
            else:
                if r == 0:
                    resolve = 'catas'
    elif th2 > 3*pi/2 and th1 > pi/2 and th1 < pi and (th1+pi) > th2:
        a1 = th1 + pi
        a2 = th2
        b = a1 - a2
        if th2 == angle1:
            if b <= th_crit:
                resolve = 'zipper-'
                if no_bdl_id:
                    dr_zip = d / sin(b)
                    new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                new_angle = th1 + pi
            else:
                if r == 0:
                    resolve = 'catas'
        else:
            if b <= th_crit:
                resolve = 'zipper-'
                if no_bdl_id:
                    dr_zip = d / sin(b)
                    new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                new_angle = th2 - pi
            else:
                if r == 0:
                    resolve = 'catas'
    elif th2 > 3*pi/2 and th1 > pi/2 and th1 < pi and (th1+pi) < th2:
        a1 = th2
        a2 = th1 + pi
        b = a1 - a2
        if th2 == angle1:
            if b <= th_crit:
                resolve = 'zipper-'
                if no_bdl_id:
                    dr_zip = d / sin(b)
                    new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                new_angle = th1 + pi
            else:
                if r == 0:
                    resolve = 'catas'
        else:
            if b <= th_crit:
                resolve = 'zipper-'
                if no_bdl_id:
                    dr_zip = d / sin(b)
                    new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                new_angle = th2 - pi
            else:
                if r == 0:
                    resolve = 'catas'
    elif th2 > 3*pi/2 and th1 > 3*pi/2:
        a1 = th2
        a2 = th1
        b = a1 - a2
        if th2 == angle1:
            if b <= th_crit:
                resolve = 'zipper+'
                if no_bdl_id:
                    dr_zip = d / sin(b)
                    new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                new_angle = th1
            else:
                if r == 0:
                    resolve = 'catas'
        else:
            if b <= th_crit:
                resolve = 'zipper+'
                if no_bdl_id:
                    dr_zip = d / sin(b)
                    new_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                    col_pt = [pt[0] - dr_zip*cos(angle1), pt[1] - dr_zip*sin(angle1)]
                new_angle = th2
            else:
                if r == 0:
                    resolve = 'catas'
    error = None
    return new_angle, new_pt, resolve, col_pt, error
