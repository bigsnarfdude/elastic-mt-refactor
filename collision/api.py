"""Backward-compatible ``zip_cat`` built from the refactored pieces.

This function has the same signature and same 5-tuple return as Tim's
original ``zippering.zip_cat``. It exists so that ``sim_algs_fixed_region``
can adopt the refactor with zero change to its call sites.

The refactored ablation pattern depends on **late binding**: this module
imports the *modules*, not the *functions* by name. That way, a runtime
monkey-patch like

    >>> import collision.decision
    >>> collision.decision.decide_outcome = my_alternative

actually takes effect, because ``zip_cat_clean`` looks up
``decision.decide_outcome`` at call time, not at import time.

If you change ``from . import decision`` to ``from .decision import
decide_outcome``, the patching surface silently breaks. The unit test
``tests/test_patch_surface.py`` enforces this invariant.
"""
from . import decision   # NOTE: module import (late binding) — see docstring
from . import geometry   # NOTE: module import (late binding) — see docstring


def zip_cat_clean(angle1, angle2, pt, pt_prev, r,
                  no_bdl_id=False, d=0.0, decision_fn=None):
    """Refactored zip_cat — same return contract as the original.

    Parameters
    ----------
    angle1, angle2 : float
        Angles of the incoming MT (1) and barrier MT (2), in [0, 2π].
    pt : list[float, float]
        Point of collision.
    pt_prev : list[float, float]
        Previous vertex of incoming MT. (Unused here — preserved for
        signature compatibility with the original.)
    r : int (0 or 1)
        Random bit, breaks the catas/cross tie at large incident angle.
    no_bdl_id : bool, optional
        Whether the simulator is in 'no bundle ID' mode. If True, zipping
        MTs step back from the collision point. Default reads from
        ``parameters.no_bdl_id``.
    d : float, optional
        Step-back distance scale. Default reads from ``parameters.dr``.
    decision_fn : callable, optional
        Ablation hook. A function ``(angle1, angle2, r) -> label`` used in
        place of ``decision.decide_outcome`` to override the collision
        outcome. Default ``None`` falls back to ``decision.decide_outcome``
        (so the un-ablated path is bit-for-bit unchanged). Threading the
        decision through as an argument — rather than monkey-patching the
        global ``decision.decide_outcome`` — lets the simulator ablate the
        *real-collision* site (sim_algs:1023) WITHOUT touching the
        branch-nucleation geometry calls (sim_algs:1379, 2020), which must
        keep using the real rule. See SPRINT_S1_rule_ablation.md (Gotcha A).

    Returns
    -------
    (new_angle, new_pt, outcome, col_pt, error)
        Matches the original ``zip_cat`` 5-tuple contract.
    """
    # Late-bound lookups — monkey-patching the module attribute Just Works.
    # decision_fn (when given) overrides only this call; None => real rule.
    outcome = (decision_fn or decision.decide_outcome)(angle1, angle2, r)

    if outcome in ('zipper+', 'zipper-'):
        # Pass the outcome label to geometry so the new_angle is consistent
        # with the chosen direction. This matters when decide_outcome has
        # been monkey-patched to force a label different from what the
        # natural (closer-alignment) geometry would have picked. See
        # tests/test_patch_label_geometry_consistency.py for the contract.
        new_angle, _ = geometry.zipper_geometry(angle1, angle2, outcome=outcome)
        if no_bdl_id:
            dx, dy = geometry.step_back_offset(
                angle1, decision.incident_angle(angle1, angle2), d
            )
            new_pt = [pt[0] + dx, pt[1] + dy]
            col_pt = [pt[0] + dx, pt[1] + dy]
        else:
            new_pt = [pt[0], pt[1]]
            col_pt = [pt[0], pt[1]]
        return new_angle, new_pt, outcome, col_pt, None

    # cross / catas: incoming MT unchanged
    return angle1, [pt[0], pt[1]], outcome, [pt[0], pt[1]], None
