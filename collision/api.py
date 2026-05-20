"""Backward-compatible ``zip_cat`` built from the refactored pieces.

This function has the same signature and same 5-tuple return as Tim's
original ``zippering.zip_cat``. It exists so that ``sim_algs_fixed_region``
can adopt the refactor with zero change to its call sites.

Once the refactor lands, ``zippering.zip_cat`` becomes a one-line forward
to this function.
"""
from .decision import decide_outcome, incident_angle
from .geometry import zipper_geometry, step_back_offset


def zip_cat_clean(angle1, angle2, pt, pt_prev, r,
                  no_bdl_id=False, d=0.0):
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

    Returns
    -------
    (new_angle, new_pt, outcome, col_pt, error)
        Matches the original ``zip_cat`` 5-tuple contract.
    """
    outcome = decide_outcome(angle1, angle2, r)

    if outcome in ('zipper+', 'zipper-'):
        new_angle, label = zipper_geometry(angle1, angle2)
        # The label from zipper_geometry is the source of truth.
        # decide_outcome agrees with it by construction.
        assert label == outcome, (
            f"internal inconsistency: decision said {outcome}, "
            f"geometry said {label}"
        )
        if no_bdl_id:
            dx, dy = step_back_offset(angle1, incident_angle(angle1, angle2), d)
            new_pt = [pt[0] + dx, pt[1] + dy]
            col_pt = [pt[0] + dx, pt[1] + dy]
        else:
            new_pt = [pt[0], pt[1]]
            col_pt = [pt[0], pt[1]]
        return new_angle, new_pt, outcome, col_pt, None

    # cross / catas: incoming MT unchanged
    return angle1, [pt[0], pt[1]], outcome, [pt[0], pt[1]], None
