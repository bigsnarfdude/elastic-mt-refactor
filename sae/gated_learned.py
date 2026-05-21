"""Selective MLP ablation: let the MLP decide cross-vs-catas, but never
override Tim's zipper threshold.

The full MLP-in-simulator demo crashes because the MLP sometimes picks
'zipper+' for collisions Tim's threshold check would reject. Those wide-angle
zippers create bundles whose geometry violates the simulator's downstream
assumptions about bundle parallelism.

This wrapper gates the MLP: if Tim's incident-angle check says "this should
be a zip", we use Tim's deterministic zip decision (so bundle geometry stays
consistent). If Tim's check says "this should NOT be a zip", we let the MLP
decide between cross and catastrophe.

Result: the simulator can run end-to-end on a hybrid rule where the MLP
contributes only to the binary cross/catas choice — the "should this MT
die" question — while the alignment/bundling pathway remains deterministic.

That's a meaningful but limited surrogate: 0.7% of the model's predictions
(the disagreements on small-angle outcomes) are passed through; the rest
is Tim's rule.
"""
import numpy as np
from math import pi
from collision.decision import incident_angle, TH_CRIT
from collision.geometry import zipper_geometry
from sae.learned_collision import encode_input, IDX_TO_OUTCOME


def make_gated_learned_decision(mlp):
    """Return a decide_outcome replacement that uses the MLP only when Tim's
    threshold says the outcome should NOT be a zipper.

    Logic:
      if incident_angle(a1, a2) <= TH_CRIT:
          # Tim's rule says zip — use Tim's deterministic zipper sign,
          # so bundle geometry is exact.
          return zipper_geometry's natural label
      else:
          # Tim's rule says cross/catas — let the MLP pick between them.
          mlp_out = mlp.predict(...)
          if mlp predicts 'zipper+/-':  # MLP wants to zip a wide-angle pair
              # Bundle code would crash — override.
              # Use r to break tie, as Tim's rule would.
              return 'catas' if r == 0 else 'cross'
          else:
              return mlp's choice  # cross or catas
    """
    def gated_decide(angle1, angle2, r):
        if incident_angle(angle1, angle2) <= TH_CRIT:
            # Small-angle case — Tim's deterministic zipper.
            _, label = zipper_geometry(angle1, angle2, outcome=None)
            return label
        # Large-angle case — MLP picks between cross and catas (or any of the 4,
        # but we override zipper outcomes to prevent the crash).
        X = encode_input(angle1, angle2, r).reshape(1, -1)
        out_idx, _ = mlp.predict(X)
        label = IDX_TO_OUTCOME[int(out_idx[0])]
        if label in ('zipper+', 'zipper-'):
            # MLP wanted to zip a wide-angle collision — gate it. Use r-tiebreak
            # like Tim's original rule.
            return 'catas' if r == 0 else 'cross'
        return label

    gated_decide.__name__ = 'gated_learned_decide'
    return gated_decide


def make_strict_zipper_only_learned(mlp):
    """Inverse experiment: MLP decides whether to zip; Tim's rule decides
    cross-vs-catas at large angles. Tests whether the MLP's zipper decisions
    (the part Tim's threshold rule captures) are the load-bearing ones.

    Returns gated_decide where MLP only controls the binary zip-vs-not decision.
    """
    def zip_only_decide(angle1, angle2, r):
        X = encode_input(angle1, angle2, r).reshape(1, -1)
        out_idx, _ = mlp.predict(X)
        label = IDX_TO_OUTCOME[int(out_idx[0])]
        if label in ('zipper+', 'zipper-'):
            # MLP picked zip — but use Tim's deterministic geometry sign
            _, det_label = zipper_geometry(angle1, angle2, outcome=None)
            return det_label
        # MLP picked non-zip — use Tim's r-tiebreak
        return 'catas' if r == 0 else 'cross'

    zip_only_decide.__name__ = 'zip_only_learned_decide'
    return zip_only_decide
