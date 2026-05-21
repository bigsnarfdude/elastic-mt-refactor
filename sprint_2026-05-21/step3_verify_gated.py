#!/usr/bin/env python3
"""Sprint Step 3 — drop the label-100% MLP into the gated sim, verify S₂ match.

Three runs, all seed 42, 1 sim-hour:
  A) baseline (Tim's rule) — reference S₂
  B) gated MLP v2 — should match baseline S₂ exactly (the gating path is
     deterministic when MLP wants zipper; only cross/catas come from MLP)
  C) full MLP v2 (no gate) — does retraining to label-100% prevent the crash?

Outputs:
  sprint_2026-05-21/step3_metrics.json
"""
import sys, os, time, json, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

OUT = REPO / "sprint_2026-05-21" / "step3_runs"
OUT.mkdir(parents=True, exist_ok=True)


def run_condition(label, decide_outcome_fn, seed=42, hours=1):
    """Run one condition by monkey-patching decide_outcome before importing the sim."""
    # Force a clean reimport of the simulator each time, so the patched
    # decide_outcome is bound fresh.
    for mod in list(sys.modules.keys()):
        if mod.startswith(('collision', 'sim_algs_fixed_region', 'plotting',
                           'zippering', 'comparison_fns')):
            del sys.modules[mod]

    import collision.decision
    if decide_outcome_fn is not None:
        collision.decision.decide_outcome = decide_outcome_fn

    import sim_algs_fixed_region as sa
    import pickle, glob
    from plotting import s2 as compute_s2

    work = OUT / f"{label}_seed{seed}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "states").mkdir(exist_ok=True)

    print(f"\n[{label}] starting seed={seed} hours={hours}")
    t0 = time.time()
    status, err = "ok", None
    try:
        sa.simulate(seed, hours, str(work) + '/', False, False, troubleshoot=False)
    except Exception:
        err = traceback.format_exc()[:600]
        status = "crash"
    wall = time.time() - t0
    print(f"[{label}] {status} wall={wall:.0f}s")

    # Extract final S2 from the last orderp pickle, if any
    s2_traj, density = [], []
    for pkl in sorted(work.glob(f"orderp_seed{seed}_*.pickle")):
        try:
            order, _ = pickle.load(open(pkl, 'rb'))
            for region_list in order:
                ws2, total_l = 0.0, 0.0
                for geom in region_list:
                    if geom and len(geom) >= 3 and geom[2] > 0:
                        try:
                            sv, _ = compute_s2(geom)
                            ws2 += sv * geom[2]
                            total_l += geom[2]
                        except Exception:
                            pass
                if total_l > 0:
                    s2_traj.append(float(ws2 / total_l))
                    density.append(float(total_l))
        except Exception:
            pass

    return dict(
        label=label, seed=seed, hours=hours, wall_s=wall,
        status=status, error=err,
        s2_final=s2_traj[-1] if s2_traj else None,
        density_final=density[-1] if density else None,
        s2_trajectory=s2_traj,
        density_trajectory=density,
    )


def main():
    out_dir = REPO / "sprint_2026-05-21"
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("STEP 3 — Verify label-100% MLP in the simulator")
    print("=" * 60)

    results = {}

    # A) Baseline (Tim's rule, untouched)
    results['baseline'] = run_condition('baseline', None)

    # Load v2 MLP
    print("\nLoading models/learned_zipcat_v2.npz ...")
    sys.path.insert(0, str(REPO / "sprint_2026-05-21"))
    from step1_retrain_100 import MLP, encode, IDX_TO_OUTCOME
    import numpy as np
    npz = np.load(REPO / "models" / "learned_zipcat_v2.npz")
    mlp_v2 = MLP(hidden=npz['W1'].shape[1], seed=0)
    for k in mlp_v2._params:
        setattr(mlp_v2, k, npz[k])

    # B) Gated MLP v2 — same logic as gated_learned but with the new model
    from collision.decision import incident_angle, TH_CRIT
    from collision.geometry import zipper_geometry

    def gated_v2(angle1, angle2, r):
        if incident_angle(angle1, angle2) <= TH_CRIT:
            _, label = zipper_geometry(angle1, angle2, outcome=None)
            return label
        X = encode(angle1, angle2, r).reshape(1, -1)
        out_idx = int(mlp_v2.predict_label(X)[0])
        label = IDX_TO_OUTCOME[out_idx]
        if label in ('zipper+', 'zipper-'):
            return 'catas' if r == 0 else 'cross'
        return label
    gated_v2.__name__ = 'gated_v2'
    results['gated_v2'] = run_condition('gated_v2', gated_v2)

    # C) Full MLP (no gate) — does label-100% prevent the crash?
    def full_mlp_v2(angle1, angle2, r):
        X = encode(angle1, angle2, r).reshape(1, -1)
        out_idx = int(mlp_v2.predict_label(X)[0])
        return IDX_TO_OUTCOME[out_idx]
    full_mlp_v2.__name__ = 'full_mlp_v2'
    results['full_mlp_v2'] = run_condition('full_mlp_v2', full_mlp_v2)

    # Summary
    print("\n" + "=" * 60)
    print("STEP 3 SUMMARY")
    print("=" * 60)
    for label, r in results.items():
        s2 = r['s2_final']
        dens = r['density_final']
        s2_str = f"{s2:.4f}" if s2 is not None else "(crash)"
        dens_str = f"{dens:.1f}" if dens is not None else "(crash)"
        print(f"  {label:14s}  status={r['status']:6s}  S2={s2_str}  density={dens_str}  "
              f"wall={r['wall_s']:.0f}s")

    # Did gated_v2 match baseline?
    b = results['baseline']['s2_final']
    g = results['gated_v2']['s2_final']
    if b is not None and g is not None:
        match = abs(b - g) < 1e-6
        print(f"\n  gated_v2 matches baseline?  {'YES' if match else 'NO'}  "
              f"(|Δ| = {abs(b-g):.2e})")

    print(f"\n  full_mlp_v2 status: {results['full_mlp_v2']['status']}")

    with open(out_dir / "step3_metrics.json", "w") as f:
        # drop big trajectories for headline file
        slim = {}
        for k, v in results.items():
            slim[k] = {kk: vv for kk, vv in v.items() if kk not in ('s2_trajectory', 'density_trajectory')}
            slim[k]['n_snapshots'] = len(v['s2_trajectory'])
        json.dump(slim, f, indent=2)
    # full trajectories in a separate file
    with open(out_dir / "step3_trajectories.json", "w") as f:
        json.dump({k: dict(s2=v['s2_trajectory'], density=v['density_trajectory'])
                   for k, v in results.items()}, f, indent=2)
    print(f"\n  metrics → sprint_2026-05-21/step3_metrics.json")


if __name__ == "__main__":
    main()
