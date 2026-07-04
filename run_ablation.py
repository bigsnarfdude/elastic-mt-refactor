#!/usr/bin/env python3
"""Sprint S1 — rule-ablation runner (clean collision-decision surface).

Runs ONE (condition, seed) of the rule-ablation matrix and writes result.json.
Parallelize tomorrow by launching one process per (condition, seed).

    python run_ablation.py <condition> <seed> <hours_idx>
    python run_ablation.py --selfcheck         # fast wiring check, no long run
    python run_ablation.py --list              # print conditions

Key difference from the v0.2 harness (run_one_v2_fix.py):
  - v0.2 monkey-patched the GLOBAL zippering.zip_cat with a wrapper that
    returned angle1 unchanged — so it DISCARDED real zipper bending AND hit all
    three zip_cat call sites (1023 real collision + 1379/2020 branch geometry).
  - This harness threads `decision_fn` through to the real-collision site ONLY
    (sim_algs.ABLATION, consumed at sim_algs:1023). Branch-nucleation geometry
    (1379/2020) keeps the real rule, and zipper geometry is preserved for every
    condition. See SPRINT_S1_rule_ablation.md (Gotcha A) and the isolation tests
    in tests/test_equivalence.py.

Config note (resolved 2026-06-19): `branch_nuc_off` is NOT in this matrix.
`branch_stuff` is an entrainment-sideness flag, not a nucleation switch; branched
(MT-templated) nucleation lives only in the LDD_bool=True path, which is OFF in
the benchmark config (parameters.LDD_bool=False => purely isotropic nucleation).
So there is no branched nucleation to ablate here, and branch_stuff=False crashes
under zippering (uninitialised path on a boundary-continuation MT). H3 is resolved
on those grounds, not by toggling branch_stuff.
"""
import sys, os, time, pickle, json, glob
from pathlib import Path
from math import sqrt, pi

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from collision.decision import decide_outcome as _orig, incident_angle, TH_CRIT

# ── The ablation conditions (collision-decision surface) ──────────────────────
# Each is a fn (angle1, angle2, r) -> outcome label. None == baseline (real rule).
# zip_cat_clean turns these labels into geometry, so real zipper bending is
# preserved wherever a zipper label is returned.
COLLISION_CONDITIONS = {
    # control — real rule, bit-for-bit baseline
    'baseline':         None,
    # H1 (coupled): remove zipper alignment AND add 50% death to shallow hits
    'no_zipper':        lambda a1, a2, r: ('catas' if r == 0 else 'cross'),
    # H1 (decoupled): remove zipper ALIGNMENT only — shallow -> cross (no death)
    'zipper_to_cross':  lambda a1, a2, r: ('cross' if incident_angle(a1, a2) <= TH_CRIT
                                           else _orig(a1, a2, r)),
    # H2: never collision-induced catastrophe (force r=1: shallow->zip, steep->cross)
    'no_induced_catas': lambda a1, a2, r: _orig(a1, a2, 1),
    # H2 extreme: every collision kills
    'always_catas':     lambda a1, a2, r: 'catas',
    # density ceiling: collisions do nothing
    'always_cross':     lambda a1, a2, r: 'cross',
}

# hours_idx -> sim_hours via parameters.time_snap (see simulate(): final_hr =
# time_snap[final_idx]). time_snap = [1,2,...,10, inf]; idx 9 == the full 10h run.
DEFAULT_HOURS_IDX = 9   # 10 sim-hours, the benchmark cut-off


# ── Metric extraction (validated digit-for-digit against the v0.2 on-disk run
# no_branched_nuc_no_zipper_seed42: density_final & s2_final reproduced exactly) ─
def _region_sums(geom):
    """Return (sum_len*cos2a, sum_len*sin2a, mt_len) for one region geom
    = (angles, leng, mt_len, theta) from plotting.order_hist."""
    ang = np.asarray(geom[0]); leng = np.asarray(geom[1]); mt_len = geom[2]
    if not mt_len or len(ang) == 0:
        return 0.0, 0.0, 0.0
    return (float(np.sum(leng * np.cos(2 * ang))),
            float(np.sum(leng * np.sin(2 * ang))), float(mt_len))


def _s2_from_sums(sc, ss, ml):
    return sqrt(sc * sc + ss * ss) / ml if ml else None


def extract_metrics(out_dir, seed):
    """Load orderp pickles in idx order and return per-snapshot trajectories.

    s2_region_lw   : length-weighted mean of per-region S2  (the v0.2 convention)
    s2_global      : S2 over all regions combined            (diagnostic; the gap
                     s2_region_lw - s2_global is the low-N estimator bias)
    density        : total scaled MT length summed over regions (v0.2 'density')
    """
    files = sorted(glob.glob(os.path.join(out_dir, f'orderp_seed{seed}_*.pickle')),
                   key=lambda f: int(f.split('_')[-1].replace('idx.pickle', '')))
    times, s2_lw, s2_glob, dens = [], [], [], []
    for f in files:
        order, order_t = pickle.load(open(f, 'rb'))
        for snap, t in zip(order, order_t):
            gsc = gss = gml = 0.0          # global accumulators
            wsum = lwsum = 0.0             # length-weighted per-region accumulators
            for geom in snap:
                if not geom:
                    continue
                sc, ss, ml = _region_sums(geom)
                if ml <= 0:
                    continue
                gsc += sc; gss += ss; gml += ml
                rs2 = _s2_from_sums(sc, ss, ml)
                if rs2 is not None:
                    lwsum += rs2 * ml; wsum += ml
            times.append(float(t))
            dens.append(gml)
            s2_lw.append(lwsum / wsum if wsum else None)
            s2_glob.append(_s2_from_sums(gsc, gss, gml))
    return dict(times=times, s2_region_lw=s2_lw, s2_global=s2_glob, density=dens)


def _config_snapshot():
    """Gotcha-B: dump the nucleation/collision config into every result.json so a
    later reader can tell exactly what physics produced these numbers."""
    import parameters as P
    import sim_algs_fixed_region as sa
    return dict(
        LDD_bool=P.LDD_bool,            # False => isotropic nucleation, no branched nuc
        branch_stuff=P.branch_stuff,    # entrainment sideness flag (NOT nucleation)
        sa_branch_stuff=sa.branch_stuff,
        accept_MT=P.accept_MT, plant=P.plant, cat_on=P.cat_on,
        no_bdl_id=P.no_bdl_id, tread_bool=P.tread_bool, deflect_on=P.deflect_on,
        r_c=P.r_c, r_n=P.r_n, R=P.R, final_hr=float(P.final_hr),
    )


def run_one(condition, seed, hours_idx, out_root=None):
    assert condition in COLLISION_CONDITIONS, f"unknown condition {condition!r}"
    out = Path(out_root or (Path.home() / 'mt_ablation')) / f'{condition}_seed{seed}'
    (out / 'states').mkdir(parents=True, exist_ok=True)

    import sim_algs_fixed_region as sa
    sa.ABLATION = COLLISION_CONDITIONS[condition]    # None == baseline

    cfg = _config_snapshot()
    print(f"[{condition} s{seed}] start idx={hours_idx} (final_hr={cfg['final_hr']}) "
          f"LDD={cfg['LDD_bool']} branch_stuff={cfg['branch_stuff']} "
          f"ABLATION={'None' if sa.ABLATION is None else condition}", flush=True)

    t0 = time.time()
    try:
        sa.simulate(seed, hours_idx, str(out) + '/', False, False, troubleshoot=False)
        status, err = 'ok', None
    except Exception as e:
        import traceback
        status, err = 'error', traceback.format_exc()
    wall = time.time() - t0

    metrics = {}
    try:
        metrics = extract_metrics(str(out), seed)
    except Exception as e:
        err = (err or '') + f"\n[metric extraction failed] {e}"

    s2lw = metrics.get('s2_region_lw') or []
    dens = metrics.get('density') or []
    result = dict(
        condition=condition, seed=seed, hours_idx=hours_idx, wall_s=wall,
        status=status, error=err, config=cfg,
        n_snapshots=len(metrics.get('times', [])),
        times=metrics.get('times', []),
        s2_region_lw=s2lw, s2_global=metrics.get('s2_global', []),
        density=dens,
        s2_final=s2lw[-1] if s2lw else None,
        s2_global_final=(metrics.get('s2_global') or [None])[-1],
        density_final=dens[-1] if dens else None,
    )
    with open(out / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[{condition} s{seed}] {status} wall={wall:.0f}s "
          f"S2(lw)={result['s2_final']} S2(global)={result['s2_global_final']} "
          f"dens={result['density_final']} n={result['n_snapshots']}", flush=True)
    return result


# ── Fast wiring self-check (no long run) ──────────────────────────────────────
def selfcheck(hours_idx=0):
    """Prove the site-1023 hook actually fires at runtime — fast.

    Runs ONE short troubleshoot sim (default idx=0 => 0.1 sim-hr) with a counting
    decision_fn installed as sim_algs.ABLATION, and asserts it was consulted >0
    times. That is the wiring fact the unit tests can't show: that the module
    global ABLATION is late-bound at the real-collision site. (The full-sim
    baseline path is already exercised whenever a real run runs; these sims are
    slow on a laptop — use nigel for production.)"""
    import sim_algs_fixed_region as sa
    tmp = Path('/tmp/mt_ablation_selfcheck')
    calls = {'n': 0}
    def counting(a1, a2, r):
        calls['n'] += 1
        return _orig(a1, a2, r)        # identical decision to baseline, just counted
    (tmp / 'count' / 'states').mkdir(parents=True, exist_ok=True)
    sa.ABLATION = counting
    sa.simulate(7, hours_idx, str(tmp / 'count') + '/', False, False, troubleshoot=True)
    sa.ABLATION = None
    print(f"  [selfcheck] ABLATION hook fired {calls['n']} times at site 1023", flush=True)
    assert calls['n'] > 0, "ABLATION never fired — the 1023 hook is not wired!"
    print("  [selfcheck] PASS — hook is live and late-bound", flush=True)


if __name__ == '__main__':
    if '--list' in sys.argv:
        print("conditions:", ", ".join(COLLISION_CONDITIONS))
    elif '--selfcheck' in sys.argv:
        selfcheck()
    elif len(sys.argv) >= 3:
        cond = sys.argv[1]
        seed = int(sys.argv[2])
        hidx = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_HOURS_IDX
        run_one(cond, seed, hidx)
    else:
        print(__doc__)
        print("conditions:", ", ".join(COLLISION_CONDITIONS))
        sys.exit(1)
