#!/usr/bin/env python3
"""Plot the multi-seed sweep results against the n=1 baseline.

Three-panel comparison:
  - Left:  S₂ trajectory by condition, n=10 with seed-spread shaded
  - Mid:   MT density vs S₂ scatter, showing condition clustering
  - Right: SAE feature preferred-angle distribution (n=10 vs n=1)

Run after the multi-seed sweep + retraining completes.
"""
import argparse, pickle, sys, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import pi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from plotting import s2 as compute_s2


def load_trajectory_from_pickles(seed_dir, seed):
    """Return (times, s2_vals, density_vals) for one (cond, seed) directory."""
    s2_traj, time_traj, density_traj = [], [], []
    pkls = sorted(seed_dir.glob(f'orderp_seed{seed}_*.pickle'))
    for pkl in pkls:
        try:
            order, order_t = pickle.load(open(pkl, 'rb'))
            for t_idx, region_list in enumerate(order):
                ws2, total_l = 0, 0
                for geom in region_list:
                    if geom and len(geom) >= 3 and geom[2] > 0:
                        try:
                            sv, _ = compute_s2(geom)
                            ws2 += sv * geom[2]; total_l += geom[2]
                        except: pass
                if total_l > 0:
                    s2_traj.append(ws2 / total_l)
                    density_traj.append(total_l)
                    if t_idx < len(order_t):
                        time_traj.append(float(order_t[t_idx]))
        except: pass
    return time_traj, s2_traj, density_traj


def collect_all(data_dir, t_cap=2.0):
    """Find all condition_seed*/ subdirs, return dict[condition] -> list of (times, s2, density)."""
    results = {}
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or '_seed' not in d.name:
            continue
        cond, sd = d.name.rsplit('_seed', 1)
        seed = int(sd)
        t, s, dens = load_trajectory_from_pickles(d, seed)
        if not t: continue
        # cap at t_cap
        idx = next((i for i, tt in enumerate(t) if tt > t_cap), len(t))
        if idx == 0: continue
        results.setdefault(cond, []).append((t[:idx], s[:idx], dens[:idx], seed))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir', type=Path)
    ap.add_argument('--out', type=Path, default=Path('docs/multiseed_results.png'))
    ap.add_argument('--t_cap', type=float, default=1.0)
    args = ap.parse_args()

    results = collect_all(args.data_dir, t_cap=args.t_cap)
    if not results:
        print("No results found.")
        return

    order_c = ['baseline', 'no_zipper', 'always_catas']
    colors = {'baseline': '#2a9d3f', 'no_zipper': '#e67e22', 'always_catas': '#c0392b'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={'width_ratios': [2, 1]})

    # Panel 1: trajectory with seed spread
    summary = {}
    for c in order_c:
        rs = results.get(c, [])
        if not rs: continue
        for (t, s, _, _) in rs:
            ax1.plot(t, s, color=colors[c], alpha=0.18, lw=0.8)
        min_len = min(len(r[1]) for r in rs)
        if min_len < 2: continue
        all_s = np.array([r[1][:min_len] for r in rs])
        all_t = rs[0][0][:min_len]
        mean = all_s.mean(axis=0)
        std = all_s.std(axis=0)
        ax1.plot(all_t, mean, color=colors[c], lw=3, label=f'{c} (n={len(rs)})')
        ax1.fill_between(all_t, mean - std, mean + std, color=colors[c], alpha=0.15)
        summary[c] = (mean[-1], std[-1], len(rs))

    ax1.set_xlabel('Sim time (hours)', fontsize=12)
    ax1.set_ylabel('S₂ (length-weighted)', fontsize=12)
    ax1.set_title(f'Multi-seed S₂ trajectories — {sum(len(r) for r in results.values())} runs total',
                  fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, args.t_cap)

    # Panel 2: density at endpoint, with seed spread
    xs, dens_means, dens_stds, cols, labels = [], [], [], [], []
    for c in order_c:
        rs = results.get(c, [])
        if not rs: continue
        finals = [r[2][-1] for r in rs if r[2]]
        if not finals: continue
        xs.append(len(xs))
        dens_means.append(np.mean(finals))
        dens_stds.append(np.std(finals))
        cols.append(colors[c])
        labels.append(f'{c}\nn={len(finals)}')
        # also overlay individual points
    ax2.bar(xs, dens_means, yerr=dens_stds, color=cols, alpha=0.85,
            capsize=8, edgecolor='#222', linewidth=1.5)
    for i, c in enumerate([cn for cn in order_c if cn in results]):
        finals = [r[2][-1] for r in results[c] if r[2]]
        ax2.scatter([i] * len(finals), finals, color='#222', zorder=3, s=18, alpha=0.5)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel(f'Total MT length at t={args.t_cap}', fontsize=11)
    ax2.set_title(f'MT density at t={args.t_cap}', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f"saved {args.out}")

    print(f"\n=== Summary (final S₂ at t={args.t_cap}) ===")
    for c, (m, s, n) in summary.items():
        print(f"  {c:18s} n={n:2d}  S₂ = {m:.4f} ± {s:.4f}")


if __name__ == '__main__':
    main()
