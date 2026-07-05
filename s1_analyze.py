#!/usr/bin/env python3
"""
Analyze result.json files from microtubule elasticity condition sweeps.
Reads 25 runs (5 conditions x 5 seeds), computes per-run metrics, aggregates by condition.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Configuration
BASE_DIR = Path("/Users/vincent/Desktop/me/microtubles/elastic-mt-refactor/s1_results")
CONDITIONS = ["baseline", "no_zipper", "zipper_to_cross", "no_induced_catas", "always_catas"]
SEEDS = list(range(1, 6))
OUTPUT_TSV = BASE_DIR / "ablation_matrix.tsv"
OUTPUT_JSON = BASE_DIR / "s2_global_trajectories.json"

def load_result(condition, seed):
    """Load a single result.json file."""
    path = BASE_DIR / f"{condition}_seed{seed}" / "result.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def compute_s2_slope_0_2h(times, s2_region_lw):
    """
    Compute slope of s2_region_lw vs times, restricted to times <= 2.0.
    Returns degree-1 polyfit coefficient (slope).
    """
    times_arr = np.array(times)
    s2_arr = np.array(s2_region_lw)
    mask = times_arr <= 2.0
    if np.sum(mask) < 2:
        return np.nan
    times_subset = times_arr[mask]
    s2_subset = s2_arr[mask]
    coeffs = np.polyfit(times_subset, s2_subset, 1)
    return coeffs[0]  # Return slope (degree-1 coefficient)

def compute_time_to_0_8(times, s2_region_lw):
    """
    Find first sim-hour at which s2_region_lw >= 0.8.
    Uses linear interpolation between bracketing points.
    Returns float (interpolated time) or string "never".
    Special case: if starts >= 0.8, return times[0].
    """
    times_arr = np.array(times)
    s2_arr = np.array(s2_region_lw)

    # Check if starts >= 0.8
    if s2_arr[0] >= 0.8:
        return times_arr[0]

    # Find first index where value crosses >= 0.8
    for i in range(len(s2_arr) - 1):
        if s2_arr[i] < 0.8 and s2_arr[i + 1] >= 0.8:
            # Linear interpolation
            t0, t1 = times_arr[i], times_arr[i + 1]
            v0, v1 = s2_arr[i], s2_arr[i + 1]
            t_interp = t0 + (0.8 - v0) / (v1 - v0) * (t1 - t0)
            return t_interp

    # Never reaches 0.8
    return "never"

def compute_metrics(result):
    """Compute all 5 metrics from a result dict."""
    if result is None:
        return None

    metrics = {
        "s2_lw_final": result["s2_final"],
        "s2_global_final": result["s2_global_final"],
        "density_final": result["density_final"],
        "s2_slope_0_2h": compute_s2_slope_0_2h(result["times"], result["s2_region_lw"]),
        "time_to_0_8": compute_time_to_0_8(result["times"], result["s2_region_lw"]),
    }
    return metrics

# Load all runs and compute metrics
all_runs = []
trajectories = {}

for condition in CONDITIONS:
    for seed in SEEDS:
        result = load_result(condition, seed)
        if result is None:
            print(f"Warning: {condition}_seed{seed} not found, skipping")
            continue

        metrics = compute_metrics(result)
        run_data = {
            "condition": condition,
            "seed": seed,
            **metrics
        }
        all_runs.append(run_data)

        # Store trajectory data
        key = f"{condition}_seed{seed}"
        trajectories[key] = {
            "times": result["times"],
            "s2_global": result["s2_global"],
            "s2_region_lw": result["s2_region_lw"],
            "density": result["density"],
        }

# Write TSV output
with open(OUTPUT_TSV, "w") as f:
    f.write("condition\tseed\ts2_lw_final\ts2_global_final\tdensity_final\ts2_slope_0_2h\ttime_to_0_8\n")
    for run in all_runs:
        time_to_0_8_str = "never" if isinstance(run["time_to_0_8"], str) else f"{run['time_to_0_8']:.3f}"
        f.write(
            f"{run['condition']}\t{run['seed']}\t"
            f"{run['s2_lw_final']:.4f}\t{run['s2_global_final']:.4f}\t"
            f"{run['density_final']:.4f}\t{run['s2_slope_0_2h']:.4f}\t{time_to_0_8_str}\n"
        )

print(f"✓ Wrote {OUTPUT_TSV}")

# Write trajectory JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(trajectories, f, indent=2)

print(f"✓ Wrote {OUTPUT_JSON}")

# Aggregate by condition
aggregates = defaultdict(list)
for run in all_runs:
    condition = run["condition"]
    aggregates[condition].append(run)

print("\n" + "="*120)
print("AGGREGATE TABLE (mean ± population std over 5 seeds)")
print("="*120)

# Print header
header = "condition\ts2_lw_final\t\t\ts2_global_final\t\t\tdensity_final\t\t\ts2_slope_0_2h\t\t\ttime_to_0_8"
print(header)
print("-"*120)

for condition in CONDITIONS:
    runs = aggregates[condition]
    if not runs:
        continue

    # Extract numeric values (skip "never" for time_to_0_8)
    s2_lw_vals = [r["s2_lw_final"] for r in runs]
    s2_global_vals = [r["s2_global_final"] for r in runs]
    density_vals = [r["density_final"] for r in runs]
    s2_slope_vals = [r["s2_slope_0_2h"] for r in runs]
    time_to_08_vals = [r["time_to_0_8"] for r in runs if isinstance(r["time_to_0_8"], (int, float))]

    # Compute stats
    s2_lw_mean, s2_lw_std = np.mean(s2_lw_vals), np.std(s2_lw_vals, ddof=0)
    s2_global_mean, s2_global_std = np.mean(s2_global_vals), np.std(s2_global_vals, ddof=0)
    density_mean, density_std = np.mean(density_vals), np.std(density_vals, ddof=0)
    s2_slope_mean, s2_slope_std = np.mean(s2_slope_vals), np.std(s2_slope_vals, ddof=0)

    # Handle time_to_0_8
    never_count = len(runs) - len(time_to_08_vals)
    if time_to_08_vals:
        time_to_08_mean, time_to_08_std = np.mean(time_to_08_vals), np.std(time_to_08_vals, ddof=0)
        time_to_08_str = f"{time_to_08_mean:.3f}±{time_to_08_std:.3f}"
        if never_count > 0:
            time_to_08_str += f" ({never_count} never)"
    else:
        time_to_08_str = f"all never ({never_count})"

    # Print row
    print(
        f"{condition}\t"
        f"{s2_lw_mean:.4f}±{s2_lw_std:.4f}\t\t"
        f"{s2_global_mean:.4f}±{s2_global_std:.4f}\t\t"
        f"{density_mean:.4f}±{density_std:.4f}\t\t"
        f"{s2_slope_mean:.4f}±{s2_slope_std:.4f}\t\t"
        f"{time_to_08_str}"
    )

# Sanity gate
print("\n" + "="*120)
print("SANITY GATE")
print("="*120)

baseline_runs = aggregates["baseline"]
no_zipper_runs = aggregates["no_zipper"]

baseline_s2_lw_mean = np.mean([r["s2_lw_final"] for r in baseline_runs])
no_zipper_s2_lw_mean = np.mean([r["s2_lw_final"] for r in no_zipper_runs])

print(f"baseline s2_lw_final mean: {baseline_s2_lw_mean:.4f} (expect ~0.85)")
baseline_pass = "PASS" if 0.80 <= baseline_s2_lw_mean <= 0.90 else "FAIL"
print(f"  → {baseline_pass}\n")

print(f"no_zipper s2_lw_final mean: {no_zipper_s2_lw_mean:.4f} (expect ~0.57)")
no_zipper_pass = "PASS" if 0.50 <= no_zipper_s2_lw_mean <= 0.65 else "FAIL"
print(f"  → {no_zipper_pass}")

# Diagnostic: zipper_to_cross and baseline
print("\n" + "="*120)
print("TRAJECTORY DIAGNOSTICS")
print("="*120)

for condition in ["baseline", "zipper_to_cross"]:
    runs = aggregates[condition]
    print(f"\n{condition.upper()}:")
    print("-"*80)

    for run in runs:
        seed = run["seed"]
        key = f"{condition}_seed{seed}"
        traj = trajectories[key]
        times = np.array(traj["times"])
        s2_global = np.array(traj["s2_global"])

        # Max single-step jump
        diffs = np.diff(s2_global)
        max_jump = np.max(np.abs(diffs))

        # Monotonicity
        neg_diffs = np.sum(diffs < 0)
        is_monotonic = "YES" if neg_diffs == 0 else f"NO ({neg_diffs} neg diffs)"

        # Values at key timepoints
        times_list = list(times)
        s2_global_list = list(s2_global)

        def get_nearest_value(target_time):
            idx = np.argmin(np.abs(times - target_time))
            return s2_global_list[idx]

        val_at_1h = get_nearest_value(1.0)
        val_at_3h = get_nearest_value(3.0)
        val_at_5h = get_nearest_value(5.0)
        val_at_10h = get_nearest_value(10.0)

        print(
            f"  seed {seed}: max_jump={max_jump:.4f}, monotonic={is_monotonic}, "
            f"@1h={val_at_1h:.4f}, @3h={val_at_3h:.4f}, @5h={val_at_5h:.4f}, @10h={val_at_10h:.4f}"
        )

print("\n" + "="*120)
print("Data loaded and analyzed successfully")
print("="*120)
