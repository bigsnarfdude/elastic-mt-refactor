# Tier 0 — classifier probe of the rule-from-trajectory question

**Date:** 2026-05-21
**Data:** 13 ablation trajectories from overnight nigel sweep (5 baseline, 5 no_zipper, 3 always_catas) + 4 OOD examples (baseline_isotropic, gated_v2, baseline_tree, baseline_mac)
**Goal:** is there any signal in "S₂(t) + density(t) → which collision rule?"

## TL;DR

- **In-distribution: trivially easy.** All classifiers (RF, GBM, KNN-k1, LogReg) hit 100% leave-one-out accuracy. First 3 timepoints are enough.
- **OOD: well-behaved.** Truncated/variant baselines all classify as baseline. Gated MLP also classifies as baseline at 0.94 confidence on apples-to-apples.
- **Decision boundary in trajectory-space:** baseline → no_zipper morph crosses 50/50 at α≈0.4 (i.e., 60% baseline + 40% no_zipper still reads as baseline).
- **Noise robustness:** classification is robust to σ_S2 up to 0.20 (huge noise) — at this n and these three classes, the shapes are too different to confuse with noise.
- **Bottom line:** Tier 0 task as stated is too easy. The next experiment needs *harder* conditions — close-parameter variants, not gross rule swaps.

## Setup

Features per trajectory (16 hand-engineered):
- S₂: final, first, mean, std, range, slope, q1 mean, q3 mean, q3-q1 diff
- Density: final, mean, std, range, growth ratio
- Joint: S₂ at end / (density/100)

## In-distribution result

```
Leave-one-out CV (n=13):
  RandomForest        100.0%  (13/13)
  GradientBoosting    100.0%
  KNN(k=1)            100.0%   ← literal nearest neighbor wins
  KNN(k=3)            100.0%
  LogReg              100.0%
```

Information curve:
```
first  3 timepoints  →  100%
first  5 timepoints  →  100%
first 10 timepoints  →  100%
first 20 timepoints  →  100%
```

Top features (RF importance): `s2_q3_mean (0.10)`, `s2_mean (0.10)`, `s2_q1_mean (0.09)`, `s2_final (0.09)`, `dens_mean (0.08)`.

**Interpretation:** the three conditions are so far apart in trajectory shape that any reasonable model separates them from the very first hour of sim time. This is the failure mode I flagged before running.

## OOD probe

4 trajectories the classifier never saw during training:

| Trajectory | What it actually is | Pred (full-traj clf) | Conf | Pred (10-pt clf) | Conf |
|---|---|---|---|---|---|
| `baseline_isotropic` | Baseline w/ accept_MT=0 | baseline | 0.81 | baseline | 0.96 |
| `baseline_mac` (2h) | Baseline rerun, truncated | baseline | 0.80 | baseline | 0.94 |
| `baseline_tree` (2h) | Tree-router baseline | baseline | 0.80 | baseline | 0.94 |
| `gated_v2` (1h) | Gated MLP collision rule | baseline | **0.57** | baseline | **0.94** |

**Important correction:** The 0.57 on `gated_v2` initially looked like the classifier detecting "MLP approximation produces a different shape than Tim's rule." On apples-to-apples (retrain on first 10 timepoints, classify gated_v2's 10 timepoints), gated_v2 classifies as baseline with the same 0.94 confidence as actual baselines. **The original low conf was a length-distribution-shift artifact, not mechanism signal.** The gated MLP is indistinguishable from baseline at this classifier resolution.

## Synthetic hybrids (baseline → no_zipper morph)

Linear interpolation of trajectories in (S₂, density) space:

| α (baseline weight) | Pred | Conf |
|---|---|---|
| 1.0 | baseline | 0.99 |
| 0.9 | baseline | 0.99 |
| 0.8 | baseline | 0.97 |
| 0.7 | baseline | 0.94 |
| 0.6 | baseline | 0.90 |
| **0.5** | **baseline** | **0.62** |
| 0.4 | baseline | 0.45 |
| **0.3** | **no_zipper** | **0.38** |
| 0.2 | no_zipper | 0.73 |
| 0.1 | no_zipper | 0.92 |
| 0.0 | no_zipper | 0.94 |

Decision boundary at α ≈ 0.35. Calibrated uncertainty: minimum confidence (max entropy) right at the crossover.

## Noise robustness

Add iid Gaussian noise to baseline_s43, 20 trials per setting:

| σ_S2 | σ_density | Predicted | Mean conf |
|---|---|---|---|
| 0.00 | 0   | baseline | 0.992 |
| 0.05 | 200 | baseline | 0.968 |
| 0.10 | 400 | baseline | 0.918 |
| 0.15 | 600 | baseline | 0.847 |
| 0.20 | 800 | baseline | 0.868 |

At σ_S2 = 0.20 (huge — 20% std on values in [0,1]), classification still holds. The shape signature is robust to plausible measurement noise.

## What this tells us about the Tier 1 / Tier 2 plan

**Confirms:** The trajectory-classifier-as-mechanism-detector framing works. It picks up rule signatures from observation alone.

**Sharpens the next experiment:** The interesting ML question isn't "can we identify rule X out of {baseline, no_zipper, always_catas}." That's solved with the simplest possible model. The interesting question is:

1. **How close can two rules be in parameter space before they're indistinguishable?** Run baseline with `TH_CRIT = 35°, 40°, 45°, 50°` (all should yield "baseline-shaped" trajectories with different alignment ceilings). Where does the classifier confuse them?
2. **Can the classifier detect ablations to the *geometry* path that don't change the *decision* path?** E.g., replace `zipper_geometry` with a noisy version but keep `decide_outcome` clean. This tests whether the trajectory carries information about the bit-exact-geometry path vs the decision path.
3. **The gated MLP is genuinely indistinguishable.** That's a real result, not a probe failure. It means the gated approach we'd recommend to Tim *truly* preserves the macroscopic dynamics — confirmed at the level a classifier can detect.

## Next concrete step

Generate 5×4 grid on nigel: condition × seeds × 1h sims for {`baseline_TH35`, `baseline_TH40` (default), `baseline_TH45`, `baseline_TH50`} × 5 seeds each. ~3-4 hours nigel wall time. Train classifier on this, see at what TH_CRIT spacing it can no longer distinguish.

That's the actual Tier 1 experiment — a controlled-parameter sweep instead of a gross-rule sweep. It tests *resolution*, not *signal*.

## Files

- Scripts: ad-hoc in conversation; saving the key bits to `tier0_probe.py` as next commit
- Data used: `/tmp/mt_results_raw.json` (re-extractable from nigel pickles), `/tmp/all_ood.json` (OOD trajectories)
- Repo state: this notes file lives in `sprint_2026-05-21/` alongside step1/step2/step3 scripts
