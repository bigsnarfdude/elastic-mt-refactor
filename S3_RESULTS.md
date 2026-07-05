# Sprint S3 — SAE and the order parameter · RESULTS (2026-07-05)

Prereg: `mt/PREREG_S3_sae_order_param.md`. Trained on the S1 ablation snapshots (250 orderp pickles,
2500 per-region angle histograms, S₂ spanning 0.07→0.89). Held-out split: train seeds 1–3, test seeds 4–5.

## Verdict: single-feature hypothesis disconfirmed → the real structure is a 2-atom rotation-invariant code.

The pre-registered single-feature test fails (best held-out single-feature |r| = 0.68, below 0.70, and
seed-unstable: 0.99 on seed 4, 0.24 on seed 5). Investigating *why* yields the actual, positive result: the
SAE encodes nematic order as a **minimal 2-atom rotation-invariant director code** that recovers S₂ at
**r = 0.99, seed-stable** — matching the hand-designed (Ω = director angle, S₂ = scalar order) decomposition.

## The decisive result: order is exactly 2-dimensional in the learned basis
Reconstruct each held-out histogram from only its top-k order-carrying atoms (the SAE's own decoder + bias,
**no fitted weights**), then apply the real S₂ function to the reconstruction:

| atoms | S₂ recovery (held-out) | seed 4 | seed 5 |
|---|---|---|---|
| 1 atom | 0.69 | 0.51 | 0.99 |
| **2 atoms** (tuned to 112° + 69°) | **0.99** | **0.99** | **0.99** |
| 3 atoms | 0.994 | 0.994 | 0.995 |

One atom is direction-dependent (reads S₂ only when the array aligns to it — 0.51 vs 0.99 by seed). **Two
atoms recover S₂ at 0.99, stably across both held-out seeds; a third adds nothing.** The jump 0.69→0.99→0.994
is the signature of a rotation-invariant magnitude that needs — and only needs — a 2D (director) representation.
Because this uses the SAE's own reconstruction and the fixed physics S₂ functional, it is an *interpretable*
low-rank readout, not an arbitrary dense probe.

## Two is the provable ceiling — and capacity only dilutes it
"2 atoms suffice" is not where the sweep got tired; it is the exact dimensionality of the target. S₂ is the
magnitude of the **2nd Fourier harmonic**, `S₂ = |Σ h·e^{i2θ}|`, and `e^{i2θ} = cos2θ + i·sin2θ` has exactly
two real components — so S₂'s per-snapshot variation lives in a **2-dimensional subspace, full stop**. One
atom is a *line* in that plane (a direction-conditional readout → the near/far split below); two non-parallel
atoms *span* it (all of S₂); a third is redundant by construction (its 2nd-harmonic content must lie in the
same 2D plane). No dictionary, however large, can beat this.

Capacity sweep (n_features 8→32→64, S₂-recovery ceiling on held-out):

| SAE size | live atoms | 2nd-harmonic energy in top-2 atoms | S₂ ceiling |
|---|---|---|---|
| **8** | 3 | **83%** | 0.994 |
| 32 | 16 | 31% | 0.994 |
| 64 | 40 | 14% | ~0.99 (needs ≫6 atoms to reassemble) |

The ceiling never rises — it is pinned at ~0.99 by the 2D physics — but larger dictionaries **fragment the
same 2D signal across more atoms** (83% → 31% → 14% concentrated in the top two), so you must gather ever more
of them just to reassemble what a clean 2-atom basis gives directly. **The smallest SAE is the best one:**
nf=8, concentrating 83% of the order signal into 2 atoms, is the most parsimonious and interpretable code.
For a low-dimensional target, more sparse capacity *smears* the feature you care about rather than sharpening
it — the opposite of "more is better." (Sweep: `s3_capacity.py`.)

## The decisive evidence (held-out seeds 4–5)
Best feature = feature 3, preferred angle **67.5°**. Splitting held-out snapshots by how far the array's
director Ω is from that angle:

| array orders … | feature 3 ↔ S₂ | n |
|---|---|---|
| **near** feature 3's angle (<30°) | **r = 0.987** | 430 |
| **far** from it (≥30°) | r = 0.405 | 570 |

A single feature *is* S₂ when the array aligns to that feature's direction, and stops tracking it when the
array orders elsewhere. That is exactly what "direction-tuned feature vs rotation-invariant target" predicts,
and it explains the seed 4 (0.99) vs seed 5 (0.24) split (per-seed S₂ variance is full in both — std ≈ 0.29 —
so it was not a low-variance artifact).

## What the SAE actually represents
- The per-region angle histograms are ~3-dimensional: 3 live atoms reconstruct them near-perfectly
  (recon MSE ≈ 6e-5). The SAE learns an **angle-tuned basis** (a director code), not a scalar-order feature.
- **S₂ is inherently 2-dimensional** (magnitude of the 2nd Fourier mode: a director angle Ω + a scalar
  magnitude S₂), so it cannot collapse into one angle-tuned atom — but it *does* collapse into two (table above).
- The two order-carrying atoms are tuned to 112° and 69°, spanning the nematic plane; their joint reconstruction
  is rotation-invariant. A dense probe over all features also recovers S₂ (R²=0.96), but the 2-atom result is
  the stronger claim: order is *low-rank and interpretable* in the learned basis, not just decodable.
- Honest correction: a first rotation-invariant readout I wrote returned r=0.31 — that was a bug (it omitted
  the decoder bias, where the histogram's uniform component lives), not a real negative. The corrected
  reconstruction (bias included) gives the 0.99 above.

## Why this is a result and not a failure
The hand-designed order parameters are (Ω = director angle, S₂ = scalar order). The unsupervised SAE
independently arrives at a **direction-tuned code** — i.e. it discovers Ω-structure — and represents order
magnitude as a pooled property across that bank, not as one feature. So "no single S₂ feature" is not the SAE
missing order; it is the SAE representing order the same 2D way the physics does. The single-feature test was
the wrong question; the near/far split is the right answer.

## Methodology note (how the negative was kept honest)
- Model selection (n_features, sparsity) was done on **train-side reconstruction + dead-feature count only**,
  blind to the S₂ outcome — not by picking the config with the best S₂ correlation (that would be fishing
  against the prereg's own kill condition).
- First-pass run was undertrained (23/32 dead features) and its "null" was NOT trusted; retrained to a healthy
  dictionary before reading the result (`Don't cite results from broken scripts`).

## Caveats / what would make it airtight
- The dead atoms (5/8 here, 37/40 at nf=64) are the *data* being ~2–3-dim, not the SAE being too small. The
  capacity sweep above rules out "a bigger dictionary would help" — it only dilutes. What *would* sharpen the
  claim is a **cleaner 2-atom basis** (an explicit orthogonal cos2θ/sin2θ pair, e.g. via a 2-feature SAE or
  rotation-augmented training), nudging 0.994→~1.0. That is a refinement, not a rescue.
- The genuinely open direction is **richer input, not more atoms**: the angle histogram is intrinsically ~2D,
  so to find structure *beyond* S₂ (tetratic S₄, defect density, bundle/domain structure, spatial correlation
  length) the SAE must be fed a representation that contains it — spatial maps, per-MT length+position, raw
  geometry. Only then does capacity earn its keep and could an SAE find order parameters not hand-designed.
- Single train/test split (seeds 1–3 / 4–5). The near/far mechanism, not the pooled number, is the load-bearing result.

## Files
`s3_train_probe.py`, `s3_v3.py` (model-selection), `s3_diag.py` (near/far mechanism), `s3_atoms_v2.py`
(k-atom reconstruction), `s3_capacity.py` (capacity/ceiling sweep). Result JSONs in `s3_results/`.
Data: S1 `~/mt_ablation/*/orderp_*.pickle`.
