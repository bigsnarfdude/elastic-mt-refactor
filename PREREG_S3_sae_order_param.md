# Sprint S3 — SAE rediscovers the order parameter · PRE-REGISTRATION

**Written 2026-07-05 (Sun ~08:00), before running.** Extends the May SAE work (`elastic-mt-refactor/sae/`).

## One-line question
Does an SAE trained **unsupervised** on per-snapshot MT geometry develop a **single sparse feature** that
tracks the hand-designed nematic order parameter S₂ across the full ordered→disordered range?

## Success criterion
≥1 learned feature with |corr| > 0.7 to per-snapshot S₂, stable at n≥3 seeds (not a one-seed artifact),
AND it is a **single dictionary atom**, not a fitted dense linear combination of many features.

## Kill / branch
S₂ is only recoverable as a **dense** linear readout of the input (trivial — S₂ is computed *from* the same
angle histogram) with no sparse feature aligning to it → the honest finding becomes "S₂ is not a natural axis
of the learned dictionary." That's a real result about SAEs on this data, reported as such (Rule 12).

## Why now / data (the S1 gift)
- Training set already exists on nigel: the 25 S1 ablation runs, `~/mt_ablation/<cond>_seed<n>/states/orderp_*.pickle`,
  ~10 snapshots each, spanning **S₂ ≈ 0.07 → 0.89**. No new simulation needed.
- The diversity is the upgrade over May (which trained on baseline-like data where S₂ barely moved): a feature
  can only track S₂ if S₂ actually varies — now it does.

## Design
- **Input:** per-region angle-histogram feature vectors (`sae/snapshot.py::load_per_region`, `N_BINS`).
- **Model:** existing single-layer SAE (`sae/model.py`), ~32 features, sparsity_coef 0.02.
- **Split:** hold out 1–2 conditions/seeds for a clean test.
- **Labels:** ground-truth S₂ + dominant angle Ω per snapshot (`plotting.s2` / `sae/probe.py`).

## Day 1 (today AM/PM) — data + train
- Deploy `sae/` to nigel (data is there). Point `retrain_multiseed.collect_per_region` at the S1 orderp set.
- **Smoke-gate:** one tiny train; confirm reconstruction loss drops and features are non-dead BEFORE trusting anything.
- Train SAE across pooled S1 data; label every snapshot with S₂/Ω.
- Checkpoint: trained SAE + (features × S₂/Ω) correlation matrix.

## Day 2 — probe, interpret, artifact
- `probe.py`: correlate each feature with S₂, Ω, density; rank.
- Verdict vs success/kill: single feature |r|>0.7 to S₂, stable across seeds?
- **Anti-trivial check (never cut):** best *single sparse feature* → S₂ vs *dense linear probe* → S₂. If only
  dense works, that's the kill finding.
- Visualize top feature's preferred-angle profile + its activation along an ordering trajectory
  (zipper_to_cross climbing vs no_zipper flat).
- Deliverables: `highlevel_*.html` + one shared-doc paragraph + correlation table.

## Risks / guardrails
- **Trivial-decodability trap (the big one):** S₂ is a function of the input, so some linear readout always
  recovers it. The claim is *sparse single-feature* alignment — bake that into the metric.
- Dead features at low seed count (May `n1_vs_n10` issue) — train n≥3, check liveness first.
- The "Gemma 3 / GemmaScope 2" memory is about LLM SAEs — NOT this numpy SAE on simulator geometry. Don't misapply.

## Descope ladder
1. Ω / defect-density probes (keep S₂ only).  2. Multi-seed robustness (n=1 + caveat).
3. Artifact trajectory overlay (static table).  **Never cut:** single-feature-vs-dense-probe comparison.
