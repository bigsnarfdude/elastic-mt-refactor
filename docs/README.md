# Findings documentation

Open `findings.html` in a browser. Self-contained — no external dependencies,
images are local in this folder.

## What's here

- `findings.html` — the four findings from this morning's work: temporal
  dynamics of S₂, SAE rediscovers orientation, neural surrogate of zip_cat,
  and the LDD_bool/sample_angle deadcode discovery.
- `sweep_results.png` — 3-panel: S₂ bars, S₂ trajectories, MT density.
  n=4 seeds × 3 conditions × 1 sim-hour.
- `sae_feature_prototypes.png` — 16 SAE feature decoder weights, each
  showing the angle preference. Orientation-selective filters.
- `sae_analysis.png` — orientation tuning curves by condition, 4 example
  feature prototypes, scatter of features by correlation with S₂ vs cos(2Ω).

## How to read this

Start with the TL;DR. Then look at the middle panel of `sweep_results.png`
(the trajectory plot) — that's the headline result. Then the four findings
sections walk through what each plot shows. The "How to phrase it for Tim"
paragraph at the bottom is the conversation-starter.

The companion data lives in `../models/` — load the trained SAE and MLP
directly from there.
