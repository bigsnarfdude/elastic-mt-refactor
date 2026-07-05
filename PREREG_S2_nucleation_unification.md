# Sprint S2 — Nucleation Unification (cross-engine) · PRE-REGISTRATION

**Written 2026-07-05 (Sun), before running.** Prereg first so the weekend can't drift.
Builds on the v0.2 tier-2 benchmark (`~/Desktop/me/may23sprint_microtubles/bench/RESULTS.md`).

## One-line question
Does matching the **areal nucleation flux and scheme** across engines close the ~5.4× zero-interaction
reference-density gap (CorticalSim `nozipper` 20.80 vs Cytosim `nonsteric` 3.83 µm/µm²)?

## Why this sprint (and why it's the prerequisite)
No cross-engine density comparison means anything until nucleation is standardized. The gap is already
diagnosed as *probably* nucleation, and the arithmetic is self-consistent:
- Density is **areal (µm/µm²)** ≈ areal_flux × MT_lifetime × mean_length.
- Mean lengths are ~equal (~30 µm both engines) → density ∝ areal_flux.
- Areal fluxes: CorticalSim 0.001 /µm²/s vs Cytosim 0.5 events / 2500 µm²/s = 2e-4 /µm²/s → **5× ratio**.
- Observed density gap = 20.80 / 3.83 = **5.4×**. MT counts 12,383 vs 317 (~39×) at equal length.

So the 5× areal-flux difference almost fully accounts for the 5.4× density gap. S2 tests that directly.

## Hypothesis
Set all engines to a **common areal flux = Cytosim's native 2e-4 /µm²/s**, isotropic in position and angle,
at zero interaction (no collision removal). Prediction: the three reference densities converge to within ~1.5×.

## Design (kept deliberately small — one knob, one tier)
- **Common flux:** 2e-4 /µm²/s (match *down* to Cytosim — never up; matching up floods the slow mechanical
  engine with thousands of MTs and re-lives the always_cross runaway).
- **Reference condition:** zero interaction — collisions/zipper/induced-catastrophe genuinely OFF in each
  engine (dynamic instability + nucleation only).
- **Engines:** CorticalSim (`~/corticalsim/corticalSim.release`) and elastic-mt (`~/elastic-mt-refactor`)
  re-run at 2e-4 (both fast). **Cytosim: reuse the existing `cyto:nonsteric` = 3.83 anchor IF its DI params
  match this spec** (verify first); one short Cytosim confirmation run only if there's param doubt.
- **Seeds:** n = 3–5.
- **DI params:** the tier-1 spec (`bench/tier1_spec.json`): v_g 0.08, v_s 0.16, r_c 0.003, r_r 0.005 µm/s.

## Metrics (measure the right thing)
- **Density in the bench's AREAL convention (µm/µm²)** — reuse `tim_to_lengths.py` / `cytosim_to_lengths.py`
  / `compare3.py`. Do NOT carry over S1's region-sum "density" (~1642) — different quantity, not comparable.
- **Realized areal flux** measured from each engine's output (nucleation events / area / time), to confirm the
  three actually match. This is the discriminating check — input knobs are parameterized differently per
  engine (Cytosim events/s; CorticalSim /µm²/s; elastic-mt r_n), and a silent unit/area conversion error
  gives a confident-wrong "convergence." Verify realized flux BEFORE comparing densities.
- Mean length + MT count (sanity: density = count × length / area should reconcile).

## Success criterion
All three zero-interaction reference densities within **~1.5×** of each other (the band has room for the
~27% tier-1 length wobble). Partial convergence (5.4× → ~1.8×) is still informative: "mostly nucleation,
residual geometry."

## Kill / branch conditions
- **Realized fluxes match but densities still diverge → the gap is geometric, not nucleation.** Pivot to
  geometry/area normalization; do not force a nucleation story. (This is the clean, informative failure.)
- Can't confirm a reference condition is truly zero-interaction / isotropic in an engine → fix that first;
  a non-isotropic or collision-on reference isn't a nucleation-only comparison.
- Cytosim anchor's DI params don't match the spec and a re-run is too slow to finish → report elastic-mt ↔
  CorticalSim convergence only, mark Cytosim as pending (honest partial).

## Guardrails (from design review)
1. Areal density convention, reused from the bench — not the S1 number.
2. Verify **realized** flux per engine from output, not the input knob.
3. Match down to Cytosim; reuse its anchor if params match (feasible-today move).
4. Confirm isotropic (position + angle) nucleation and collisions-off in each reference.
5. One flux, one tier, n≤5, three engines (two re-run, one reused). No mid-sprint condition creep.

## Feasibility (launchable today)
All engines built on nigel. CorticalSim fast; elastic-mt known-fast; Cytosim reused or one short confirm.
Smoke test first (verify each runner produces a density + realized flux), THEN launch — do not burn compute
on an unverified runner (the RRMA scar).

## Deliverables
1. `s2_reference_density.tsv` — per (engine, seed): areal density, realized flux, mean length, MT count.
2. The convergence verdict (within 1.5×? partial? kill?) with the realized-flux confirmation table.
3. `highlevel_*.html` artifact + one paragraph for the shared doc (feeds the June nucleation caveat →
   "here's what standardizing it actually does").
