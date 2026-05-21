# 24 hours on Tim's elastic MT simulator

This is my exploration log from 2026-05-20 → 21 on Tim Tian's plant cortical microtubule simulator (his [thesis](https://dx.doi.org/10.14288/1.0417339), his [repo](https://github.com/yytimtian/Elastic-Cortical-Microtubules)). I wanted to see whether mech-interp tools — ablation, surrogate models, SAEs — work on a hand-built biological simulator the same way they work on neural nets.

Short answer: yes, with one large refactor first, and the findings reframe what the simulator is doing.

**Full writeup with charts:** [`LEARNED_IN_24H.html`](LEARNED_IN_24H.html) (open in browser).

---

## The physics, so I don't lose track

Plant cells build a thin shell of microtubules (MTs) on the inner face of the plasma membrane — the **cortical array**. Each MT is a stiff polar filament a few micrometres long, growing from one end. Through repeated collisions with neighbouring MTs, the array spontaneously organizes into a single dominant orientation — a **nematic state**. This orientation tells the cell which way to deposit cellulose, which determines which way the cell elongates. Plant shape is downstream of MT alignment.

Three coupled mechanisms in the sim:

| Mechanism | What it does | Code |
|-----------|--------------|------|
| **Dynamic instability** | MT tips grow / shrink stochastically (rates `v+`, `v_t-`, `v_s-`, catastrophe `r_c`, rescue `r_r`) | `parameters.py` + event loop in `sim_algs_fixed_region.py` |
| **Collision rule** | Outcome depends on incident angle θ between two MTs | `collision/decision.py` + `collision/geometry.py` |
| **Elastic deflection** | MT bends between membrane anchor points (Euler–Lagrange BVP) | `sim_anchoring_alg_test.py` |

**The collision rule** (the thing I came here to ablate):

```
  if  θ ≤ TH_CRIT (= 40°):
      → zipper+ or zipper-     (MT 1 bends to align with MT 2)
  else:
      → catastrophe   if r == 0
      → crossover     if r == 1
```

40° is from Tim's biophysics calibration. Everything about alignment in the cortical array depends on the fraction of collisions in each branch.

**The readout** is the nematic order parameter:

```
S₂ = ⟨ cos 2(θᵢ − θ̄) ⟩    (length-weighted, per region)
```

`S₂ = 0` for isotropic, `S₂ = 1` for perfectly parallel, real BY-2 cells sit around `0.7–0.9`.

---

## What I did

### Day 1 — refactor zip_cat so I can ablate it without editing Tim's code

Tim's `zippering.py::zip_cat()` is 300 lines of nested angle-quadrant cases doing three things at once: decide the outcome, compute post-collision geometry, AND provide the 5-tuple return contract the rest of the sim depends on. Tried to ablate in place. Hit walls every time — the bookkeeping is wired into the function signature.

Split into three pure modules:

```
collision/
├── decision.py    # decide_outcome(a1, a2, r) -> str
├── geometry.py    # zipper_geometry(a1, a2) -> (new_angle, label)
└── api.py         # zip_cat_clean(...) -> 5-tuple  (shim for the rest of the sim)

zippering.py       # unchanged interface, now forwards to collision.api
sim_algs_fixed_region.py, parameters.py — untouched
```

Now ablation is one line:

```python
import collision.decision
collision.decision.decide_outcome = lambda a1, a2, r: 'cross'
# every collision is now a crossover. run the sim.
```

**Equivalence check (because this is meaningless if the physics changed):**

| Check | N | Result |
|-------|---|--------|
| Outcome label + new_angle + new_pt match | 100,000 random cases | match within 1e-9 |
| Same checks against Tim's `main` branch | 50,000 random cases | match within 1e-9 |
| End-to-end S₂ trajectory, seed-matched | 1 sim-hour | bit-identical for first ~0.5h, then 1–3 % drift from FP amplification (statistical equivalence holds; bitstream doesn't) |
| MT density distribution | 30 seeds × 1h | 775–794 vs Tim's 771–792 — overlapping |
| `tests/` | — | 81 tests pass on this branch |

The drift is expected for a chaotic event-driven sim. The refactor preserves Tim's physics, not his bitstream.

### Day 2 — actually ablate, train an MLP, train an SAE

Three ablation conditions × 4 seeds × 10 sim-hours on nigel:

| Condition | Patch | What I expected | What happened |
|-----------|-------|-----------------|---------------|
| `baseline` | none | reference S₂ | S₂ 0.59 → **0.86** over 9h, density 491 → 1713 |
| `no_zipper` | `'cross' if r==1 else 'catas'` | S₂ collapses | S₂ locks at 0.57 in the first hour, **never moves**, density flat at ~360 |
| `always_catas` | `lambda: 'catas'` | very low S₂ | S₂ plateaus at 0.70 at density ~221 |

The static-S₂-at-1h gap (baseline 0.78 vs no_zipper 0.56) understates the effect. Over 10h the gap grows to +0.29 because density compounds. The interesting variable isn't S₂ at any single time, it's the **slope of S₂** — zippering is a rate, not a value.

Then two surrogates:

- **MLP collision rule.** 5k-param MLP, inputs `(cos a1, sin a1, cos a2, sin a2, r)`, trained on 1M `zip_cat` calls. Hits 99.5 % top-1 accuracy and **plateaus there** — boundary oversampling didn't help, so the 0.5 % residual is capacity or feature encoding, not data. Dropping it straight into the sim crashes because the geometry head's regression error breaks bundle bookkeeping. The gated version (MLP only for cross-vs-catas, deterministic geometry for zippers) runs end-to-end and reproduces baseline S₂ exactly.
- **Trajectory SAE.** 32-feature SAE on 148k per-region snapshots, unsupervised. All 32 features alive, each tuned to a preferred angle in `[12°, 158°]`. The SAE recovers angular tuning as the dominant axis without being told. Tim's hand-designed S₂ ≈ what an unsupervised method finds.
- **Probe on the MLP's hidden layer (this morning's sprint).** Linear probe finds `h2 → incident_angle` at **R² = 0.93**. An SAE on the same hidden layer recovers feature tuning curves with three features peaked at `TH_CRIT = 40°` and two features at `~2°` with opposite zipper-sign selectivity. The MLP discovered Tim's threshold variable AND alignment direction as separate latent axes.

---

## What this changed in how I think about the simulator

1. **Zippering does two jobs, not one.** It aligns MTs at collision *and* keeps them alive (a non-zipper collision is much more likely to end in catastrophe). The dominant macroscopic effect is the density story — alignment compounds because aligned MTs survive longer and meet more partners. This isn't visible in single-snapshot S₂.

2. **S₂ alone is misleading.** Always report `(S₂, density)` jointly. At 9h: baseline `(0.86, 1713)`, always_catas `(0.70, 221)`. Both look like "high alignment" if you only see S₂. The 0.70 with 221 MTs is mostly small-sample bias (S₂ on sparse populations doesn't average to zero).

3. **I had the floor story wrong on Day 1.** First write-up said "branched nucleation contaminates S₂ at low density (`accept_MT = 0.24`)." Checking the gates: in default BY-2 the `LDD_bool = False` flag makes `sample_angle` (the branched-nucleation generator) dead code regardless of `accept_MT`. The 0.57 floor on `no_zipper` and 0.70 on `always_catas` are estimator bias at low N, not nucleation-correlated alignment. Need Tim to confirm I'm reading the gate right.

4. **Mech-interp tools transfer.** The same ablation / surrogate / SAE / probe stack that's standard for neural nets works on a Python biophysical simulator, *if* the modular boundaries are clean. The whole Day 1 was just making the boundaries match the conceptual decomposition so Day 2 was tractable.

---

## Things I'm curious about (to chat over)

Not a list of asks — just what I bumped into where Tim's perspective would change my read.

1. **`LDD_bool` semantics.** Does `LDD_bool = False` in default BY-2 actually disable branched nucleation entirely? If so, my Day 1 take on the floor was wrong (see point 3 above) and the corrected story is in `LEARNED_IN_24H.html` §5.
2. **Catastrophe source.** Right now `r_c` (spontaneous catastrophe) and collision-induced catastrophe share one code path. To cleanly ablate "what kills MTs?" we'd want to tag the source of each. Are these biologically the same process, or two distinct things?
3. **S₂ + density convention.** If you report only S₂ in the thesis, the always_catas-style cases (high S₂ at very low density) would read as "aligned" when really they're estimator-bias. Curious how you've thought about this.
4. **The `deflect_on = False` crash.** Tried to ablate elastic deflection. `comparison_fns.region_traj.add_traj` asserts no two trajectories share both angle and point — true under elasticity, false when MTs go straight. Patching crashes deeper. The next refactor is to separate trajectory ID from elastic-bend uniqueness. If we ever sit down to look, this is the one I'd most want to do with you.
5. **10-hour run pickles.** If you happen to have any of your own 10h runs around, it'd be a useful sanity check against my refactor's trajectories. Not blocking anything — just nice to have.

The refactor itself: if you find any of the modular split useful, all yours — graft it, change it, ignore it, whatever. I'm working in a branch on my fork; nothing to push your way unless you want it.

---

## Repo layout

```
collision/             # added: clean ablation surface (decision / geometry / api)
sae/                   # added: trajectory SAE + learned collision MLP
sprint_2026-05-21/     # added: retrain MLP + probe hidden layer
tests/                 # added: equivalence + unit tests
models/                # added: trained MLP weights

# Tim's original — untouched:
parameters.py
run_sim_array_rerun.py
sim_algs_fixed_region.py
sim_anchoring_alg_test.py
zippering.py           # now a thin forwarding shim
comparison_fns.py
plotting.py

LEARNED_IN_24H.html    # full writeup with charts
README.md              # this file
```

---

## Running stuff

**Tim's original sim** (unchanged):

```bash
python -c "import run_sim_array_rerun; run_sim_array_rerun.run(...)"
```

**An ablation:**

```python
import collision.decision
collision.decision.decide_outcome = lambda a1, a2, r: 'cross'  # always-cross
# or 'catas', or random, or a learned MLP — see sae/learned_collision.py

import sim_algs_fixed_region as sa
sa.simulate(seed=42, hours=1, out_dir='./out/', troubleshoot=False)
```

**The equivalence tests** (run this before trusting anything else):

```bash
pytest tests/test_equivalence.py tests/test_patch_label_geometry_consistency.py -v
```

**The MLP retrain sprint** (Mac, ~5 min):

```bash
bash sprint_2026-05-21/run_sprint.sh
```
