# S1 Rule-Ablation — LAUNCHED on nigel (Sat 2026-07-04, ~08:30 MDT)

Launch-and-collect sprint per `mt/SPRINT_S1_rule_ablation.md`. This file = the collect runbook.

## What's running
- **Host:** nigel (`vincent@nigel.birs.ca`), 24 cores, `~/venv/bin/python` (numpy 1.26.4).
- **Code:** `~/elastic-mt-refactor/` — S1 harness rsync'd from mac (branch `refactor/split-zip-cat`,
  local snapshot commit `582e2fb`). Verified on nigel: `run_ablation.py` present, `decision_fn`
  hook (api ×4), `ABLATION` global, sim imports OK. Local Step-0 = **34 tests passed** (incl. the
  override-fires isolation tests = the selfcheck's load-bearing property).
- **Matrix:** 6 conditions × 5 seeds × idx=9 (**10 sim-hours** = benchmark cutoff) = **30 runs**.
  - baseline, no_zipper, zipper_to_cross, no_induced_catas, always_catas, always_cross
- **Launch mechanics:** `s1_fleet.sh` (nohup) launched 29 runs, throttled MAXJOBS=22, thread-pinned
  (`OMP/OPENBLAS/MKL/NUMEXPR=1`).
- **REVISED TIMING (from live logs):** ~55–60 min wall per sim-hour under full load ⇒ **~10 h wall/run**.
  First 22 (launched 08:30) done ~18:30 Sat; the 7 queued behind the throttle finish ~04:30 **Sun AM**.
  So the full matrix is ready to collect **Sunday morning**.
- **Self-healing:** `s1_sweeper.sh` (nohup, every 15 min) relaunches any matrix run that dies without
  `result.json` (OOM/segfault). Race-safe (only touches runs the driver already launched), capped at
  2 attempts/run, logs to `~/logs/s1_sweeper.log`, exits when all 30 have `result.json`.
- **Known death handled:** `no_induced_catas_seed1` was OOM-killed ~1 sim-hr in (no traceback, no
  result.json; density-runaway condition, peaked during initial dense launch). Manually relaunched
  11:52 MDT; memory now ample (52 G free). The sweeper covers any recurrence.
- **30th run:** `baseline_seed1` handled separately. A 2-sim-hr **canary** (idx=1) validates the
  write path first; `s1_after_canary.sh` (nohup watcher) then auto-launches `baseline seed1 idx=9`
  once the canary's `result.json` validates (status ok + s2_final present). If validation fails it
  writes `!!! CANARY VALIDATION FAILED` to `~/logs/s1_after_canary.log` and does NOT launch.
- **Output:** `~/mt_ablation/<condition>_seed<seed>/result.json` (+ `states/` pickles). Logs in
  `~/logs/s1_*.log`, driver progress in `~/logs/s1_fleet_driver.log`.

## Check progress (any time)
```bash
ssh vincent@nigel.birs.ca 'echo procs $(pgrep -fc run_ablation.py); \
  echo done $(find ~/mt_ablation -name result.json|wc -l)/30; \
  grep -c "^launch" ~/logs/s1_fleet_driver.log; tail -2 ~/logs/s1_after_canary.log'
```

## Collect (Sunday)
1. `rsync -az vincent@nigel.birs.ca:~/mt_ablation/ ~/Desktop/me/microtubles/elastic-mt-refactor/s1_results/`
2. **Sanity gate FIRST** (validates harness vs v0.2 before trusting any ablation):
   - `baseline` → `s2_region_lw` final ≈ **0.86**, high density.
   - `no_zipper` → `s2_region_lw` final ≈ **0.56** (the floor), density flat ~360.
   - If these two anchors don't reproduce → extraction/wiring wrong; STOP before reading ablations.
3. Build `ablation_matrix.tsv` = per-run (condition, seed, s2_final, s2_global_final, density_final,
   early-phase S₂ slope, time-to-S₂=0.8, status). Aggregate mean±std over 5 seeds.
4. Causal table + H1/H2 verdicts; report **both** `s2_region_lw` and `s2_global` (gap = low-N bias).
5. `highlevel_*.html` artifact (trajectories + table) + 1 paragraph for Alex's shared doc.
6. Archive: commit results + push (chaos protocol — data durable before analysis).

## Pre-registered predictions (so results can disconfirm) — from sprint doc
| Condition | S₂ slope | final density | tests |
|---|---|---|---|
| baseline | strong+ (→0.86) | high | reference |
| no_zipper | ~0 (locks ~0.57) | low/flat (~360) | H1 (coupled) |
| zipper_to_cross | + weaker than baseline | between no_zipper & baseline | align vs keep-alive |
| no_induced_catas | weak/uncertain | bounded rise (r_c still caps) | H2 |
| always_catas | n/a sparse | very low (~221), S₂ estimator-biased | H2 |
| always_cross | ~0 | highest | density ceiling |

## Notes / deviations
- H3 (branch-nucleation) dropped upstream — `branch_stuff` is entrainment sideness, not a nuc switch;
  branched nuc is `LDD_bool`-gated OFF in this config. Not in the matrix.
- Archival is rsync-back + commit-on-mac (nigel's dir is NOT a git repo), not git-on-nigel. Achieves
  the same durability-before-analysis requirement.
- Config confirmed at launch: `LDD=False branch_stuff=True ABLATION=None` for baseline = benchmark config.
