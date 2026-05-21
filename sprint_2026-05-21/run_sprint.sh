#!/bin/bash
# 90-min sprint runner: retrain MLP to label-100%, probe hidden layer, verify in sim.
#
# Run from the elastic-mt-refactor repo root:
#   cd ~/Desktop/microtubles/elastic-mt-refactor
#   bash sprint_2026-05-21/run_sprint.sh
#
# Or just:
#   bash run_sprint.sh   (if you're already in this dir)

set -e

cd "$(dirname "$0")/.."   # repo root
REPO="$(pwd)"
SPRINT="$REPO/sprint_2026-05-21"
LOG="$SPRINT/sprint.log"

mkdir -p "$SPRINT"
> "$LOG"

echo "================================================================" | tee -a "$LOG"
echo "MT MLP SPRINT — $(date)" | tee -a "$LOG"
echo "Repo: $REPO" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"

START_ALL=$(date +%s)

# ─── Step 1: Retrain to label-100% ───────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "▶ STEP 1 — Retraining MLP (target: 100% on label)" | tee -a "$LOG"
START_1=$(date +%s)
python3 sprint_2026-05-21/step1_retrain_100.py 2>&1 | tee -a "$LOG"
ELAPSED_1=$(( $(date +%s) - START_1 ))
echo "  step 1 wall: ${ELAPSED_1}s" | tee -a "$LOG"

# ─── Step 2: Probe hidden layer ──────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "▶ STEP 2 — Probing hidden layer (h2 → incident_angle, SAE)" | tee -a "$LOG"
START_2=$(date +%s)
python3 sprint_2026-05-21/step2_probe_hidden.py 2>&1 | tee -a "$LOG"
ELAPSED_2=$(( $(date +%s) - START_2 ))
echo "  step 2 wall: ${ELAPSED_2}s" | tee -a "$LOG"

# ─── Step 3: Verify in the simulator ─────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "▶ STEP 3 — Verifying in simulator (baseline / gated_v2 / full_mlp_v2)" | tee -a "$LOG"
START_3=$(date +%s)
python3 sprint_2026-05-21/step3_verify_gated.py 2>&1 | tee -a "$LOG"
ELAPSED_3=$(( $(date +%s) - START_3 ))
echo "  step 3 wall: ${ELAPSED_3}s" | tee -a "$LOG"

# ─── Summary ─────────────────────────────────────────────────────────────────
TOTAL=$(( $(date +%s) - START_ALL ))
echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "SPRINT COMPLETE — total wall: ${TOTAL}s ($((TOTAL/60))m $((TOTAL%60))s)" | tee -a "$LOG"
echo "  step 1 (retrain):  ${ELAPSED_1}s" | tee -a "$LOG"
echo "  step 2 (probe):    ${ELAPSED_2}s" | tee -a "$LOG"
echo "  step 3 (verify):   ${ELAPSED_3}s" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Results:" | tee -a "$LOG"
echo "  models/learned_zipcat_v2.npz" | tee -a "$LOG"
echo "  sprint_2026-05-21/step1_metrics.json" | tee -a "$LOG"
echo "  sprint_2026-05-21/step2_metrics.json" | tee -a "$LOG"
echo "  sprint_2026-05-21/step2_sae_features.png" | tee -a "$LOG"
echo "  sprint_2026-05-21/step2_feat_by_outcome.png" | tee -a "$LOG"
echo "  sprint_2026-05-21/step3_metrics.json" | tee -a "$LOG"
echo "  sprint_2026-05-21/sprint.log  (this file)" | tee -a "$LOG"
