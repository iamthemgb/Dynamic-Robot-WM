#!/bin/bash
# The smoke run. Usage:
#   sbatch projectile_smoke/slurm/smoke.sbatch [configs/smoke_lora.yaml]
# Auto-resumes from the latest checkpoint in out_dir if requeued.
#SBATCH --job-name=proj_smoke
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --constraint="a100|h100|h200"
#SBATCH --output=/gpfs/radev/scratch/sous/mzl7/wan_projectile_smoke_runs/logs/smoke_%j.log

set -euo pipefail
# Submit from the repo root (the dir containing projectile_smoke/).
cd "${SLURM_SUBMIT_DIR}"
export WAN21_ROOT=${WAN21_ROOT:-/gpfs/radev/scratch/sous/mzl7/Wan2.1}
PY=~/.venvs/wan21/bin/python
CFG=${1:-projectile_smoke/configs/smoke_lora.yaml}
CACHE=/gpfs/radev/scratch/sous/mzl7/wan_projectile_smoke_cache

echo "=== BLOCKING GATES (plan P0/P1: no training before these pass) ==="
test -f "$CACHE/reprojection_check/PASS" || {
  echo "GATE FAIL: reprojection go/no-go has not passed"; exit 1; }
test -f "$CACHE/vae_roundtrip/PASS" || {
  echo "GATE FAIL: VAE round-trip has not passed (run slurm/tests.sbatch)"; exit 1; }
$PY projectile_smoke/data/verify_cache.py
$PY projectile_smoke/tests/test_flow_convention.py  # sign bug = beautiful wrong curve

echo "=== SMOKE TRAINING: $CFG ==="
$PY projectile_smoke/training/train_smoke.py --config "$CFG"

echo "SMOKE RUN COMPLETE"
