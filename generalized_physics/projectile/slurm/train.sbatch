#!/bin/bash
# Generalized-arm projectile training test. Usage:
#   sbatch generalized_physics/projectile/slurm/train.sbatch
# Auto-resumes from the latest checkpoint in out_dir if requeued.
#SBATCH --job-name=gen_proj
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --constraint="a100|h100|h200"
#SBATCH --output=/gpfs/radev/home/mzl7/scratch/wan_generalized_runs/logs/train_%j.log

set -euo pipefail
cd /gpfs/radev/home/mzl7/scratch/wan_scripts_new
PY=~/.venvs/wan21/bin/python
CACHE=/gpfs/radev/home/mzl7/scratch/wan_projectile_smoke_cache
WAN=/gpfs/radev/home/mzl7/scratch/Wan2.1

echo "=== BLOCKING GATES (cache provenance + flow convention) ==="
test -f "$CACHE/reprojection_check/PASS" || {
  echo "GATE FAIL: reprojection go/no-go has not passed"; exit 1; }
test -f "$CACHE/vae_roundtrip/PASS" || {
  echo "GATE FAIL: VAE round-trip has not passed"; exit 1; }
(cd "$WAN" && $PY physics_finetune/tests/test_flow_convention.py)

echo "=== UNIT TESTS (generalized package, CPU) ==="
$PY -m generalized_physics.tests.run_all

echo "=== GENERALIZED PHASE-1 TRAINING TEST ==="
$PY -m generalized_physics.projectile.train_real \
    --config generalized_physics/projectile/configs/train_real.yaml

echo "GENERALIZED PROJECTILE RUN COMPLETE"
