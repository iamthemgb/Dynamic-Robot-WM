#!/bin/bash
# P1 pre-flight tests (1 GPU). ALL must pass before the smoke run.
#SBATCH --job-name=proj_smoke_tests
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --output=/gpfs/radev/scratch/sous/mzl7/wan_projectile_smoke_runs/logs/tests_%j.log

set -euo pipefail
# Submit from the repo root (the dir containing projectile_smoke/).
cd "${SLURM_SUBMIT_DIR}"
export WAN21_ROOT=${WAN21_ROOT:-/gpfs/radev/scratch/sous/mzl7/Wan2.1}
PY=~/.venvs/wan21/bin/python

echo "=== 1/4 flow-matching sign convention vs FlowUniPCMultistepScheduler ==="
$PY projectile_smoke/tests/test_flow_convention.py

echo "=== 2/4 encoder shapes + grad-through-frozen-text_embedding + LoRA ==="
$PY projectile_smoke/tests/test_encoder.py

echo "=== 3/4 VAE round-trip + cached-latent provenance (go/no-go) ==="
$PY projectile_smoke/data/check_vae_roundtrip.py

echo "=== 4/4 overfit 4 episodes + save/resume continuity ==="
$PY projectile_smoke/tests/test_overfit.py

echo "ALL P1 TESTS PASSED"
