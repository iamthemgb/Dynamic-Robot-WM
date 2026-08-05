#!/bin/bash
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=96
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# Frugal packed v3 group generation: ONE job holding ONE GPU, with many
# concurrent group-generation workers sharing it. The workload is CPU-bound
# (MuJoCo stepping + ffmpeg encode, ~4 cores/worker); EGL rendering leaves
# the GPU ~idle (a few hundred MB VRAM per worker), so one RTX PRO 6000
# happily serves 24+ workers. rtx-batch bills GPU-only: this shape does all
# of P1 for roughly 5 GPU-hours billed.
#
# Defaults: 96 CPUs -> 24 workers -> P1 (1000 groups, 4000 episodes) in
# ~5.5 h expected wall. 128 CPUs/32 workers is ~4.2 h but needs a node with
# every CPU free; 96 fits current fragmentation.
#
# Work assignment strides over BOTH families: worker w of task t handles
# groups where g % (ARRAY_SIZE*WORKERS) == t*WORKERS+w. Requeue-safe:
# group_dataset_generation skips finished siblings; a requeued job recomputes
# the same slice and resumes where it left off.
#
# Submit from the bundle root (no --array needed for the single-GPU shape):
#   cd <bundle root>
#   DATASET_ROOT=... PHASE_CONFIG=configs/p1_full.json \
#     sbatch --job-name=gen_packed slurm/generate_groups_packed.sbatch
# Optional scale-out to N GPUs (N-fold faster, N jobs of 1 GPU each):
#   ARRAY_SIZE=N sbatch --array=0-(N-1) ... (all tasks must use equal WORKERS)

set -uo pipefail

DATASET_ROOT="${DATASET_ROOT:?DATASET_ROOT must be exported}"
PHASE_CONFIG="${PHASE_CONFIG:?PHASE_CONFIG must be exported, e.g. configs/p1_full.json}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${SLURM_SUBMIT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/zl664_yale/world_model_robotics/envs/pbc-native/bin/python}"
FAMILIES="${FAMILIES:-style020_seed0 style055_seed1000}"
WORKERS="${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-96} / 4 ))}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_SIZE="${ARRAY_SIZE:-${SLURM_ARRAY_TASK_COUNT:-1}}"
WIDTH="${WIDTH:-832}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"
DURATION="${DURATION:-2.5}"

export MUJOCO_GL=egl
export PYTHONPATH="${BUNDLE_ROOT}:${PYTHONPATH:-}"

GROUPS_PER_FAMILY="$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["groups_per_family"])' \
  "${PHASE_CONFIG}")"

STRIDE=$((ARRAY_SIZE * WORKERS))

run_worker() {
  local w="$1"
  # stagger startup: avoids all workers racing the first generation_config.json
  # write and creating EGL contexts simultaneously
  sleep $((w * 3))
  local offset=$((TASK_ID * WORKERS + w))
  local family g
  for family in ${FAMILIES}; do
    for ((g = offset; g < GROUPS_PER_FAMILY; g += STRIDE)); do
      "${PYTHON_BIN}" -m projectile_ball_catch_robocasa_kitchen_scenes_scripts.group_dataset_generation group \
        --family "${family}" \
        --group-index "${g}" \
        --dataset-root "${DATASET_ROOT}" \
        --phase-config "${PHASE_CONFIG}" \
        --width "${WIDTH}" \
        --height "${HEIGHT}" \
        --fps "${FPS}" \
        --duration "${DURATION}" \
        || echo "FAILED family=${family} group=${g} worker=${w}"
    done
  done
}

for ((w = 0; w < WORKERS; w++)); do
  run_worker "${w}" &
done
wait
