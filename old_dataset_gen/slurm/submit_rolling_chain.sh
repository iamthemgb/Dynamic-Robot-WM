#!/bin/bash
# Submit the full rolling-dynamics chain with slurm dependencies:
#
#   p0 gen (3 family arrays) -> p0 finalize+validate (3)
#     -> p1 gen (3 family arrays) -> p1 finalize+validate (3)
#       -> cache index -> VAE encode (2 shards) -> t5 + oracle ceiling
#         -> phase-1 training (1.3B)
#
# Every link is --dependency=afterok, so a validator hard-gate failure
# (exit 1) stops everything downstream. Run FROM old_dataset_gen/:
#
#   bash slurm/submit_rolling_chain.sh
#
# DRY_RUN=1 prints the sbatch commands with fake job ids instead of
# submitting. Roots override via P0_ROOT / P1_ROOT.

set -euo pipefail
# command substitutions must propagate sbatch failures, or one rejected job
# silently poisons every downstream dependency with an empty id
shopt -s inherit_errexit

OLD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WM_ROOT="$(cd "${OLD_ROOT}/.." && pwd)"
PW_SLURM="${WM_ROOT}/mzl7/physics_wan/slurm"
P0_ROOT="${P0_ROOT:-${WM_ROOT}/mzl7/physics_wan/datasets/roll_groups_p0}"
P1_ROOT="${P1_ROOT:-${WM_ROOT}/mzl7/physics_wan/datasets/roll_groups_p1}"
FAMILIES=(rolling_layout38_style42 rolling_layout48_style41 rolling_layout51_style34)
ARM="${ARM:-wan21_t2v_1p3b}"
DRY_RUN="${DRY_RUN:-0}"
FAKE_ID_FILE="$(mktemp)"
echo 1000 > "${FAKE_ID_FILE}"
trap 'rm -f "${FAKE_ID_FILE}"' EXIT

cd "${OLD_ROOT}"
mkdir -p logs

submit() {
  # submit <name-for-log> <sbatch args...> ; echoes the job id
  local label="$1"
  shift
  if [ "${DRY_RUN}" = "1" ]; then
    # command substitution runs submit in a subshell, so the fake-id
    # counter must live in a file to survive across calls
    local fake_id
    fake_id=$(($(cat "${FAKE_ID_FILE}") + 1))
    echo "${fake_id}" > "${FAKE_ID_FILE}"
    echo "DRY [${fake_id}] sbatch $*" >&2
    echo "${fake_id}"
  else
    local id
    id="$(sbatch --parsable "$@")"
    echo "submitted ${label}: job ${id}" >&2
    echo "${id}"
  fi
}

dep() {
  # dep id1 id2 ... -> "afterok:id1:id2..."
  local IFS=:
  echo "afterok:$*"
}

# ---------------------------------------------------------------- P0 gen --
P0_VAL_IDS=()
for fam in "${FAMILIES[@]}"; do
  gen_id=$(submit "p0-gen-${fam}" \
    --job-name="roll-p0-gen-${fam}" \
    --export=ALL,FAMILY="${fam}",DATASET_ROOT="${P0_ROOT}",PHASE_CONFIG="${OLD_ROOT}/configs/rolling_groups_p0.json" \
    --array=0-9 \
    slurm/generate_rolling_group_array.sbatch)
  val_id=$(submit "p0-finalize-${fam}" \
    --job-name="roll-p0-fin-${fam}" \
    --dependency="$(dep "${gen_id}")" \
    --export=ALL,FAMILY="${fam}",DATASET_ROOT="${P0_ROOT}" \
    slurm/finalize_rolling_groups.sbatch)
  P0_VAL_IDS+=("${val_id}")
done

# ---------------------------------------------------------------- P1 gen --
P1_VAL_IDS=()
for fam in "${FAMILIES[@]}"; do
  gen_id=$(submit "p1-gen-${fam}" \
    --job-name="roll-p1-gen-${fam}" \
    --dependency="$(dep "${P0_VAL_IDS[@]}")" \
    --export=ALL,FAMILY="${fam}",DATASET_ROOT="${P1_ROOT}",PHASE_CONFIG="${OLD_ROOT}/configs/rolling_groups_p1.json" \
    --array=0-332%32 \
    slurm/generate_rolling_group_array.sbatch)
  val_id=$(submit "p1-finalize-${fam}" \
    --job-name="roll-p1-fin-${fam}" \
    --dependency="$(dep "${gen_id}")" \
    --export=ALL,FAMILY="${fam}",DATASET_ROOT="${P1_ROOT}" \
    slurm/finalize_rolling_groups.sbatch)
  P1_VAL_IDS+=("${val_id}")
done

# ------------------------------------------------------- cache -> train ---
index_id=$(submit "cache-index" \
  --dependency="$(dep "${P1_VAL_IDS[@]}")" \
  --export=ALL,DATASET_ROOT="${P1_ROOT}" \
  "${PW_SLURM}/roll_10_index.sbatch")

encode_id=$(submit "vae-encode" \
  --dependency="$(dep "${index_id}")" \
  --array=0-1 \
  "${PW_SLURM}/roll_11_encode.sbatch" wan21)

t5_id=$(submit "t5-cache" \
  --dependency="$(dep "${encode_id}")" \
  "${PW_SLURM}/roll_15_t5.sbatch")

ceiling_id=$(submit "oracle-ceiling" \
  --dependency="$(dep "${encode_id}")" \
  "${PW_SLURM}/roll_16_ceiling.sbatch")

train_id=$(submit "phase1-${ARM}" \
  --dependency="$(dep "${t5_id}" "${ceiling_id}")" \
  "${PW_SLURM}/roll_20_train.sbatch" "${ARM}")

echo ""
echo "chain submitted:"
echo "  p0 validate : ${P0_VAL_IDS[*]}"
echo "  p1 validate : ${P1_VAL_IDS[*]}"
echo "  index/encode: ${index_id} / ${encode_id}"
echo "  t5/ceiling  : ${t5_id} / ${ceiling_id}"
echo "  phase1      : ${train_id}"
