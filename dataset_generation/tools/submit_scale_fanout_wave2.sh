#!/usr/bin/env bash
# Wave-2 fan-out: P0b (7 blocks) and F2c (31 blocks), which passed the
# calibration gate in round 2 (block 9001: P0b 0.99, F2c 0.95).
# Same chaining as wave 1; job IDs land in outputs/scale/fanout_wave2_jobs.tsv.
set -euo pipefail

ROOT=/gpfs/radev/scratch/sous/mzl7/dynamic_rollouts
REPO=/gpfs/radev/scratch/sous/mzl7/dataset_generation_fork
MANIFEST="$REPO/outputs/scale/fanout_wave2_jobs.tsv"
cd "$REPO"

if [ -s "$MANIFEST" ]; then
  echo "manifest $MANIFEST already exists; refusing to double-submit" >&2
  exit 1
fi

declare -A BLOCKS=( [P0b]=7 [F2c]=31 )
printf 'leaf\tstage\tjob_id\tunit\n' > "$MANIFEST"

for LEAF in P0b F2c; do
  N=${BLOCKS[$LEAF]}
  PREP=$(sbatch --parsable --array=0-$((N-1)) slurm/scale_prepare.sbatch "$LEAF" "$ROOT")
  printf '%s\tprepare\t%s\tarray 0-%d\n' "$LEAF" "$PREP" "$((N-1))" >> "$MANIFEST"
  for ((i = 0; i < N; i++)); do
    BLOCK=$(printf 'block-%04d' "$i")
    BLOCKDIR="$ROOT/$LEAF/$BLOCK"
    GEN=$(sbatch --parsable --kill-on-invalid-dep=yes \
      --dependency="afterok:${PREP}_${i}" --array=0-15%8 \
      slurm/scale_generate_a40.sbatch "$BLOCKDIR")
    FIN=$(sbatch --parsable --kill-on-invalid-dep=yes \
      --dependency="afterok:${GEN}" \
      slurm/scale_finalize.sbatch "$BLOCKDIR")
    printf '%s\tgenerate\t%s\t%s\n' "$LEAF" "$GEN" "$BLOCK" >> "$MANIFEST"
    printf '%s\tfinalize\t%s\t%s\n' "$LEAF" "$FIN" "$BLOCK" >> "$MANIFEST"
  done
  echo "$LEAF: prepare=$PREP, $N blocks chained"
done

echo "manifest: $MANIFEST"
