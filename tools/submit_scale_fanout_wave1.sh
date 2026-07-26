#!/usr/bin/env bash
# Wave-1 fan-out: full generation for the five leaves that passed the
# calibration gate (hard validity >= 95%): F1a, F3b, P0a, P0c, P0d.
#
# Per leaf: one prepare array (one task per block), then per block a
# 16-shard A40 generate array (afterok on that block's prepare task) and
# a finalize job (afterok on the generate array).  Failed upstream deps
# kill downstream jobs instead of leaving DependencyNeverSatisfied
# zombies.  All job IDs land in outputs/scale/fanout_wave1_jobs.tsv.
set -euo pipefail

ROOT=/gpfs/radev/scratch/sous/mzl7/dynamic_rollouts
REPO=/gpfs/radev/scratch/sous/mzl7/dataset_generation_fork
MANIFEST="$REPO/outputs/scale/fanout_wave1_jobs.tsv"
cd "$REPO"
mkdir -p outputs/scale

if [ -s "$MANIFEST" ]; then
  echo "manifest $MANIFEST already exists; refusing to double-submit" >&2
  exit 1
fi

declare -A BLOCKS=( [F1a]=28 [F3b]=36 [P0a]=7 [P0c]=5 [P0d]=3 )
printf 'leaf\tstage\tjob_id\tunit\n' > "$MANIFEST"

for LEAF in F1a F3b P0a P0c P0d; do
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
