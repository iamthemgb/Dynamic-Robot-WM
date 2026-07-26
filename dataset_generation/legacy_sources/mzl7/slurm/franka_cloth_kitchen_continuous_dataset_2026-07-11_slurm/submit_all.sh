#!/bin/bash
# Full continuous-physics dataset: 4 tasks x 501 episodes (167 seeds x 3
# variants) = 2004 episodes. Run after the pilot bundle checks out.
set -euo pipefail
cd "$(dirname "$0")"
for task in poke_cloth lift_corner_release fold_edge_fixed_line dual_franka_tshirt_fold_box; do
  sbatch --job-name="cont_${task}" --array=0-500 \
    --export=ALL,TASK="${task}" generate_task_array.sbatch
done
