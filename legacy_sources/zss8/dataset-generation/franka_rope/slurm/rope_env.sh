#!/bin/bash
set -euo pipefail

# Franka-rope deformable dataset (zss8). Code + data live under zss8 dirs only.
# The Franka Panda MJCF is referenced READ-ONLY from zl664's Menagerie checkout
# (resolved inside scene_builder.find_panda_xml); nothing there is copied/edited.
export PROJECT_DIR="/gpfs/radev/project/sous/zss8/dataset-generation"
export PYTHON="/gpfs/radev/project/sous/zss8/dataset-generation/.venv/bin/python"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
