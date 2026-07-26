#!/bin/bash
# Shared environment for Franka ball-catch rendering jobs.
# Sourced by the sbatch scripts. Uses the project venv + EGL headless GL.
set -euo pipefail

export PROJECT_DIR="/gpfs/radev/project/sous/zss8/dataset-generation"
export PKG_DIR="${PROJECT_DIR}/demo_mujoco_arm_gripper"
export PYTHON="${PROJECT_DIR}/.venv/bin/python"

# Headless rendering on a GPU node via EGL (libEGL is already on the system).
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${PKG_DIR}:${PYTHONPATH:-}"
# Keep BLAS from oversubscribing cores within a single episode.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
