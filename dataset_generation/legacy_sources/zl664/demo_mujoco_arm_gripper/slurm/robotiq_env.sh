#!/bin/bash
set -euo pipefail

export PROJECT_DIR="/gpfs/radev/home/zl664/project/demo_mujoco_arm_gripper"
export PYTHON="/gpfs/radev/project/sous/zss8/dataset-generation/.venv/bin/python"

export ROBOCASA_ROOT="/gpfs/radev/project/sous/mzl7/robocasa"
export ROBOCASA_ASSETS_ROOT="/gpfs/radev/project/sous/mzl7/robocasa/robocasa/models/assets"
export ROBOSUITE_ROOT="${PROJECT_DIR}/third_party/robosuite"
export ROBOTWIN_1_0_MODELS="${PROJECT_DIR}/third_party/robotwin_1_0/models"
export ROBOTWIN_2_0_OBJECTS="${PROJECT_DIR}/third_party/robotwin_2_0/assets/objects"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${PROJECT_DIR}:${ROBOCASA_ROOT}:${ROBOSUITE_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
