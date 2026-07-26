#!/bin/bash
set -euo pipefail

# zss8 copy of the Robotiq catch pipeline. Code + data live under zss8 dirs only.
# Large asset trees (robocasa, robosuite, robotwin) are referenced READ-ONLY in
# place and are never copied or modified.
export PROJECT_DIR="/gpfs/radev/project/sous/zss8/dataset-generation/robotiq_arm_gripper"
export PYTHON="/gpfs/radev/project/sous/zss8/dataset-generation/.venv/bin/python"

export ROBOCASA_ROOT="/gpfs/radev/project/sous/mzl7/robocasa"
export ROBOCASA_ASSETS_ROOT="/gpfs/radev/project/sous/mzl7/robocasa/robocasa/models/assets"

# read-only reference to zl664's third_party asset checkout (33GB; not copied)
export TP_ROOT="/gpfs/radev/project/sous/zl664/demo_mujoco_arm_gripper/third_party"
export ROBOSUITE_ROOT="${TP_ROOT}/robosuite"
export ROBOTWIN_1_0_MODELS="${TP_ROOT}/robotwin_1_0/models"
export ROBOTWIN_2_0_OBJECTS="${TP_ROOT}/robotwin_2_0/assets/objects"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${PROJECT_DIR}:${ROBOCASA_ROOT}:${ROBOSUITE_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
