#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/home/bricsbot/catkin_ws/src/hofa_policy/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec "${HOFA_POLICY_PYTHON:-/home/bricsbot/miniconda3/envs/oceangym/bin/python}" \
  /home/bricsbot/catkin_ws/src/hofa_policy/scripts/hofa_policy_controller_node.py "$@"
