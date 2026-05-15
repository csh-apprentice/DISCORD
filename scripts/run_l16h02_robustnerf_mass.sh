#!/usr/bin/env bash
set -euo pipefail

VGGT_PYTHONPATH="${VGGT_PYTHONPATH:-/home/shihan/project/ConditionVGGT/vggt}"
export PYTHONPATH="${VGGT_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m discord3d.evaluation.robustnerf_l16h02_mass \
  --dataset_root /data/shihan/robustnerf \
  --out_dir outputs/experiments/l16h02_robustnerf_mass \
  --device "${DISCORD_DEVICE:-cuda:1}" \
  "$@"
