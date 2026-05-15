#!/usr/bin/env bash
set -euo pipefail

VGGT_PYTHONPATH="${VGGT_PYTHONPATH:-/home/shihan/project/ConditionVGGT/vggt}"
export PYTHONPATH="${VGGT_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m discord3d.evaluation.phototourism_l16h02_entropy \
  --bundle_root /home/shihan/project/DISCORD/datasets/examples/phototourism_nv5_t3 \
  --out_dir outputs/experiments/l16h02_phototourism_entropy \
  --device "${DISCORD_DEVICE:-cuda:1}" \
  "$@"
