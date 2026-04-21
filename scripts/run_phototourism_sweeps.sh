#!/usr/bin/env bash
set -euo pipefail

# Canonical full-sweep launcher for the paper-facing Phototourism evaluation.
# By default this runs the exact final method on the full benchmark with
# N in {3,5,8}. Override env vars when needed:
#
#   LAYERS="4 8 16" VIEWS="3 5 8" TRIALS=3 \
#   PYTHON_BIN=/home/shihan/anaconda3/envs/CP/bin/python \
#   bash scripts/run_phototourism_sweeps.sh
#
# Extra CLI args passed to this script are forwarded to the eval driver.

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
TRIALS="${TRIALS:-3}"
LAYERS="${LAYERS:-8}"
VIEWS="${VIEWS:-3 5 8}"
OUT_DIR="${OUT_DIR:-outputs/eval}"

mkdir -p "${OUT_DIR}"

for layer in ${LAYERS}; do
  for n_views in ${VIEWS}; do
    out_csv="${OUT_DIR}/phototourism_full_nv${n_views}_t${TRIALS}_L${layer}.csv"
    echo "[final_eval] layer=${layer} n_views=${n_views} -> ${out_csv}"
    "${PYTHON_BIN}" discord3d/evaluation/eval_phototourism.py \
      --dataset phototourism \
      --all_scenes \
      --n_views "${n_views}" \
      --n_trials "${TRIALS}" \
      --device "${DEVICE}" \
      --entropy_layer "${layer}" \
      --split_region_bridges \
      --trust_stat quantile \
      --trust_quantile 0.9 \
      --gmm_max_fit_points 0 \
      --out_csv "${out_csv}" \
      "$@"
  done
done
