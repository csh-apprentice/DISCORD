#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <img_dir> <out_root> [device] [visualize_flag]"
  echo "Example: $0 examples/my_scene outputs/taj_demo cuda:0 1"
  exit 1
fi

IMG_DIR="$1"
OUT_ROOT="$2"
DEVICE="${3:-cuda:0}"
VISUALIZE="${4:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

VIS_ARGS=()
if [[ "${VISUALIZE}" == "1" ]]; then
  VIS_ARGS+=(--visualize)
fi

STAGE1="${OUT_ROOT}/stage1_confidence"
STAGE2="${OUT_ROOT}/stage2_entropy"
STAGE3="${OUT_ROOT}/stage3_regions"
STAGE4="${OUT_ROOT}/stage4_final"

mkdir -p "${STAGE1}" "${STAGE2}" "${STAGE3}" "${STAGE4}"

"${PYTHON_BIN}" -m discord3d.pipeline.confidence \
  --img_dir "${IMG_DIR}" \
  --out_dir "${STAGE1}" \
  --device "${DEVICE}" \
  "${VIS_ARGS[@]}"

"${PYTHON_BIN}" -m discord3d.pipeline.entropy \
  --img_dir "${IMG_DIR}" \
  --conf_npy "${STAGE1}/confidence_depth.npy" \
  --out_dir "${STAGE2}" \
  --device "${DEVICE}" \
  "${VIS_ARGS[@]}"

"${PYTHON_BIN}" -m discord3d.pipeline.region_trust \
  --img_dir "${IMG_DIR}" \
  --conf_npy "${STAGE1}/confidence_depth.npy" \
  --entropy_act_npy "${STAGE2}/entropy_layer08_act.npy" \
  --tolerance_json "${STAGE2}/tolerance_summary.json" \
  --out_dir "${STAGE3}" \
  --trust_quantile 0.9 \
  "${VIS_ARGS[@]}"

"${PYTHON_BIN}" -m discord3d.pipeline.finalize \
  --img_dir "${IMG_DIR}" \
  --mask_region_npy "${STAGE3}/mask_region.npy" \
  --valid_masks_npy "${STAGE1}/valid_masks.npy" \
  --out_dir "${STAGE4}" \
  "${VIS_ARGS[@]}"

echo "Final masked images: ${STAGE4}/masked_images"
