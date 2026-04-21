#!/usr/bin/env bash
set -euo pipefail

# Canonical launcher for exporting curated DISCORD example bundles from an
# evaluation summary. Override env vars when needed:
#
#   SUMMARY_JSON=outputs/eval/phototourism_full_nv5_t3_L8.summary.json \
#   IMG_ROOT=/data/shihan/phototourism \
#   OUT_DIR=examples/phototourism_nv5_t3 \
#   SELECTORS="colosseum_exterior:1 temple_nara_japan:0" \
#   PYTHON_BIN=/home/shihan/anaconda3/envs/CP/bin/python \
#   bash scripts/export_curated_examples.sh
#
# By default images are copied. Set COPY_MODE=symlink to export symlinks instead.

PYTHON_BIN="${PYTHON_BIN:-python3}"
SUMMARY_JSON="${SUMMARY_JSON:-outputs/eval/phototourism_full_nv5_t3_L8.summary.json}"
IMG_ROOT="${IMG_ROOT:-/data/shihan/phototourism}"
OUT_DIR="${OUT_DIR:-examples/phototourism_nv5_t3}"
SELECTORS="${SELECTORS:-}"
COPY_MODE="${COPY_MODE:-copy}"

ARGS=(
  --summary-json "${SUMMARY_JSON}"
  --img-root "${IMG_ROOT}"
  --out-dir "${OUT_DIR}"
)

if [[ "${COPY_MODE}" == "symlink" ]]; then
  ARGS+=(--symlink)
else
  ARGS+=(--copy)
fi

for selector in ${SELECTORS}; do
  ARGS+=(--select "${selector}")
done

echo "[export_examples] summary=${SUMMARY_JSON}"
echo "[export_examples] img_root=${IMG_ROOT}"
echo "[export_examples] out_dir=${OUT_DIR}"
if [[ -n "${SELECTORS}" ]]; then
  echo "[export_examples] selectors=${SELECTORS}"
fi

"${PYTHON_BIN}" discord3d/rendering/export_trial_bundles.py "${ARGS[@]}" "$@"
