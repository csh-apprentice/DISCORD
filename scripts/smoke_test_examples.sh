#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/shihan/anaconda3/envs/CP/bin/python}"
DEVICE="${DEVICE:-}"
OUT_DIR="${OUT_DIR:-outputs/smoke}"
export PYTHON_BIN

mkdir -p "${OUT_DIR}"

if [[ -z "${DEVICE}" ]]; then
  if "${PYTHON_BIN}" - <<'PY'
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    DEVICE="cuda:0"
  else
    DEVICE="cpu"
  fi
fi

echo "[smoke] validating curated example bundles"
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
root = Path("examples/phototourism_nv5_t3")
expected = [
    "brandenburg_gate__trial_00",
    "buckingham_palace__trial_00",
    "colosseum_exterior__trial_01",
    "temple_nara_japan__trial_00",
]
for name in expected:
    bundle = root / name
    assert bundle.exists(), f"Missing bundle: {bundle}"
    assert (bundle / "bundle_meta.json").exists(), f"Missing bundle_meta.json in {bundle}"
    assert (bundle / "preview_grid.jpg").exists(), f"Missing preview_grid.jpg in {bundle}"
    images = sorted((bundle / "images").glob("*"))
    assert len(images) == 5, f"Expected 5 images in {bundle}, found {len(images)}"
print("Validated curated bundles.")
PY

echo "[smoke] compiling python modules"
"${PYTHON_BIN}" -m py_compile $(find . -name '*.py' | tr '\n' ' ')

if [[ "${DEVICE}" == "cpu" ]]; then
  echo "[smoke] no CUDA device available; skipping model-level smoke run"
  exit 0
fi

echo "[smoke] running staged pipeline on colosseum_exterior__trial_01"
bash scripts/run_pipeline.sh \
  examples/phototourism_nv5_t3/colosseum_exterior__trial_01/images \
  "${OUT_DIR}/pipeline_colosseum" \
  "${DEVICE}" \
  0

echo "[smoke] evaluating bundled examples"
"${PYTHON_BIN}" -m discord3d.evaluation.eval_phototourism \
  --dataset phototourism \
  --bundle_root examples/phototourism_nv5_t3 \
  --all_scenes \
  --n_trials 1 \
  --n_views 5 \
  --device "${DEVICE}" \
  --entropy_layer 8 \
  --split_region_bridges \
  --trust_stat quantile \
  --trust_quantile 0.9 \
  --gmm_max_fit_points 0 \
  --out_csv "${OUT_DIR}/phototourism_examples_eval.csv"

echo "[smoke] done"
