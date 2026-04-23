# DISCORD

DISCORD is a post-hoc trust-segmentation pipeline for cluttered multi-view 3D reconstruction. It builds on top of VGGT and converts cross-view geometric disagreement into a binary trust mask that can be used both:

- in **2D**, to highlight image regions that are likely to belong to stable scene geometry, and
- in **3D**, to filter noisy geometry from feed-forward reconstructions.

The current public code layout is intentionally limited to the active method path and the paper-facing evaluation path.

## Quick Start

1. Install the Python dependencies from [requirements.txt](requirements.txt).
2. Make the VGGT-related dependencies available as described in [third_party/README.md](third_party/README.md).
3. Run the bundled smoke test on the curated examples:

```bash
bash scripts/smoke_test_examples.sh
```

If the smoke test passes, the fastest next checks are:

- `python demo.py` for the interactive Gradio demo
- `bash scripts/run_pipeline.sh <img_dir> <out_root>` for the staged CLI pipeline

## Repository Layout

```text
DISCORD/
├── demo.py
├── discord3d/
│   ├── pipeline/
│   ├── evaluation/
│   ├── rendering/
│   ├── third_party.py
│   └── vggt_support.py
├── scripts/
├── examples/
└── third_party/
```

### Core package

- [discord3d/pipeline](discord3d/pipeline): canonical DISCORD pipeline
- [discord3d/evaluation](discord3d/evaluation): benchmark and sweep utilities
- [discord3d/rendering](discord3d/rendering): figure and camera-pose rendering helpers

### Public entry points

- [demo.py](demo.py): Gradio demo for `Baseline VGGT`, `Floor-only`, and `DISCORD`
- [scripts/run_pipeline.sh](scripts/run_pipeline.sh): one-shot staged pipeline runner
- [scripts/run_phototourism_sweeps.sh](scripts/run_phototourism_sweeps.sh): paper-facing sweep launcher
- [scripts/export_curated_examples.sh](scripts/export_curated_examples.sh): export Gradio-ready example bundles from an evaluation summary
- [scripts/smoke_test_examples.sh](scripts/smoke_test_examples.sh): lightweight regression check on bundled examples

## Current Method

The locked paper version uses:

1. floor-only confidence preprocessing
2. activated cross-view attention entropy
3. bridge-enabled confidence-region partitioning
4. quantile-based trust voting (`q = 0.9`)
5. component-aware hole filling

In the public demo and evaluation code, this corresponds to the bridge-enabled, quantile-trust version of DISCORD with the paper default `q = 0.9`.

## External Dependencies

DISCORD depends on [VGGT](https://github.com/facebookresearch/vggt), which is not vendored here. Clone it into `third_party/` and install its requirements:

```bash
git clone https://github.com/facebookresearch/vggt.git third_party/vggt
pip install -r third_party/vggt/requirements.txt
pip install -r third_party/vggt/requirements_demo.txt
```

See [third_party/README.md](third_party/README.md) for the expected layout and the fallback path used in our research workspace.

## Hardware Notes

- The Gradio demo and the full Phototourism sweeps are intended for a CUDA-capable GPU.
- The bundled smoke test automatically skips the heavy model run when CUDA is unavailable, so it can still serve as a lightweight regression check on CPU-only machines.

## Dataset Path Defaults

Some evaluation and export helpers still default to our local research layout, for example:

- `/data/shihan/phototourism`
- `/data/shihan/robustnerf`
- `/data/shihan/llff_full`

That is acceptable for the current research snapshot, but external users should treat these as defaults to override, either by:

- passing CLI flags such as `--img-root` and `--colmap-root`, or
- overriding environment variables in the provided shell wrappers

## What Is Not Included Here

This cleaned repo intentionally excludes:

- historical research scripts
- temporary Gradio outputs
- paper drafting assets
- large intermediate research outputs
- full datasets and heavyweight run folders

Those remain in the research workspace and are not part of the public-facing method path.

## Recommended Workflow

### Run the demo

```bash
python3 demo.py
```

### Run the staged pipeline on a folder of images

```bash
bash scripts/run_pipeline.sh <img_dir> <out_root>
```

### Run the Phototourism sweeps

```bash
bash scripts/run_phototourism_sweeps.sh
python3 discord3d/evaluation/summarize_phototourism.py
```

### Export curated examples

```bash
bash scripts/export_curated_examples.sh
```

For a custom dataset root or a smaller selection:

```bash
IMG_ROOT=/path/to/phototourism \
SELECTORS="colosseum_exterior:1 temple_nara_japan:0" \
bash scripts/export_curated_examples.sh
```

### Run the bundled smoke test

```bash
bash scripts/smoke_test_examples.sh
```
