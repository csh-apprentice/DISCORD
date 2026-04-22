# Examples

This folder contains small curated bundles used by the DISCORD demo and smoke tests.

These examples serve two roles:

- they provide immediately loadable Gradio bundles, and
- they act as a lightweight regression target for [scripts/smoke_test_examples.sh](../scripts/smoke_test_examples.sh)

The current repo keeps only a lightweight subset of the internal Phototourism examples:

- `brandenburg_gate__trial_00`
- `buckingham_palace__trial_00`
- `colosseum_exterior__trial_01`
- `temple_nara_japan__trial_00`

Each bundle includes:

- `images/`
- `bundle_meta.json`
- `preview_grid.jpg`

We intentionally exclude cached reconstructions and large intermediate outputs.

Suggested use:

- keep only a very small number of lightweight example bundles here, or
- generate additional bundles locally with:

```bash
bash scripts/export_curated_examples.sh
```

The export wrapper inherits the current research defaults, so on a different machine you will usually want to override at least:

```bash
IMG_ROOT=/path/to/phototourism \
SUMMARY_JSON=outputs/eval/phototourism_full_nv5_t3_L8.summary.json \
bash scripts/export_curated_examples.sh
```
