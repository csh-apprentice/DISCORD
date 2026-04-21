#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-discord")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from discord3d.rendering.render_curated_comparisons import ensure_baseline, ensure_floor_only, render_run  # noqa: E402
from discord3d.rendering.render_setting_comparisons import ensure_new_entropy_run, _load_model  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Render baseline vs floor-only vs ours comparisons for curated trial bundles.")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/render/paper_3d_comparisons/manifest.json"),
    )
    ap.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("examples/phototourism_nv5_t3"),
    )
    ap.add_argument(
        "--bundles",
        nargs="+",
        required=True,
        help="Bundle names to include, in row order.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/render/paper_3d_threeway"),
    )
    ap.add_argument("--ours-run-name", default="entropy_final_L8_rb1_bk3_q90_gf0")
    ap.add_argument("--entropy-layer", type=int, default=8)
    ap.add_argument("--trust-stat", choices=["mean", "quantile"], default="quantile")
    ap.add_argument("--trust-quantile", type=float, default=0.9)
    ap.add_argument("--split-region-bridges", action="store_true", default=True)
    ap.add_argument("--prediction-mode", default="Depthmap and Camera Branch")
    ap.add_argument("--background", choices=["black", "white"], default="black")
    ap.add_argument("--point-size-scale", type=float, default=1.0)
    ap.add_argument("--conf-eps", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def _add_header(image: Image.Image, title: str) -> Image.Image:
    pad = 34
    canvas = Image.new("RGB", (image.width, image.height + pad), (255, 255, 255))
    canvas.paste(image, (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(20, 20, 20))
    return canvas


def make_threeway_sheet(rows: list[tuple[str, Path, Path, Path]], out_path: Path):
    target_width = 360
    rendered = []
    for scene_label, base_path, floor_path, ours_path in rows:
        imgs = []
        for title, path in [
            ("Baseline VGGT", base_path),
            ("Floor-only", floor_path),
            ("Ours", ours_path),
        ]:
            img = Image.open(path).convert("RGB")
            scale = target_width / float(img.width)
            new_size = (target_width, int(round(img.height * scale)))
            img = img.resize(new_size, Image.Resampling.BICUBIC)
            imgs.append(_add_header(img, f"{scene_label}  |  {title}"))
        rendered.append(tuple(imgs))

    gap = 16
    row_h = rendered[0][0].height
    col_w = rendered[0][0].width
    width = col_w * 3 + gap * 4
    height = row_h * len(rendered) + gap * (len(rendered) + 1)
    canvas = Image.new("RGB", (width, height), (248, 248, 248))

    y = gap
    for row in rendered:
        x = gap
        for img in row:
            canvas.paste(img, (x, y))
            x += col_w + gap
        y += row_h + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    args = parse_args()
    manifest = {item["bundle"]: item for item in json.loads(args.manifest.read_text())}
    model = _load_model(args.device)

    rows = []
    out_manifest = []

    for bundle_name in args.bundles:
        if bundle_name not in manifest:
            raise KeyError(f"Missing bundle in manifest: {bundle_name}")
        item = manifest[bundle_name]
        bundle_dir = args.bundle_root / bundle_name
        camera_json = Path(item["camera_json"])
        scene_label = bundle_name.replace("__trial_", " / trial ")
        scene_slug = bundle_name

        baseline_dir = ensure_baseline(bundle_dir)
        floor_dir = ensure_floor_only(bundle_dir, args.conf_eps)
        ours_dir = ensure_new_entropy_run(
            model=model,
            bundle_dir=bundle_dir,
            run_name=args.ours_run_name,
            entropy_layer=args.entropy_layer,
            trust_stat=args.trust_stat,
            trust_quantile=args.trust_quantile,
            split_region_bridges=args.split_region_bridges,
            device=args.device,
        )

        baseline_out = args.out_dir / f"{scene_slug}__baseline.png"
        floor_out = args.out_dir / f"{scene_slug}__floor_only.png"
        ours_out = args.out_dir / f"{scene_slug}__ours.png"

        render_run(baseline_dir, camera_json, baseline_out, args.prediction_mode, args.background, args.point_size_scale)
        render_run(floor_dir, camera_json, floor_out, args.prediction_mode, args.background, args.point_size_scale)
        render_run(ours_dir, camera_json, ours_out, args.prediction_mode, args.background, args.point_size_scale)

        rows.append((scene_label, baseline_out, floor_out, ours_out))
        out_manifest.append(
            {
                "bundle": bundle_name,
                "camera_json": str(camera_json),
                "baseline_run_dir": str(baseline_dir),
                "floor_run_dir": str(floor_dir),
                "ours_run_dir": str(ours_dir),
                "baseline_render": str(baseline_out),
                "floor_render": str(floor_out),
                "ours_render": str(ours_out),
            }
        )

    make_threeway_sheet(rows, args.out_dir / "comparison_sheet_3x3.png")
    (args.out_dir / "manifest.json").write_text(json.dumps(out_manifest, indent=2))
    print(f"Saved {len(rows) * 3} individual renders + 1 three-way sheet to {args.out_dir}")


if __name__ == "__main__":
    main()
