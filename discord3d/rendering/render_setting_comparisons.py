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

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from discord3d.third_party import setup_vggt_paths  # noqa: E402

setup_vggt_paths()

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from discord3d.pipeline.common import (  # noqa: E402
    DTYPE,
    list_images,
    load_crop_images_with_valid_masks,
    maybe_autocast,
)
from discord3d.pipeline.runtime import (  # noqa: E402
    apply_final_masks_to_predictions,
    compute_final_outputs,
)
from discord3d.rendering.render_curated_comparisons import render_run  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Render old-vs-new entropy setting comparisons for curated bundles.")
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
        "--out-dir",
        type=Path,
        default=Path("outputs/render/paper_3d_setting_compare"),
    )
    ap.add_argument("--old-run-name", default="entropy_final_L8")
    ap.add_argument("--new-run-name", default="entropy_final_L8_rb1_p90")
    ap.add_argument("--entropy-layer", type=int, default=8)
    ap.add_argument("--trust-stat", choices=["mean", "quantile"], default="quantile")
    ap.add_argument("--trust-quantile", type=float, default=0.9)
    ap.add_argument("--split-region-bridges", action="store_true", default=True)
    ap.add_argument("--prediction-mode", default="Depthmap and Camera Branch")
    ap.add_argument("--background", choices=["black", "white"], default="black")
    ap.add_argument("--point-size-scale", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def _postprocess_predictions(predictions: dict, image_hw: tuple[int, int]) -> dict:
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], image_hw)
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().float().numpy().squeeze(0)

    predictions["pose_enc_list"] = None
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
    )
    return predictions


def _load_model(device: str):
    model = VGGT.from_pretrained("facebook/VGGT-1B", local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    return model


def _save_predictions_npz(run_dir: Path, predictions: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "predictions.npz", **{k: v for k, v in predictions.items() if isinstance(v, np.ndarray)})


def ensure_new_entropy_run(
    model,
    bundle_dir: Path,
    run_name: str,
    entropy_layer: int,
    trust_stat: str,
    trust_quantile: float,
    split_region_bridges: bool,
    device: str,
) -> Path:
    run_dir = bundle_dir / run_name
    preds_path = run_dir / "predictions.npz"
    if preds_path.exists():
        return run_dir

    image_paths = list_images(bundle_dir / "images")
    names = [p.name for p in image_paths]
    imgs, valid_masks = load_crop_images_with_valid_masks(image_paths)
    image_hw = tuple(int(d) for d in imgs.shape[-2:])

    with torch.inference_mode():
        with maybe_autocast(device):
            model_out = model(imgs.to(device).unsqueeze(0))
    predictions = model_out[0] if isinstance(model_out, tuple) else model_out
    predictions = _postprocess_predictions(predictions, image_hw)
    predictions["image_files"] = np.array(names)

    outputs = compute_final_outputs(
        model=model,
        imgs=imgs,
        valid_masks=valid_masks,
        conf=predictions["depth_conf"].astype(np.float32),
        device=device,
        entropy_layer=int(entropy_layer),
        split_region_bridges=bool(split_region_bridges),
        trust_stat=str(trust_stat),
        trust_quantile=float(trust_quantile),
    )
    predictions = apply_final_masks_to_predictions(predictions, outputs["final_mask"])
    _save_predictions_npz(run_dir, predictions)

    meta = {
        "entropy_layer": int(entropy_layer),
        "trust_stat": str(trust_stat),
        "trust_quantile": float(trust_quantile),
        "split_region_bridges": bool(split_region_bridges),
        "views": outputs["meta"],
    }
    (run_dir / "final_pipeline_meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def _add_header(image: Image.Image, title: str) -> Image.Image:
    pad = 34
    canvas = Image.new("RGB", (image.width, image.height + pad), (255, 255, 255))
    canvas.paste(image, (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(20, 20, 20))
    return canvas


def make_setting_sheet(rows: list[tuple[str, Path, Path]], out_path: Path):
    rendered = []
    target_width = 520
    for scene_label, old_path, new_path in rows:
        old_img = Image.open(old_path).convert("RGB")
        new_img = Image.open(new_path).convert("RGB")
        scale = target_width / float(old_img.width)
        new_size = (target_width, int(round(old_img.height * scale)))
        old_img = old_img.resize(new_size, Image.Resampling.BICUBIC)
        new_img = new_img.resize(new_size, Image.Resampling.BICUBIC)
        old_img = _add_header(old_img, f"{scene_label}  |  old (mean, no-bridge)")
        new_img = _add_header(new_img, f"{scene_label}  |  new (p90, bridge)")
        rendered.append((old_img, new_img))

    gap = 16
    width = rendered[0][0].width * 2 + gap * 3
    height = sum(pair[0].height for pair in rendered) + gap * (len(rendered) + 1)
    canvas = Image.new("RGB", (width, height), (248, 248, 248))

    y = gap
    for old_img, new_img in rendered:
        canvas.paste(old_img, (gap, y))
        canvas.paste(new_img, (gap * 2 + old_img.width, y))
        y += old_img.height + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    device = args.device
    model = _load_model(device)

    rows = []
    out_manifest = []

    for item in manifest:
        bundle_name = item["bundle"]
        bundle_dir = args.bundle_root / bundle_name
        camera_json = Path(item["camera_json"])
        old_dir = bundle_dir / args.old_run_name
        if not (old_dir / "predictions.npz").exists():
            raise FileNotFoundError(f"Missing old run predictions: {old_dir / 'predictions.npz'}")

        new_dir = ensure_new_entropy_run(
            model=model,
            bundle_dir=bundle_dir,
            run_name=args.new_run_name,
            entropy_layer=args.entropy_layer,
            trust_stat=args.trust_stat,
            trust_quantile=args.trust_quantile,
            split_region_bridges=args.split_region_bridges,
            device=device,
        )

        scene_slug = bundle_name
        scene_label = bundle_name.replace("__trial_", " / trial ")
        old_out = args.out_dir / f"{scene_slug}__old.png"
        new_out = args.out_dir / f"{scene_slug}__new.png"

        render_run(old_dir, camera_json, old_out, args.prediction_mode, args.background, args.point_size_scale)
        render_run(new_dir, camera_json, new_out, args.prediction_mode, args.background, args.point_size_scale)

        rows.append((scene_label, old_out, new_out))
        out_manifest.append(
            {
                "bundle": bundle_name,
                "camera_json": str(camera_json),
                "old_run_dir": str(old_dir),
                "new_run_dir": str(new_dir),
                "old_render": str(old_out),
                "new_render": str(new_out),
                "new_config": {
                    "entropy_layer": int(args.entropy_layer),
                    "trust_stat": str(args.trust_stat),
                    "trust_quantile": float(args.trust_quantile),
                    "split_region_bridges": bool(args.split_region_bridges),
                },
            }
        )

    make_setting_sheet(rows, args.out_dir / "comparison_sheet.png")
    (args.out_dir / "manifest.json").write_text(json.dumps(out_manifest, indent=2))
    print(f"Saved {len(rows) * 2} individual renders + 1 sheet to {args.out_dir}")


if __name__ == "__main__":
    main()
