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

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from demo import gradio_reconstruct
from discord3d.pipeline.common import load_crop_images_with_valid_masks
from discord3d.rendering.render_saved_view import load_camera_pose, load_predictions, render_scene, predictions_to_glb


def parse_args():
    ap = argparse.ArgumentParser(description="Render floor-only vs ours comparisons for curated trial bundles.")
    ap.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("examples/phototourism_nv5_t3"),
    )
    ap.add_argument(
        "--bundles",
        nargs="+",
        required=True,
        help="Bundle folder names under bundle-root.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/render/paper_3d_comparisons"),
    )
    ap.add_argument("--prediction-mode", default="Depthmap and Camera Branch")
    ap.add_argument("--background", choices=["black", "white"], default="black")
    ap.add_argument("--point-size-scale", type=float, default=1.0)
    ap.add_argument("--conf-eps", type=float, default=1e-5)
    return ap.parse_args()


def sorted_image_paths(bundle_dir: Path) -> list[Path]:
    return sorted([p for p in (bundle_dir / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])


def ensure_baseline(bundle_dir: Path) -> Path:
    baseline_dir = bundle_dir / "baseline"
    preds_path = baseline_dir / "predictions.npz"
    if preds_path.exists():
        return baseline_dir

    result = gradio_reconstruct(
        str(bundle_dir),
        "Baseline VGGT",
        [],
        "16-23",
        "8",
        3,
        0.90,
        True,
        "suppress",
        0.5,
        "SVD",
        1.0,
        8,
        0.5,
        0.5,
        50.0,
        "All",
        False,
        False,
        False,
        False,
        "Depthmap and Camera Branch",
    )
    return Path(result[-1])


def ensure_floor_only(bundle_dir: Path, conf_eps: float) -> Path:
    floor_dir = bundle_dir / "floor_only"
    preds_out = floor_dir / "predictions.npz"
    if preds_out.exists():
        return floor_dir

    baseline_dir = ensure_baseline(bundle_dir)
    pred = np.load(baseline_dir / "predictions.npz", allow_pickle=True)
    preds = {k: np.array(pred[k]) for k in pred.files}

    image_paths = sorted_image_paths(bundle_dir)
    _, valid_masks = load_crop_images_with_valid_masks(image_paths)
    valid = valid_masks.numpy().astype(bool)

    if "depth_conf" in preds:
        conf = preds["depth_conf"].astype(np.float32)
    elif "world_points_conf" in preds:
        conf = preds["world_points_conf"].astype(np.float32)
    else:
        raise RuntimeError("No confidence map found in baseline predictions")

    mask = np.zeros_like(valid, dtype=np.float32)
    for i in range(conf.shape[0]):
        vals = conf[i][valid[i]]
        thr = float(vals.min()) + float(conf_eps)
        mask[i] = (valid[i] & (conf[i] > thr)).astype(np.float32)

    out = dict(preds)
    if "world_points_conf" in out:
        out["world_points_conf"] = out["world_points_conf"].astype(np.float32) * mask
    if "depth_conf" in out:
        out["depth_conf"] = out["depth_conf"].astype(np.float32) * mask
    out["final_mask"] = mask.astype(np.uint8)

    floor_dir.mkdir(parents=True, exist_ok=True)
    np.savez(preds_out, **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
    meta = {"mean_keep_frac": float(mask.sum() / valid.sum())}
    (floor_dir / "floor_only_meta.json").write_text(json.dumps(meta, indent=2))
    return floor_dir


def latest_camera_json(bundle_dir: Path) -> Path:
    snaps = sorted((bundle_dir / "entropy_final_L8" / "camera_snapshots").glob("*.json"))
    if not snaps:
        raise FileNotFoundError(f"No camera snapshots found for {bundle_dir}")
    return snaps[-1]


def render_run(run_dir: Path, camera_json: Path, out_path: Path, prediction_mode: str, background: str, point_size_scale: float):
    predictions = load_predictions(run_dir)
    pose = load_camera_pose(camera_json)
    width = int(pose.get("width") or 1280)
    height = int(pose.get("height") or 720)
    scene = predictions_to_glb(
        predictions,
        conf_thres=50.0,
        filter_by_frames="All",
        mask_black_bg=False,
        mask_white_bg=False,
        show_cam=False,
        mask_sky=False,
        target_dir=str(run_dir.parent),
        prediction_mode=prediction_mode,
    )
    image = render_scene(
        scene,
        pose=pose,
        width=width,
        height=height,
        show_cam=False,
        background=background,
        point_size_scale=point_size_scale,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return out_path


def add_header(image: Image.Image, title: str) -> Image.Image:
    pad = 34
    canvas = Image.new("RGB", (image.width, image.height + pad), (255, 255, 255))
    canvas.paste(image, (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(20, 20, 20))
    return canvas


def make_sheet(rows: list[tuple[str, Path, Path]], out_path: Path):
    rendered = []
    target_width = 520
    for scene_label, floor_path, ours_path in rows:
        floor_img = Image.open(floor_path).convert("RGB")
        ours_img = Image.open(ours_path).convert("RGB")
        scale = target_width / float(floor_img.width)
        new_size = (target_width, int(round(floor_img.height * scale)))
        floor_img = floor_img.resize(new_size, Image.Resampling.BICUBIC)
        ours_img = ours_img.resize(new_size, Image.Resampling.BICUBIC)
        floor_img = add_header(floor_img, f"{scene_label}  |  floor-only")
        ours_img = add_header(ours_img, f"{scene_label}  |  ours")
        rendered.append((floor_img, ours_img))

    gap = 16
    width = rendered[0][0].width * 2 + gap * 3
    height = sum(pair[0].height for pair in rendered) + gap * (len(rendered) + 1)
    canvas = Image.new("RGB", (width, height), (248, 248, 248))

    y = gap
    for floor_img, ours_img in rendered:
        canvas.paste(floor_img, (gap, y))
        canvas.paste(ours_img, (gap * 2 + floor_img.width, y))
        y += floor_img.height + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    args = parse_args()
    rows = []
    manifest = []

    for bundle_name in args.bundles:
        bundle_dir = args.bundle_root / bundle_name
        if not bundle_dir.exists():
            raise FileNotFoundError(bundle_dir)

        camera_json = latest_camera_json(bundle_dir)
        floor_dir = ensure_floor_only(bundle_dir, args.conf_eps)
        ours_dir = bundle_dir / "entropy_final_L8"

        scene_label = bundle_name.replace("__trial_", " / trial ")
        scene_slug = bundle_name

        floor_out = args.out_dir / f"{scene_slug}__floor_only.png"
        ours_out = args.out_dir / f"{scene_slug}__ours.png"

        render_run(floor_dir, camera_json, floor_out, args.prediction_mode, args.background, args.point_size_scale)
        render_run(ours_dir, camera_json, ours_out, args.prediction_mode, args.background, args.point_size_scale)

        rows.append((scene_label, floor_out, ours_out))
        manifest.append(
            {
                "bundle": bundle_name,
                "camera_json": str(camera_json),
                "floor_only": str(floor_out),
                "ours": str(ours_out),
            }
        )

    make_sheet(rows, args.out_dir / "comparison_sheet.png")
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(rows) * 2} individual renders + 1 sheet to {args.out_dir}")


if __name__ == "__main__":
    main()
