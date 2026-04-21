#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    ap = argparse.ArgumentParser(description="Export evaluation trials into Gradio-loadable image bundles.")
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/eval/phototourism_full_nv5_t3_L8.summary.json"),
        help="Evaluation summary json containing per-trial sample stems.",
    )
    ap.add_argument(
        "--img-root",
        type=Path,
        default=Path("/data/shihan/phototourism"),
        help="Phototourism dataset root.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("examples/phototourism_nv5_t3"),
        help="Output root for exported trial bundles.",
    )
    ap.add_argument(
        "--select",
        action="append",
        default=[],
        metavar="SCENE:TRIAL",
        help="Optional explicit selectors, e.g. colosseum_exterior:2 . Can be repeated.",
    )
    ap.add_argument(
        "--copy",
        dest="copy",
        action="store_true",
        default=True,
        help="Copy images into the exported bundle (default).",
    )
    ap.add_argument(
        "--symlink",
        dest="copy",
        action="store_false",
        help="Use symlinks instead of copies.",
    )
    return ap.parse_args()


def normalize_selectors(values: list[str]) -> set[tuple[str, int]]:
    selectors: set[tuple[str, int]] = set()
    for value in values:
        if ":" not in value:
            raise ValueError(f"Invalid selector '{value}'. Expected SCENE:TRIAL.")
        scene, trial = value.rsplit(":", 1)
        selectors.add((scene.strip(), int(trial)))
    return selectors


def find_image(scene_dir: Path, stem: str) -> Path:
    for ext in IMG_EXTS:
        candidate = scene_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find image for stem '{stem}' in {scene_dir}")


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def symlink_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    if copy_files:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def make_preview_grid(image_paths: list[Path], out_path: Path, labels: list[str]) -> None:
    tile_w = 220
    tile_h = 150
    gap = 12
    cols = min(3, len(image_paths))
    rows = (len(image_paths) + cols - 1) // cols
    label_h = 28
    canvas = Image.new(
        "RGB",
        (cols * tile_w + (cols + 1) * gap, rows * (tile_h + label_h) + (rows + 1) * gap),
        (255, 255, 255),
    )

    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        row = idx // cols
        col = idx % cols
        x0 = gap + col * (tile_w + gap)
        y0 = gap + row * (tile_h + label_h + gap)
        img = Image.open(img_path).convert("RGB")
        img = ImageOps.contain(img, (tile_w, tile_h))
        tile = Image.new("RGB", (tile_w, tile_h), (245, 245, 245))
        paste_x = (tile_w - img.width) // 2
        paste_y = (tile_h - img.height) // 2
        tile.paste(img, (paste_x, paste_y))
        canvas.paste(tile, (x0, y0))
        draw = ImageDraw.Draw(canvas)
        draw.text((x0 + 4, y0 + tile_h + 4), label, fill=(20, 20, 20))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def bundle_name(scene: str, trial: int) -> str:
    return f"{scene}__trial_{trial:02d}"


def build_bundle_meta(trial_entry: dict, summary_json: Path, source_paths: list[Path]) -> dict:
    metrics = trial_entry["metrics"]
    return {
        "dataset": trial_entry["dataset"],
        "scene": trial_entry["scene"],
        "trial": int(trial_entry["trial"]),
        "sample_stems": list(trial_entry["sample_stems"]),
        "sample_sources": [str(p) for p in source_paths],
        "summary_artifact": summary_json.name,
        "n_views": len(trial_entry["sample_stems"]),
        "metrics": {
            "raw_fscore_1cm": float(metrics["raw"]["fscore_1cm"]),
            "confidence_topk_fscore_1cm": float(metrics["confidence_topk"]["fscore_1cm"]),
            "discord_fscore_1cm": float(metrics["discord"]["fscore_1cm"]),
            "discord_minus_conf_fscore_1cm": float(
                metrics["discord"]["fscore_1cm"] - metrics["confidence_topk"]["fscore_1cm"]
            ),
            "discord_minus_raw_fscore_1cm": float(
                metrics["discord"]["fscore_1cm"] - metrics["raw"]["fscore_1cm"]
            ),
        },
        "view_meta": trial_entry.get("view_meta", []),
    }


def export_trial_bundle(trial_entry: dict, img_root: Path, out_root: Path, summary_json: Path, copy_files: bool) -> Path:
    scene = trial_entry["scene"]
    trial = int(trial_entry["trial"])
    stems = trial_entry["sample_stems"]
    source_scene_dir = img_root / scene / "dense" / "images"
    source_paths = [find_image(source_scene_dir, stem) for stem in stems]

    out_dir = out_root / bundle_name(scene, trial)
    safe_rmtree(out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    linked_paths = []
    for src, stem in zip(source_paths, stems):
        dst = image_dir / src.name
        symlink_or_copy(src.resolve(), dst, copy_files)
        linked_paths.append(dst)

    meta = build_bundle_meta(trial_entry, summary_json, source_paths)
    (out_dir / "bundle_meta.json").write_text(json.dumps(meta, indent=2))
    make_preview_grid(linked_paths, out_dir / "preview_grid.jpg", stems)
    return out_dir


def main():
    args = parse_args()
    summary = json.loads(args.summary_json.read_text())
    selectors = normalize_selectors(args.select)
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    exported = []
    for trial_entry in summary["trials"]:
        key = (trial_entry["scene"], int(trial_entry["trial"]))
        if selectors and key not in selectors:
            continue
        bundle_dir = export_trial_bundle(
            trial_entry=trial_entry,
            img_root=args.img_root,
            out_root=out_root,
            summary_json=args.summary_json,
            copy_files=args.copy,
        )
        exported.append(bundle_dir)
        print(bundle_dir)

    print(f"Exported {len(exported)} bundles to {out_root}")


if __name__ == "__main__":
    main()
