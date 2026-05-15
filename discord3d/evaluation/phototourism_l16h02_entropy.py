#!/usr/bin/env python3
"""Visualize L16 h02 entropy in the RGB-guided Otsu pipeline on Phototourism bundles."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from discord3d.pipeline.common import PATCH_SIZE, VGGT, list_images, load_crop_images_with_valid_masks  # noqa: E402
from discord3d.pipeline.head_signals import patch_stack_to_full  # noqa: E402
from discord3d.pipeline.l16h02 import compute_l16h02_entropy_guided_otsu_outputs  # noqa: E402
from discord3d.pipeline.pca_otsu import compute_pca_otsu_outputs  # noqa: E402
from discord3d.rendering.attention_heatmaps import (  # noqa: E402
    diff_rgb,
    heat_rgb,
    mask_rgb,
    normalize_stack,
    overlay_mask,
    save_grid,
)
from discord3d.vggt_support import run_pass1_with_layers  # noqa: E402


DEFAULT_BUNDLES = (
    "brandenburg_gate__trial_00,"
    "buckingham_palace__trial_00,"
    "colosseum_exterior__trial_01,"
    "pantheon_exterior__trial_00,"
    "taj_mahal__trial_00,"
    "temple_nara_japan__trial_00"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--bundle_root", default="/home/shihan/project/DISCORD/datasets/examples/phototourism_nv5_t3")
    ap.add_argument("--bundles", default=DEFAULT_BUNDLES, help="Comma-separated bundle dirs, or 'all'.")
    ap.add_argument("--out_dir", default="outputs/experiments/l16h02_phototourism_entropy")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--head", type=int, default=2)
    ap.add_argument("--compare_l8_pca", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pca_layer", type=int, default=8)
    ap.add_argument("--conf_eps", type=float, default=1e-5)
    ap.add_argument("--guided_radius", type=int, default=8)
    ap.add_argument("--guided_eps", type=float, default=1e-3)
    ap.add_argument("--otsu_bins", type=int, default=256)
    ap.add_argument("--overlay_alpha", type=float, default=0.56)
    ap.add_argument("--dpi", type=int, default=165)
    return ap.parse_args()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_csv(spec: str) -> list[str]:
    return [part.strip() for part in str(spec).split(",") if part.strip()]


def bundle_dirs(root: Path, spec: str) -> list[Path]:
    if str(spec).strip().lower() == "all":
        return sorted(path for path in root.iterdir() if path.is_dir())
    return [root / name for name in parse_csv(spec)]


def mask_iou(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    a = a.astype(bool) & valid.astype(bool)
    b = b.astype(bool) & valid.astype(bool)
    union = a | b
    if not union.any():
        return 1.0
    return float((a & b).sum() / max(int(union.sum()), 1))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_case_sheet(
    path: Path,
    title: str,
    images: np.ndarray,
    valid: np.ndarray,
    names: list[str],
    l16: dict,
    raw_entropy_full: np.ndarray,
    l8: dict | None,
    alpha: float,
    dpi: int,
) -> None:
    entropy_disp = normalize_stack(raw_entropy_full, valid, 2.0, 98.0)
    trust_disp = normalize_stack(l16["trust_guided"], valid, 2.0, 98.0)
    rows = [
        ("Input", [np.clip(images[i], 0.0, 1.0) for i in range(images.shape[0])]),
        ("L16 h02 entropy", [heat_rgb(entropy_disp[i], valid[i], "magma") for i in range(images.shape[0])]),
        ("L16 h02 RGB-guided trust", [heat_rgb(trust_disp[i], valid[i], "magma") for i in range(images.shape[0])]),
        ("Confidence nonfloor", [mask_rgb(l16["floor_keep"][i], valid[i]) for i in range(images.shape[0])]),
        ("L16 h02 Otsu overlay", [overlay_mask(images[i], l16["otsu_mask"][i], valid[i], alpha) for i in range(images.shape[0])]),
        ("L16 h02 final overlay", [overlay_mask(images[i], l16["final_mask"][i], valid[i], alpha) for i in range(images.shape[0])]),
    ]
    if l8 is not None:
        l8_disp = normalize_stack(l8["trust_guided"], valid, 2.0, 98.0)
        rows.extend(
            [
                ("L8 PCA1 RGB-guided trust", [heat_rgb(l8_disp[i], valid[i], "magma") for i in range(images.shape[0])]),
                ("L8 PCA1 final overlay", [overlay_mask(images[i], l8["final_mask"][i], valid[i], alpha) for i in range(images.shape[0])]),
                ("Final diff: both/L8/L16", [diff_rgb(l8["final_mask"][i], l16["final_mask"][i], valid[i]) for i in range(images.shape[0])]),
            ]
        )
    save_grid(path, rows, names, title, dpi=dpi)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    case_dir = ensure_dir(out_dir / "by_case")
    bundles = bundle_dirs(Path(args.bundle_root), str(args.bundles))

    print(f"Loading VGGT-1B on {args.device}...")
    model = VGGT.from_pretrained("facebook/VGGT-1B", local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)

    rows: list[dict] = []
    for bundle in bundles:
        img_dir = bundle / "images" if (bundle / "images").is_dir() else bundle
        paths = list_images(img_dir)
        names = [path.stem for path in paths]
        print(f"[{bundle.name}] {len(paths)} views")
        imgs, valid_t = load_crop_images_with_valid_masks(paths)
        images = imgs.detach().cpu().float().numpy().transpose(0, 2, 3, 1).astype(np.float32)
        valid = valid_t.detach().cpu().numpy().astype(bool)
        full_h, full_w = valid.shape[1:]
        ph, pw = full_h // PATCH_SIZE, full_w // PATCH_SIZE

        pred, _selected = run_pass1_with_layers(imgs, model, str(args.device), feat_layers=[])
        conf = pred["depth_conf"][0].detach().float().cpu().numpy().astype(np.float32)

        l16 = compute_l16h02_entropy_guided_otsu_outputs(
            model,
            imgs,
            valid_t,
            conf,
            str(args.device),
            layer=int(args.layer),
            head=int(args.head),
            conf_eps=float(args.conf_eps),
            guided_radius=int(args.guided_radius),
            guided_eps=float(args.guided_eps),
            otsu_bins=int(args.otsu_bins),
        )
        raw_entropy_full = patch_stack_to_full(
            l16["head_scores"]["entropy_high"],
            ph,
            pw,
            full_h,
            full_w,
        )

        l8 = None
        if args.compare_l8_pca:
            l8 = compute_pca_otsu_outputs(
                model,
                imgs,
                valid_t,
                conf,
                str(args.device),
                layer=int(args.pca_layer),
                conf_eps=float(args.conf_eps),
                guided_radius=int(args.guided_radius),
                guided_eps=float(args.guided_eps),
                otsu_bins=int(args.otsu_bins),
            )

        save_case_sheet(
            case_dir / f"{bundle.name}_L{args.layer:02d}h{args.head:02d}_entropy.png",
            f"{bundle.name}: L{args.layer:02d} h{args.head:02d} entropy guided Otsu",
            images,
            valid,
            names,
            l16,
            raw_entropy_full,
            l8,
            float(args.overlay_alpha),
            int(args.dpi),
        )

        for view_idx, name in enumerate(names):
            row = {
                "bundle": bundle.name,
                "view": name,
                "layer": int(args.layer),
                "head": int(args.head),
                "l16h02_otsu_keep_frac": l16["meta"][view_idx]["otsu_keep_frac"],
                "l16h02_final_keep_frac": l16["meta"][view_idx]["final_keep_frac"],
                "confidence_nonfloor_keep_frac": l16["meta"][view_idx]["floor"]["floor_keep_frac"],
            }
            if l8 is not None:
                row.update(
                    {
                        "l8_pca_otsu_keep_frac": l8["meta"][view_idx]["otsu_keep_frac"],
                        "l8_pca_final_keep_frac": l8["meta"][view_idx]["final_keep_frac"],
                        "final_mask_iou_l8_l16h02": mask_iou(l8["final_mask"][view_idx], l16["final_mask"][view_idx], valid[view_idx]),
                    }
                )
            rows.append(row)

        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = out_dir / "phototourism_l16h02_entropy_summary.csv"
    write_csv(csv_path, rows)
    summary = {
        "config": vars(args),
        "bundles": [str(path) for path in bundles],
        "case_dir": str(case_dir.resolve()),
        "summary_csv": str(csv_path.resolve()),
        "row_count": len(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved heatmaps to {case_dir}")
    print(f"Saved metrics to {csv_path}")


if __name__ == "__main__":
    main()
