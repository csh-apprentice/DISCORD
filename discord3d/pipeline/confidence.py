from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from .common import (
    ensure_dir,
    image_overlay,
    list_images,
    load_crop_images_with_valid_masks,
    load_or_compute_confidence,
    save_json,
)


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf_npy", default=None, help="Optional existing confidence cache to reuse")
    ap.add_argument("--conf_eps", type=float, default=1e-5)
    ap.add_argument("--visualize", action="store_true")
    return ap.parse_args()


def save_panel(path, image, conf, floor_keep, valid):
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")
    conf_vis = np.ma.array(conf, mask=~valid)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(np.clip(image, 0.0, 1.0))
    axes[0].set_title("input")
    axes[1].imshow(conf_vis, cmap=cmap)
    axes[1].set_title("depth confidence")
    axes[2].imshow(image_overlay(image, floor_keep))
    axes[2].set_title("floor-only keep")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    viz_dir = ensure_dir(out_dir / "viz") if args.visualize else None

    paths = list_images(args.img_dir)
    imgs, valid_masks = load_crop_images_with_valid_masks(paths)
    imgs_np = imgs.cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
    valid_np = valid_masks.cpu().numpy().astype(bool)

    conf_path = out_dir / "confidence_depth.npy"
    conf = load_or_compute_confidence(imgs, args.device, args.conf_npy, conf_path)
    np.save(out_dir / "valid_masks.npy", valid_np.astype(np.uint8))

    summary = {
        "img_dir": str(args.img_dir),
        "confidence_npy": str(conf_path.resolve()),
        "views": [],
    }

    for i, path in enumerate(paths):
        valid = valid_np[i]
        conf_i = conf[i]
        floor_thr = float(conf_i[valid].min()) + float(args.conf_eps)
        floor_keep = valid & (conf_i > floor_thr)
        summary["views"].append(
            {
                "view": path.stem,
                "path": str(path.resolve()),
                "floor_threshold": floor_thr,
                "floor_keep_frac": float(floor_keep.sum() / valid.sum()),
            }
        )
        if args.visualize:
            save_panel(viz_dir / f"{path.stem}_confidence_panel.png", imgs_np[i], conf_i, floor_keep, valid)

    save_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
