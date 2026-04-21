from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from .common import (
    ensure_dir,
    fill_component_holes,
    fill_small_holes,
    image_overlay,
    list_images,
    load_crop_images_with_valid_masks,
    save_json,
    split_thin_bridges,
)


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--mask_region_npy", required=True)
    ap.add_argument("--valid_masks_npy", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_hole_area_frac", type=float, default=0.0025)
    ap.add_argument("--bridge_kernel", type=int, default=3)
    ap.add_argument("--min_bridge_residue_area", type=int, default=196)
    ap.add_argument("--visualize", action="store_true")
    return ap.parse_args()


def save_compare_panel(path, image, final_mask, filled_mask, bridge_mask):
    final_overlay = image_overlay(image, final_mask)
    filled_overlay = image_overlay(image, filled_mask)
    hole_mask = filled_mask & (~final_mask)
    hole_rgb = np.zeros((*hole_mask.shape, 3), dtype=np.float32)
    hole_rgb[..., 0] = hole_mask.astype(np.float32)
    hole_rgb[..., 1] = 0.9 * hole_mask.astype(np.float32)
    bridge_rgb = np.zeros((*bridge_mask.shape, 3), dtype=np.float32)
    bridge_rgb[..., 0] = bridge_mask.astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.clip(image, 0.0, 1.0))
    axes[0].set_title("input")
    axes[1].imshow(np.clip(final_overlay, 0.0, 1.0))
    axes[1].set_title("curvature + region")
    axes[2].imshow(np.clip(filled_overlay, 0.0, 1.0))
    axes[2].imshow(np.clip(hole_rgb, 0.0, 1.0), alpha=0.35)
    axes[2].imshow(np.clip(bridge_rgb, 0.0, 1.0), alpha=0.20)
    axes[2].set_title("final filled")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    viz_dir = ensure_dir(out_dir / "viz") if args.visualize else None
    masked_dir = ensure_dir(out_dir / "masked_images")

    paths = list_images(args.img_dir)
    imgs, _ = load_crop_images_with_valid_masks(paths)
    imgs_np = imgs.cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
    valid_np = np.load(args.valid_masks_npy).astype(bool)
    final_masks = np.load(args.mask_region_npy).astype(bool)

    max_hole_area = int(args.max_hole_area_frac * final_masks.shape[-2] * final_masks.shape[-1])
    packed = []
    summary = {"img_dir": str(args.img_dir), "views": []}

    for i, path in enumerate(paths):
        image = np.clip(imgs_np[i], 0.0, 1.0)
        valid = valid_np[i]
        final_mask = final_masks[i] & valid

        bridge_cut_mask, bridge_removed_pixels = split_thin_bridges(
            final_mask,
            valid,
            args.bridge_kernel,
            args.min_bridge_residue_area,
        )
        bridge_filled_mask, filled_pixels = fill_small_holes(bridge_cut_mask, valid, max_hole_area)
        merged_mask = final_mask | bridge_filled_mask
        final_mask_filled, component_filled_pixels = fill_component_holes(merged_mask, valid, max_hole_area)
        bridge_mask = final_mask & (~bridge_cut_mask)

        packed.append(final_mask_filled.astype(np.uint8))
        plt.imsave(masked_dir / f"{path.stem}_masked.png", image_overlay(image, final_mask_filled))

        if args.visualize:
            save_compare_panel(viz_dir / f"{path.stem}_final_compare.png", image, final_mask, final_mask_filled, bridge_mask)

        summary["views"].append(
            {
                "view": path.stem,
                "region_keep_frac": float(final_mask.sum() / valid.sum()),
                "final_keep_frac": float(final_mask_filled.sum() / valid.sum()),
                "bridge_removed_pixels": int(bridge_removed_pixels),
                "filled_pixels": int(filled_pixels),
                "component_filled_pixels": int(component_filled_pixels),
                "masked_image": str((masked_dir / f"{path.stem}_masked.png").resolve()),
            }
        )

    np.save(out_dir / "final_mask.npy", np.stack(packed).astype(np.uint8))
    save_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
