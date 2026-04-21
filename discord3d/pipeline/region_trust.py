from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import (
    build_region_map,
    cluster_vis,
    colorize_trust,
    ensure_dir,
    fit_gmm_labels,
    image_overlay,
    list_images,
    load_crop_images_with_valid_masks,
    save_json,
)


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--conf_npy", required=True)
    ap.add_argument("--entropy_act_npy", required=True)
    ap.add_argument("--tolerance_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--conf_eps", type=float, default=1e-5)
    ap.add_argument("--min_k", type=int, default=2)
    ap.add_argument("--max_k", type=int, default=6)
    ap.add_argument("--min_region_area_frac", type=float, default=0.001)
    ap.add_argument("--bridge_kernel", type=int, default=3)
    ap.add_argument("--min_bridge_residue_area", type=int, default=196)
    ap.add_argument("--gmm_max_fit_points", type=int, default=0)
    ap.add_argument("--trust_quantile", type=float, default=0.9)
    ap.add_argument("--visualize", action="store_true")
    return ap.parse_args()


def save_panel(path, image, floor_keep, cluster_rgb, ent_act, region_rgb, final_mask, valid):
    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="white")
    ent_vis = np.ma.array(ent_act, mask=~valid)
    region_overlay = (0.55 * image + 0.45 * region_rgb).clip(0.0, 1.0)
    region_overlay[~floor_keep] = image[~floor_keep] * 0.18

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    axes[0].imshow(np.clip(image, 0.0, 1.0))
    axes[0].set_title("input")
    axes[1].imshow(image_overlay(image, floor_keep))
    axes[1].set_title("floor-only")
    axes[2].imshow(np.clip(cluster_rgb, 0.0, 1.0))
    axes[2].set_title("confidence regions")
    axes[3].imshow(ent_vis, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[3].set_title("L8 activated entropy")
    axes[4].imshow(np.clip(region_overlay, 0.0, 1.0))
    axes[4].set_title("region trust")
    axes[5].imshow(image_overlay(image, final_mask))
    axes[5].set_title("curvature + region")
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

    conf = np.load(args.conf_npy).astype(np.float32)
    ent_act = np.load(args.entropy_act_npy).astype(np.float32)
    tolerance_summary = json.loads(Path(args.tolerance_json).read_text())
    tol_by_view = {v["view"]: float(v["curvature_tolerance"]) for v in tolerance_summary["views"]}

    min_region_area = int(args.min_region_area_frac * imgs.shape[-2] * imgs.shape[-1])
    region_maps = []
    label_maps = []
    final_masks = []
    region_rows = []
    summary = {"img_dir": str(args.img_dir), "views": []}

    for i, path in enumerate(paths):
        name = path.stem
        image = np.clip(imgs_np[i], 0.0, 1.0)
        valid = valid_np[i]
        conf_i = conf[i]
        ent_i = ent_act[i]

        floor_thr = float(conf_i[valid].min()) + float(args.conf_eps)
        floor_keep = valid & (conf_i > floor_thr)
        log_conf = np.log(np.clip(conf_i[floor_keep] - 1.0 + args.conf_eps, 1e-8, None)).astype(np.float32)
        labels, best_k, cluster_info = fit_gmm_labels(
            log_conf,
            args.min_k,
            args.max_k,
            max_fit_points=(None if args.gmm_max_fit_points <= 0 else args.gmm_max_fit_points),
        )

        label_map = np.full(conf_i.shape, -1, dtype=np.int32)
        label_map[floor_keep] = labels
        region_map = build_region_map(
            label_map,
            floor_keep,
            min_region_area,
            split_bridges=True,
            bridge_kernel=args.bridge_kernel,
            min_bridge_residue_area=args.min_bridge_residue_area,
        )
        tolerance = tol_by_view[name]

        trusted_by_id = {}
        for rid in sorted(int(r) for r in np.unique(region_map) if r >= 0):
            region = region_map == rid
            vals = ent_i[region]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            region_q90 = float(np.quantile(vals, args.trust_quantile))
            trusted = region_q90 <= tolerance
            trusted_by_id[rid] = trusted
            region_rows.append(
                {
                    "view": name,
                    "region_id": int(rid),
                    "area_pixels": int(region.sum()),
                    "area_frac": float(region.sum() / region.size),
                    "entropy_q90": region_q90,
                    "trusted": bool(trusted),
                    "curvature_tolerance": float(tolerance),
                    "best_k": int(best_k),
                }
            )

        final_mask = np.zeros_like(floor_keep, dtype=bool)
        for rid, trusted in trusted_by_id.items():
            if trusted:
                final_mask |= region_map == int(rid)

        region_maps.append(region_map)
        label_maps.append(label_map)
        final_masks.append(final_mask.astype(np.uint8))

        summary["views"].append(
            {
                "view": name,
                "best_k": int(best_k),
                "curvature_tolerance": float(tolerance),
                "floor_keep_frac": float(floor_keep.sum() / valid.sum()),
                "region_final_keep_frac": float(final_mask.sum() / valid.sum()),
                "cluster_info": cluster_info,
                "trust_stat": "quantile",
                "trust_quantile": float(args.trust_quantile),
            }
        )

        if args.visualize:
            cluster_rgb = cluster_vis(label_map, floor_keep, best_k)
            region_rgb = colorize_trust(region_map, trusted_by_id)
            save_panel(viz_dir / f"{name}_region_panel.png", image, floor_keep, cluster_rgb, ent_i, region_rgb, final_mask, valid)

    np.save(out_dir / "region_map.npy", np.stack(region_maps).astype(np.int32))
    np.save(out_dir / "cluster_label_map.npy", np.stack(label_maps).astype(np.int16))
    np.save(out_dir / "mask_region.npy", np.stack(final_masks).astype(np.uint8))

    with (out_dir / "region_stats.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(region_rows[0].keys()))
        writer.writeheader()
        writer.writerows(region_rows)
    save_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
