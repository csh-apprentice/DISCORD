from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from .common import (
    activate_entropy_valid,
    ensure_dir,
    find_curvature_peak,
    list_images,
    load_crop_images_with_valid_masks,
    load_or_compute_entropy,
    save_json,
    smooth_curve,
)


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--conf_npy", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--entropy_layer", type=int, default=8)
    ap.add_argument("--entropy_gain", type=float, default=3.0)
    ap.add_argument("--conf_eps", type=float, default=1e-5)
    ap.add_argument("--curve_points", type=int, default=200)
    ap.add_argument("--visualize", action="store_true")
    return ap.parse_args()


def save_curve(path, thresholds, counts, tol, idx):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, counts, lw=2)
    ax.plot(thresholds, smooth_curve(counts), lw=2, alpha=0.5)
    ax.axvline(tol, color="darkgreen", ls="--", lw=2, label=f"curvature={tol:.3f}")
    ax.scatter([thresholds[idx]], [counts[idx]], color="darkgreen", s=36)
    ax.set_xlabel("entropy tolerance")
    ax.set_ylabel("# pixels with entropy > tolerance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_heatmap(path, image, ent_act, valid):
    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="white")
    ent_vis = np.ma.array(ent_act, mask=~valid)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.clip(image, 0.0, 1.0))
    axes[0].set_title("input")
    axes[1].imshow(ent_vis, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[1].set_title("L8 activated entropy")
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

    ent_raw_path = out_dir / f"entropy_layer{args.entropy_layer:02d}_raw.npy"
    ent_raw = load_or_compute_entropy(imgs, args.device, args.entropy_layer, ent_raw_path)
    ent_act = np.full_like(ent_raw, np.nan, dtype=np.float32)

    summary = {
        "img_dir": str(args.img_dir),
        "conf_npy": str(args.conf_npy),
        "entropy_raw_npy": str(ent_raw_path.resolve()),
        "views": [],
    }

    curve_pack = {}
    for i, path in enumerate(paths):
        valid = valid_np[i]
        conf_i = conf[i]
        floor_thr = float(conf_i[valid].min()) + float(args.conf_eps)
        floor_keep = valid & (conf_i > floor_thr)

        ent_act_i = activate_entropy_valid(ent_raw[i], valid, args.entropy_gain)
        ent_act[i] = ent_act_i

        masked_entropy = ent_act_i[floor_keep]
        masked_entropy = masked_entropy[np.isfinite(masked_entropy)]
        thresholds = np.linspace(float(masked_entropy.min()), float(masked_entropy.max()), args.curve_points, dtype=np.float32)
        counts = np.array([(masked_entropy > t).sum() for t in thresholds], dtype=np.int32)
        tol, idx = find_curvature_peak(thresholds, counts)

        summary["views"].append(
            {
                "view": path.stem,
                "curvature_tolerance": float(tol),
                "curve_index": int(idx),
                "entropy_min_masked": float(masked_entropy.min()),
                "entropy_max_masked": float(masked_entropy.max()),
            }
        )
        curve_pack[path.stem] = {
            "thresholds": thresholds.tolist(),
            "counts": counts.tolist(),
            "curvature_tolerance": float(tol),
            "curve_index": int(idx),
        }

        if args.visualize:
            save_curve(viz_dir / f"{path.stem}_tolerance_curve.png", thresholds, counts, tol, idx)
            save_heatmap(viz_dir / f"{path.stem}_entropy_heatmap.png", imgs_np[i], ent_act_i, valid)

    np.save(out_dir / f"entropy_layer{args.entropy_layer:02d}_act.npy", ent_act)
    save_json(out_dir / "tolerance_summary.json", summary)
    save_json(out_dir / "tolerance_curves.json", curve_pack)


if __name__ == "__main__":
    main()
