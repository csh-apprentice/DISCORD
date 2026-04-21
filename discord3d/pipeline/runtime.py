from __future__ import annotations

from pathlib import Path
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .common import (
        PATCH_SIZE,
        PATCH_START,
        _install_entropy_hooks,
        _restore_hooks,
        activate_entropy_valid,
        build_region_map,
        cluster_vis,
        colorize_trust,
        fill_component_holes,
        fill_small_holes,
        find_curvature_peak,
        fit_gmm_labels,
        maybe_autocast,
        split_thin_bridges,
    )
except ImportError:
    from common import (
        PATCH_SIZE,
        PATCH_START,
        _install_entropy_hooks,
        _restore_hooks,
        activate_entropy_valid,
        build_region_map,
        cluster_vis,
        colorize_trust,
        fill_component_holes,
        fill_small_holes,
        find_curvature_peak,
        fit_gmm_labels,
        maybe_autocast,
        split_thin_bridges,
    )


def parse_entropy_layer(layer_str: str) -> int:
    layer_str = str(layer_str).strip()
    if not layer_str:
        return 8
    if "," in layer_str:
        return int(layer_str.split(",")[0].strip())
    if "-" in layer_str:
        return int(layer_str.split("-")[0].strip())
    return int(layer_str)


def _sync_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def compute_entropy_raw_from_model(model, imgs: torch.Tensor, device: str, layer: int) -> np.ndarray:
    _, _, h, w = imgs.shape
    p_patch = (h // PATCH_SIZE) * (w // PATCH_SIZE)
    p_total = PATCH_START + p_patch
    storage, originals = _install_entropy_hooks(model, [layer], imgs.shape[0], p_total, 128)
    try:
        with torch.inference_mode():
            with maybe_autocast(device):
                model.aggregator(imgs.unsqueeze(0).to(device))
    finally:
        _restore_hooks(originals)

    ph = h // PATCH_SIZE
    pw = w // PATCH_SIZE
    patch = storage[layer]["entropy"].astype(np.float32).reshape(imgs.shape[0], ph, pw)
    up = np.repeat(np.repeat(patch, PATCH_SIZE, axis=1), PATCH_SIZE, axis=2)
    return up[:, :h, :w].astype(np.float32)


def compute_final_outputs(
    model,
    imgs: torch.Tensor,
    valid_masks: torch.Tensor,
    conf: np.ndarray,
    device: str,
    entropy_layer: int = 8,
    entropy_gain: float = 3.0,
    conf_eps: float = 1e-5,
    curve_points: int = 200,
    min_k: int = 2,
    max_k: int = 6,
    min_region_area_frac: float = 0.001,
    max_hole_area_frac: float = 0.0025,
    gmm_max_fit_points: int | None = 20000,
    bridge_kernel: int = 3,
    min_bridge_residue_area: int = 196,
    split_region_bridges: bool = False,
    trust_stat: str = "mean",
    trust_quantile: float = 0.9,
    entropy_raw_override: np.ndarray | None = None,
):
    _sync_device(device)
    t_total0 = time.time()
    imgs_np = imgs.detach().cpu().float().numpy().transpose(0, 2, 3, 1).astype(np.float32)
    valid_np = valid_masks.detach().cpu().numpy().astype(bool)
    t_entropy0 = time.time()
    if entropy_raw_override is None:
        ent_raw = compute_entropy_raw_from_model(model, imgs, device, entropy_layer)
    else:
        ent_raw = entropy_raw_override.astype(np.float32)
    _sync_device(device)
    entropy_compute_s = time.time() - t_entropy0

    h, w = conf.shape[-2:]
    min_region_area = int(min_region_area_frac * h * w)
    max_hole_area = int(max_hole_area_frac * h * w)

    ent_act_all = []
    floor_keep_all = []
    label_maps = []
    region_maps = []
    final_masks = []
    filled_masks = []
    trusted_rgb = []
    cluster_rgb_all = []
    metas = []

    trust_stat = str(trust_stat).strip().lower()
    if trust_stat not in {"mean", "quantile"}:
        raise ValueError(f"Unsupported trust_stat: {trust_stat}")

    t_region0 = time.time()
    t_gmm_total = 0.0
    t_region_build_total = 0.0
    t_trust_total = 0.0
    t_fill_total = 0.0
    for i in range(conf.shape[0]):
        valid = valid_np[i]
        conf_i = conf[i].astype(np.float32)
        ent_raw_i = ent_raw[i]

        floor_thr = float(conf_i[valid].min()) + float(conf_eps)
        floor_keep = valid & (conf_i > floor_thr)

        ent_act_i = activate_entropy_valid(ent_raw_i, valid, entropy_gain)
        masked_entropy = ent_act_i[floor_keep]
        masked_entropy = masked_entropy[np.isfinite(masked_entropy)]
        thresholds = np.linspace(float(masked_entropy.min()), float(masked_entropy.max()), curve_points, dtype=np.float32)
        counts = np.array([(masked_entropy > t).sum() for t in thresholds], dtype=np.int32)
        tol, idx = find_curvature_peak(thresholds, counts)

        log_conf = np.log(np.clip(conf_i[floor_keep] - 1.0 + conf_eps, 1e-8, None)).astype(np.float32)
        t_gmm0 = time.time()
        labels, best_k, cluster_info = fit_gmm_labels(log_conf, min_k, max_k, max_fit_points=gmm_max_fit_points)
        t_gmm_total += time.time() - t_gmm0
        label_map = np.full(conf_i.shape, -1, dtype=np.int32)
        label_map[floor_keep] = labels
        t_region_build0 = time.time()
        region_map = build_region_map(
            label_map,
            floor_keep,
            min_region_area,
            split_bridges=split_region_bridges,
            bridge_kernel=bridge_kernel,
            min_bridge_residue_area=min_bridge_residue_area,
        )
        t_region_build_total += time.time() - t_region_build0

        trusted_by_id = {}
        raw_mask = np.zeros_like(floor_keep, dtype=bool)
        t_trust0 = time.time()
        for rid in sorted(int(r) for r in np.unique(region_map) if r >= 0):
            region = region_map == rid
            vals = ent_act_i[region]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            if trust_stat == "mean":
                region_score = float(vals.mean())
            else:
                region_score = float(np.quantile(vals, trust_quantile))
            trusted = region_score <= float(tol)
            trusted_by_id[rid] = trusted
            if trusted:
                raw_mask |= region
        t_trust_total += time.time() - t_trust0

        t_fill0 = time.time()
        bridge_cut_mask, bridge_removed = split_thin_bridges(raw_mask & valid, valid, bridge_kernel, min_bridge_residue_area)
        bridge_filled_mask, filled_pixels = fill_small_holes(bridge_cut_mask, valid, max_hole_area)
        merged_mask = (raw_mask & valid) | bridge_filled_mask
        final_mask_filled, component_filled = fill_component_holes(merged_mask, valid, max_hole_area)
        t_fill_total += time.time() - t_fill0

        ent_act_all.append(ent_act_i.astype(np.float32))
        floor_keep_all.append(floor_keep.astype(bool))
        label_maps.append(label_map.astype(np.int32))
        region_maps.append(region_map.astype(np.int32))
        final_masks.append(raw_mask.astype(bool))
        filled_masks.append(final_mask_filled.astype(bool))
        cluster_rgb_all.append(cluster_vis(label_map, floor_keep, best_k).astype(np.float32))
        trusted_rgb.append(colorize_trust(region_map, trusted_by_id).astype(np.float32))
        metas.append(
            {
                "entropy_layer": int(entropy_layer),
                "floor_threshold": float(floor_thr),
                "curvature_tolerance": float(tol),
                "curve_index": int(idx),
                "floor_keep_frac": float(floor_keep.sum() / valid.sum()),
                "region_keep_frac": float(raw_mask.sum() / valid.sum()),
                "final_keep_frac": float(final_mask_filled.sum() / valid.sum()),
                "best_k": int(best_k),
                "cluster_info": cluster_info,
                "split_region_bridges": bool(split_region_bridges),
                "trust_stat": trust_stat,
                "trust_quantile": float(trust_quantile),
                "bridge_removed_pixels": int(bridge_removed),
                "filled_pixels": int(filled_pixels),
                "component_filled_pixels": int(component_filled),
            }
        )
    region_post_s = time.time() - t_region0
    total_s = time.time() - t_total0

    return {
        "images": imgs_np,
        "valid_masks": valid_np,
        "entropy_raw": np.stack(ent_raw).astype(np.float32),
        "entropy_act": np.stack(ent_act_all).astype(np.float32),
        "floor_keep": np.stack(floor_keep_all).astype(bool),
        "label_map": np.stack(label_maps).astype(np.int32),
        "region_map": np.stack(region_maps).astype(np.int32),
        "region_mask": np.stack(final_masks).astype(bool),
        "final_mask": np.stack(filled_masks).astype(bool),
        "cluster_rgb": np.stack(cluster_rgb_all).astype(np.float32),
        "region_rgb": np.stack(trusted_rgb).astype(np.float32),
        "meta": metas,
        "timing": {
            "entropy_compute_s": float(entropy_compute_s),
            "region_post_s": float(region_post_s),
            "gmm_s": float(t_gmm_total),
            "region_build_s": float(t_region_build_total),
            "trust_vote_s": float(t_trust_total),
            "fill_s": float(t_fill_total),
            "total_s": float(total_s),
        },
    }


def apply_final_masks_to_predictions(predictions: dict, final_masks: np.ndarray) -> dict:
    out = dict(predictions)
    mask = final_masks.astype(np.float32)
    if "world_points_conf" in out:
        out["world_points_conf"] = out["world_points_conf"] * mask
    if "depth_conf" in out:
        out["depth_conf"] = out["depth_conf"] * mask
    out["final_mask"] = final_masks.astype(np.uint8)
    return out


def _entropy_rgb(ent_act: np.ndarray, valid: np.ndarray) -> np.ndarray:
    cmap = plt.cm.magma
    rgb = np.ones((*ent_act.shape, 3), dtype=np.float32)
    vals = np.nan_to_num(ent_act, nan=0.0)
    rgb_valid = cmap(np.clip(vals, 0.0, 1.0))[..., :3].astype(np.float32)
    rgb[valid] = rgb_valid[valid]
    rgb[~valid] = 1.0
    return rgb


def _overlay_mask(image: np.ndarray, mask: np.ndarray, dark: float = 0.18) -> np.ndarray:
    out = image.copy() * dark
    out[mask] = image[mask]
    return np.clip(out, 0.0, 1.0)


def build_method_gallery(run_dir: str | Path, names: list[str], outputs: dict) -> list[tuple[str, str]]:
    run_dir = Path(run_dir)
    heat_dir = run_dir / "method_gallery"
    heat_dir.mkdir(parents=True, exist_ok=True)
    gallery = []
    for i, name in enumerate(names):
        image = np.clip(outputs["images"][i], 0.0, 1.0)
        valid = outputs["valid_masks"][i]
        floor_keep = outputs["floor_keep"][i]
        cluster_rgb = outputs["cluster_rgb"][i]
        ent_rgb = _entropy_rgb(outputs["entropy_act"][i], valid)
        region_overlay = (0.55 * image + 0.45 * outputs["region_rgb"][i]).clip(0.0, 1.0)
        region_overlay[~floor_keep] = image[~floor_keep] * 0.18
        final_overlay = _overlay_mask(image, outputs["final_mask"][i])

        divider = np.full((image.shape[0], 8, 3), 1.0, dtype=np.float32)
        panel = np.concatenate(
            [image, divider, _overlay_mask(image, floor_keep), divider, cluster_rgb, divider, ent_rgb, divider, region_overlay, divider, final_overlay],
            axis=1,
        )
        out_path = heat_dir / f"{i:02d}_{name}_final_pipeline.png"
        cv2.imwrite(str(out_path), cv2.cvtColor((panel * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))
        keep = outputs["meta"][i]["final_keep_frac"]
        gallery.append((str(out_path), f"[{i}] {name} — final pipeline, keep={keep:.3f}"))
    return gallery
