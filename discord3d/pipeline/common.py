from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.mixture import GaussianMixture
from torchvision import transforms as TF

from discord3d.third_party import setup_vggt_paths
from discord3d.vggt_support import PATCH_SIZE, PATCH_START, VGGT


REPO_ROOT = Path(__file__).resolve().parents[2]
setup_vggt_paths()

DTYPE = torch.bfloat16
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _restore_hooks(originals):
    for am, orig in originals:
        if orig is None:
            am.__dict__.pop("forward", None)
        else:
            am.forward = orig


def _install_entropy_hooks(model, layers: list[int], num_views: int, p_total: int, query_chunk: int = 128):
    layers_set = set(layers)
    p_patch = p_total - PATCH_START
    storage = {layer: {} for layer in layers}
    originals = []

    def _view_patch_range(view_idx: int):
        start = view_idx * p_total + PATCH_START
        end = start + p_patch
        return start, end

    def _compute_entropy_from_probs(probs: torch.Tensor) -> np.ndarray:
        # probs: [B, H, N, N]
        batch, heads, _, _ = probs.shape
        out = np.zeros((num_views, p_patch), dtype=np.float32)
        log_k = float(np.log(max((num_views - 1) * p_patch, 2)))

        for view_idx in range(num_views):
            q0, q1 = _view_patch_range(view_idx)
            cross_idx = []
            for other in range(num_views):
                if other == view_idx:
                    continue
                k0, k1 = _view_patch_range(other)
                cross_idx.extend(range(k0, k1))
            if not cross_idx:
                continue

            cross = probs[:, :, q0:q1, :][:, :, :, cross_idx]  # [B,H,P,K]
            cross_mass = cross.sum(dim=-1, keepdim=True)
            cross_norm = cross / cross_mass.clamp_min(1e-8)
            ent = -(cross_norm.clamp_min(1e-8) * cross_norm.clamp_min(1e-8).log()).sum(dim=-1) / log_k  # [B,H,P]
            ent = ent.mean(dim=1).mean(dim=0)  # [P]
            out[view_idx] = ent.detach().cpu().float().numpy().astype(np.float32)
        return out

    for layer_idx, blk in enumerate(model.aggregator.global_blocks):
        if layer_idx not in layers_set:
            continue
        am = blk.attn
        originals.append((am, am.__dict__.get("forward", None)))

        def make_fwd(am_, layer_id: int):
            def fwd(x_, pos=None, attn_mask=None, v_proj_cfg=None, **kwargs):
                batch, n_tok, cdim = x_.shape
                qkv = am_.qkv(x_).reshape(batch, n_tok, 3, am_.num_heads, am_.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q, k = am_.q_norm(q), am_.k_norm(k)

                if am_.rope is not None:
                    q = am_.rope(q, pos)
                    k = am_.rope(k, pos)

                scale = getattr(am_, "scale", am_.head_dim ** -0.5)
                logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * float(scale)
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        logits = logits.masked_fill(attn_mask, float("-inf"))
                    else:
                        logits = logits + attn_mask.to(logits.dtype)
                probs = logits.softmax(dim=-1)
                storage[layer_id]["entropy"] = _compute_entropy_from_probs(probs)

                out = torch.matmul(probs.to(v.dtype), v)
                out = out.transpose(1, 2).reshape(batch, n_tok, cdim)
                out = am_.proj(out)
                out = am_.proj_drop(out)
                return out

            return fwd

        am.forward = make_fwd(am, layer_idx)

    return storage, originals


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))


def list_images(img_dir: str | Path) -> list[Path]:
    img_dir = Path(img_dir)
    paths = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if not paths:
        raise RuntimeError(f"No images found in {img_dir}")
    return paths


def maybe_autocast(device: str):
    if str(device).startswith("cuda"):
        return torch.cuda.amp.autocast(dtype=DTYPE)
    return torch.cpu.amp.autocast(enabled=False)


def load_crop_images_with_valid_masks(paths: list[Path], target_size: int = 518):
    to_tensor = TF.ToTensor()
    imgs = []
    valid_masks = []
    shapes = []
    for path in paths:
        img = Image.open(path)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")
        width, height = img.size

        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img_t = to_tensor(img)
        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            img_t = img_t[:, start_y : start_y + target_size, :]

        h, w = img_t.shape[1:]
        imgs.append(img_t)
        valid_masks.append(torch.ones((h, w), dtype=torch.bool))
        shapes.append((h, w))

    max_h = max(h for h, _ in shapes)
    max_w = max(w for _, w in shapes)
    padded_imgs = []
    padded_masks = []
    for img_t, valid in zip(imgs, valid_masks):
        h_padding = max_h - img_t.shape[1]
        w_padding = max_w - img_t.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            img_t = torch.nn.functional.pad(img_t, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
            valid = torch.nn.functional.pad(valid.float(), (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0).bool()
        padded_imgs.append(img_t)
        padded_masks.append(valid)
    return torch.stack(padded_imgs), torch.stack(padded_masks)


def compute_depth_confidence(imgs: torch.Tensor, device: str) -> np.ndarray:
    model = VGGT.from_pretrained("facebook/VGGT-1B", local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    with torch.inference_mode():
        with maybe_autocast(device):
            pred = model(imgs.to(device).unsqueeze(0))
    conf = pred["depth_conf"][0].detach().float().cpu().numpy().astype(np.float32)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return conf


def load_or_compute_confidence(imgs: torch.Tensor, device: str, conf_npy: str | None, save_path: str | Path | None) -> np.ndarray:
    if conf_npy:
        conf = np.load(conf_npy).astype(np.float32)
    else:
        conf = compute_depth_confidence(imgs, device)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, conf)
    return conf


def load_or_compute_entropy(imgs: torch.Tensor, device: str, layer: int, save_raw_npy: str | Path | None = None) -> np.ndarray:
    if save_raw_npy is not None and Path(save_raw_npy).exists():
        return np.load(save_raw_npy).astype(np.float32)

    _, _, h, w = imgs.shape
    p_patch = (h // PATCH_SIZE) * (w // PATCH_SIZE)
    p_total = PATCH_START + p_patch

    model = VGGT.from_pretrained("facebook/VGGT-1B", local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    storage, originals = _install_entropy_hooks(model, [layer], imgs.shape[0], p_total, 128)
    try:
        with torch.inference_mode():
            with maybe_autocast(device):
                model.aggregator(imgs.unsqueeze(0).to(device))
    finally:
        _restore_hooks(originals)
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    ph = h // PATCH_SIZE
    pw = w // PATCH_SIZE
    patch = storage[layer]["entropy"].astype(np.float32).reshape(imgs.shape[0], ph, pw)
    up = np.repeat(np.repeat(patch, PATCH_SIZE, axis=1), PATCH_SIZE, axis=2)
    ent_raw = up[:, :h, :w].astype(np.float32)
    if save_raw_npy is not None:
        save_raw_npy = Path(save_raw_npy)
        save_raw_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_raw_npy, ent_raw)
    return ent_raw


def activate_entropy_valid(raw: np.ndarray, valid_mask: np.ndarray, gain: float) -> np.ndarray:
    vals = raw[valid_mask]
    mu = float(vals.mean())
    sigma = float(vals.std())
    out = np.full_like(raw, np.nan, dtype=np.float32)
    if sigma < 1e-6:
        out[valid_mask] = 0.5
        return out
    z = (raw - mu) / sigma
    out[valid_mask] = (1.0 / (1.0 + np.exp(-gain * z[valid_mask]))).astype(np.float32)
    return out


def smooth_curve(y: np.ndarray) -> np.ndarray:
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    y_pad = np.pad(y.astype(np.float32), (pad, pad), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid").astype(np.float32)


def find_curvature_peak(thresholds: np.ndarray, counts: np.ndarray) -> tuple[float, int]:
    x = thresholds.astype(np.float32)
    y = smooth_curve(counts.astype(np.float32))
    if len(x) < 7:
        return float(x[len(x) // 2]), len(x) // 2
    d1 = np.gradient(y, x)
    d2 = np.gradient(d1, x)
    curv = np.abs(d2) / np.power(1.0 + d1 * d1, 1.5)
    lo = max(2, int(0.05 * len(x)))
    hi = min(len(x) - 2, int(0.95 * len(x)))
    idx_rel = int(curv[lo:hi].argmax())
    idx = lo + idx_rel
    return float(x[idx]), idx


def fit_gmm_labels(
    vals: np.ndarray,
    min_k: int,
    max_k: int,
    max_fit_points: int | None = 20000,
) -> tuple[np.ndarray, int, list[dict]]:
    x = vals.reshape(-1, 1).astype(np.float64)
    x_fit = x
    if max_fit_points is not None and int(max_fit_points) > 0 and len(x) > int(max_fit_points):
        x_sorted = np.sort(x[:, 0], axis=0)
        sample_idx = np.linspace(0, len(x_sorted) - 1, int(max_fit_points), dtype=np.int64)
        x_fit = x_sorted[sample_idx].reshape(-1, 1).astype(np.float64)

    best = None
    best_bic = float("inf")
    best_k = min_k
    for k in range(min_k, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=0,
            reg_covar=1e-6,
        )
        gmm.fit(x_fit)
        bic = float(gmm.bic(x_fit))
        if bic < best_bic:
            best = gmm
            best_bic = bic
            best_k = k
    assert best is not None
    labels = best.predict(x)
    means = best.means_.reshape(-1)
    order = np.argsort(means)
    remap = {int(old): int(new) for new, old in enumerate(order.tolist())}
    labels = np.array([remap[int(l)] for l in labels], dtype=np.int32)
    cluster_info = []
    for old in order.tolist():
        new = remap[int(old)]
        cluster_info.append(
            {
                "cluster_id": int(new),
                "mean_log_conf": float(means[old]),
                "weight": float(best.weights_[old]),
            }
        )
    return labels, best_k, cluster_info


def _partition_region_by_thin_bridges(
    region: np.ndarray,
    bridge_kernel_sizes: tuple[int, ...] = (3, 5),
    min_seed_area: int = 48,
) -> list[np.ndarray]:
    region = region.astype(bool)
    if int(region.sum()) == 0:
        return [region]

    best_parts = [region]
    best_score = (1, int(region.sum()), 0.0)

    for kernel_size in bridge_kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1).astype(bool)
        if not opened.any():
            continue

        num_seed, seed_comp = cv2.connectedComponents(opened.astype(np.uint8), connectivity=8)
        seeds = []
        for seed_id in range(1, num_seed):
            seed = seed_comp == seed_id
            if int(seed.sum()) < int(min_seed_area):
                continue
            restored = cv2.dilate(seed.astype(np.uint8), kernel, iterations=1).astype(bool) & region
            if restored.any():
                seeds.append(restored)

        if len(seeds) <= 1:
            continue

        centroids = []
        for seed in seeds:
            ys, xs = np.where(seed)
            centroids.append(np.array([ys.mean(), xs.mean()], dtype=np.float32))
        centroids = np.stack(centroids, axis=0)

        ys, xs = np.where(region)
        coords = np.stack([ys, xs], axis=1).astype(np.float32)
        dist2 = ((coords[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        nearest = np.argmin(dist2, axis=1)

        parts = []
        for part_id in range(len(seeds)):
            part = np.zeros_like(region, dtype=bool)
            sel = nearest == part_id
            if not np.any(sel):
                continue
            part[ys[sel], xs[sel]] = True
            parts.append(part)

        if len(parts) <= 1:
            continue

        part_areas = [int(p.sum()) for p in parts]
        score = (len(parts), min(part_areas), -float(kernel_size))
        if score > best_score:
            best_parts = parts
            best_score = score

    return best_parts


def build_region_map(
    label_map: np.ndarray,
    floor_keep: np.ndarray,
    min_area: int,
    split_bridges: bool = False,
    bridge_kernel: int = 3,
    min_bridge_residue_area: int = 196,
) -> np.ndarray:
    region_map = np.full(label_map.shape, -1, dtype=np.int32)
    next_id = 0
    cluster_ids = sorted(int(c) for c in np.unique(label_map[floor_keep]) if c >= 0)
    for cluster_id in cluster_ids:
        mask = (label_map == cluster_id) & floor_keep
        num, comp_map = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        for comp_id in range(1, num):
            region = comp_map == comp_id
            region_parts = [region]
            if split_bridges:
                split_seed_area = max(48, int(min(min_area, min_bridge_residue_area) // 2))
                region_parts = _partition_region_by_thin_bridges(
                    region,
                    bridge_kernel_sizes=(int(bridge_kernel), max(int(bridge_kernel) + 2, 5)),
                    min_seed_area=split_seed_area,
                )

            for region_part in region_parts:
                area_thresh = min_area if len(region_parts) == 1 else max(8, int(min_area // 8))
                if int(region_part.sum()) < area_thresh:
                    continue
                region_map[region_part] = next_id
                next_id += 1
    return region_map


def cluster_vis(label_map: np.ndarray, floor_keep: np.ndarray, num_clusters: int) -> np.ndarray:
    colors = plt.cm.tab10(np.linspace(0, 1, max(num_clusters, 2)))[:, :3].astype(np.float32)
    rgb = np.ones((*label_map.shape, 3), dtype=np.float32)
    rgb[~floor_keep] = 0.1
    for cid in range(num_clusters):
        rgb[(label_map == cid) & floor_keep] = colors[cid % len(colors)]
    return rgb


def colorize_trust(mask_map: np.ndarray, trusted_by_id: dict[int, bool]) -> np.ndarray:
    rgb = np.zeros((*mask_map.shape, 3), dtype=np.float32)
    for rid in np.unique(mask_map):
        if rid < 0:
            continue
        trusted = trusted_by_id.get(int(rid), False)
        color = np.array([0.2, 0.85, 0.35], dtype=np.float32) if trusted else np.array([0.95, 0.25, 0.2], dtype=np.float32)
        rgb[mask_map == rid] = color
    return rgb


def split_thin_bridges(mask: np.ndarray, valid_mask: np.ndarray, kernel_size: int, min_residue_area: int) -> tuple[np.ndarray, int]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool) & valid_mask
    if not eroded.any():
        return mask.copy(), 0
    num, comp = cv2.connectedComponents(eroded.astype(np.uint8), connectivity=8)
    rebuilt = np.zeros_like(mask, dtype=bool)
    for cid in range(1, num):
        core = comp == cid
        restored = cv2.dilate(core.astype(np.uint8), kernel, iterations=1).astype(bool)
        rebuilt |= restored & mask & valid_mask
    residue = mask & valid_mask & (~rebuilt)
    if residue.any():
        num_r, comp_r = cv2.connectedComponents(residue.astype(np.uint8), connectivity=8)
        for cid in range(1, num_r):
            piece = comp_r == cid
            if int(piece.sum()) >= int(min_residue_area):
                rebuilt |= piece
    removed = int(mask.sum()) - int(rebuilt.sum())
    return rebuilt, max(removed, 0)


def fill_small_holes(mask: np.ndarray, valid_mask: np.ndarray, max_hole_area: int) -> tuple[np.ndarray, int]:
    if max_hole_area <= 0:
        return mask.copy(), 0
    inv = valid_mask & (~mask)
    num, comp = cv2.connectedComponents(inv.astype(np.uint8), connectivity=8)
    filled = mask.copy()
    filled_count = 0
    h, w = mask.shape
    for cid in range(1, num):
        hole = comp == cid
        area = int(hole.sum())
        if area == 0 or area > max_hole_area:
            continue
        ys, xs = np.where(hole)
        if ys.size == 0:
            continue
        if ys.min() == 0 or ys.max() == h - 1 or xs.min() == 0 or xs.max() == w - 1:
            continue
        filled[hole] = True
        filled_count += area
    return filled, filled_count


def fill_component_holes(mask: np.ndarray, valid_mask: np.ndarray, max_hole_area: int, max_hole_area_frac: float = 0.02) -> tuple[np.ndarray, int]:
    filled = mask.copy()
    total_filled = 0
    num, comp = cv2.connectedComponents((mask & valid_mask).astype(np.uint8), connectivity=8)
    for cid in range(1, num):
        comp_mask = comp == cid
        comp_area = int(comp_mask.sum())
        if comp_area == 0:
            continue
        ys, xs = np.where(comp_mask)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        sub = comp_mask[y0 : y1 + 1, x0 : x1 + 1]
        bg = ~sub
        vis = np.pad(bg.astype(np.uint8), 1, constant_values=1)
        ffmask = np.zeros((vis.shape[0] + 2, vis.shape[1] + 2), np.uint8)
        cv2.floodFill(vis, ffmask, (0, 0), 2)
        holes = (vis[1:-1, 1:-1] == 1) & bg
        if not holes.any():
            continue
        num_h, comp_h = cv2.connectedComponents(holes.astype(np.uint8), connectivity=8)
        area_limit = max(max_hole_area, int(max_hole_area_frac * comp_area))
        for hid in range(1, num_h):
            hole = comp_h == hid
            hole_area = int(hole.sum())
            if hole_area == 0 or hole_area > area_limit:
                continue
            filled[y0 : y1 + 1, x0 : x1 + 1][hole] = True
            total_filled += hole_area
    return filled, total_filled


def image_overlay(image: np.ndarray, mask: np.ndarray, dark: float = 0.18) -> np.ndarray:
    out = image.copy() * dark
    out[mask] = image[mask]
    return np.clip(out, 0.0, 1.0)
