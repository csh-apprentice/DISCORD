from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .eval_phototourism import _load_robustnerf_entries
from discord3d.pipeline.common import PATCH_SIZE, load_crop_images_with_valid_masks
from discord3d.pipeline.pca_otsu import EPS, otsu_threshold


@dataclass
class PseudoLabelConfig:
    label_align: bool = True
    label_strong_pct: float = 98.5
    label_weak_pct: float = 94.5
    label_max_component_frac: float = 0.25
    label_border_margin_frac: float = 0.035
    label_border_component_max_frac: float = 0.05
    label_max_aspect: float = 6.0
    label_thin_component_max_frac: float = 0.08
    positive_min_frac: float = 0.12
    negative_max_frac: float = 0.01
    min_pos_patches: int = 6
    min_neg_patches: int = 40


def parse_csv(spec: str) -> list[str]:
    return [part.strip() for part in str(spec).split(",") if part.strip()]


def clean_clutter_pairs(dataset_root: str | Path, scene: str) -> list[tuple[dict, dict]]:
    entries = _load_robustnerf_entries(str(dataset_root), str(scene))
    clean = {e["pair_id"]: e for e in entries if e["state"] == "clean" and e["pair_id"] is not None}
    clutter = {e["pair_id"]: e for e in entries if e["state"] == "clutter" and e["pair_id"] is not None}
    pair_ids = sorted(set(clean) & set(clutter))
    return [(clean[pair_id], clutter[pair_id]) for pair_id in pair_ids]


def sample_index_even(pairs: list[tuple[dict, dict]], n_pairs: int) -> list[tuple[dict, dict]]:
    if len(pairs) <= int(n_pairs):
        return list(pairs)
    idx = np.unique(np.round(np.linspace(0, len(pairs) - 1, int(n_pairs))).astype(int))
    while idx.size < int(n_pairs):
        missing = [i for i in range(len(pairs)) if i not in set(idx.tolist())]
        idx = np.sort(np.concatenate([idx, np.asarray(missing[: int(n_pairs) - idx.size], dtype=int)]))
    return [pairs[int(i)] for i in idx[: int(n_pairs)]]


def sample_pose_spread(pairs: list[tuple[dict, dict]], n_pairs: int) -> list[tuple[dict, dict]]:
    if len(pairs) <= int(n_pairs):
        return list(pairs)
    centers = np.stack([clean["center"] for clean, _ in pairs], axis=0).astype(np.float64)
    centroid = centers.mean(axis=0)
    selected = [int(np.argmax(np.linalg.norm(centers - centroid[None, :], axis=1)))]
    while len(selected) < int(n_pairs):
        selected_centers = centers[np.asarray(selected)]
        dists = np.linalg.norm(centers[:, None, :] - selected_centers[None, :, :], axis=-1)
        min_dist = dists.min(axis=1)
        min_dist[np.asarray(selected)] = -1.0
        selected.append(int(np.argmax(min_dist)))
    return [pairs[i] for i in sorted(selected)]


def select_pairs(
    pairs: list[tuple[dict, dict]],
    n_pairs: int,
    sampling: str,
    rng: random.Random,
) -> list[tuple[dict, dict]]:
    n_pairs = len(pairs) if int(n_pairs) < 0 else int(n_pairs)
    if len(pairs) <= n_pairs:
        return list(pairs)
    if sampling == "pose_spread":
        return sample_pose_spread(pairs, n_pairs)
    if sampling == "index_even":
        return sample_index_even(pairs, n_pairs)
    return rng.sample(pairs, n_pairs)


def select_clean_context(
    all_pairs: list[tuple[dict, dict]],
    target_clean: dict,
    n_views: int,
    mode: str = "nearest",
) -> list[dict]:
    clean_entries = [
        clean
        for clean, _clutter in all_pairs
        if (str(clean.get("pair_id")), str(clean.get("stem"))) != (str(target_clean.get("pair_id")), str(target_clean.get("stem")))
    ]
    needed = max(int(n_views) - 1, 0)
    if needed <= 0:
        return [target_clean]
    if len(clean_entries) < needed:
        raise RuntimeError(f"Need {needed} clean context views but only found {len(clean_entries)}")

    centers = np.stack([entry["center"] for entry in clean_entries], axis=0).astype(np.float64)
    target_center = np.asarray(target_clean["center"], dtype=np.float64)
    if mode == "nearest":
        order = np.argsort(np.linalg.norm(centers - target_center[None, :], axis=1), kind="stable")
        chosen = [clean_entries[int(i)] for i in order[:needed]]
    else:
        centroid = centers.mean(axis=0)
        selected = [int(np.argmax(np.linalg.norm(centers - centroid[None, :], axis=1)))]
        while len(selected) < needed:
            selected_centers = centers[np.asarray(selected)]
            dists = np.linalg.norm(centers[:, None, :] - selected_centers[None, :, :], axis=-1)
            min_dist = dists.min(axis=1)
            min_dist[np.asarray(selected)] = -1.0
            selected.append(int(np.argmax(min_dist)))
        chosen = [clean_entries[i] for i in sorted(selected)]
    return [target_clean] + chosen


def load_single_image(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    imgs, valid = load_crop_images_with_valid_masks([Path(path)])
    image = imgs.detach().cpu().float().numpy()[0].transpose(1, 2, 0).astype(np.float32)
    valid_np = valid.detach().cpu().numpy()[0].astype(bool)
    return image, valid_np


def block_mean_2d(values: np.ndarray, ph: int, pw: int) -> np.ndarray:
    h = int(ph) * PATCH_SIZE
    w = int(pw) * PATCH_SIZE
    cropped = values[:h, :w].astype(np.float32)
    grid = cropped.reshape(int(ph), PATCH_SIZE, int(pw), PATCH_SIZE)
    return grid.mean(axis=(1, 3)).reshape(int(ph) * int(pw)).astype(np.float32)


def block_any_2d(values: np.ndarray, ph: int, pw: int) -> np.ndarray:
    h = int(ph) * PATCH_SIZE
    w = int(pw) * PATCH_SIZE
    cropped = values[:h, :w].astype(bool)
    grid = cropped.reshape(int(ph), PATCH_SIZE, int(pw), PATCH_SIZE)
    return grid.any(axis=(1, 3)).reshape(int(ph) * int(pw))


def align_clean_to_clutter(
    clean_img: np.ndarray,
    clutter_img: np.ndarray,
    valid: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not enabled:
        return clean_img.astype(np.float32), valid.astype(bool), {"enabled": False, "status": "disabled"}

    clean_gray = cv2.cvtColor((np.clip(clean_img, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    clutter_gray = cv2.cvtColor((np.clip(clutter_img, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    clean_gray = cv2.GaussianBlur(clean_gray, (0, 0), 1.2).astype(np.float32) / 255.0
    clutter_gray = cv2.GaussianBlur(clutter_gray, (0, 0), 1.2).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    try:
        cc, warp = cv2.findTransformECC(clutter_gray, clean_gray, warp, cv2.MOTION_AFFINE, criteria, None, 3)
        h, w = valid.shape
        aligned = cv2.warpAffine(
            clean_img.astype(np.float32),
            warp,
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(1.0, 1.0, 1.0),
        )
        aligned_valid = cv2.warpAffine(
            valid.astype(np.uint8),
            warp,
            (w, h),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        return aligned.astype(np.float32), aligned_valid & valid.astype(bool), {
            "enabled": True,
            "status": "ok",
            "ecc": float(cc),
            "warp": warp.astype(float).tolist(),
        }
    except cv2.error as exc:
        return clean_img.astype(np.float32), valid.astype(bool), {"enabled": True, "status": "failed", "error": str(exc)}


def robust_color_match(source: np.ndarray, target: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    mask = valid.astype(bool)
    src = source[mask].reshape(-1, 3).astype(np.float64)
    tgt = target[mask].reshape(-1, 3).astype(np.float64)
    if src.shape[0] < 32:
        return source.astype(np.float32), {"status": "too_few_pixels", "fit_pixels": int(src.shape[0])}

    keep = np.ones(src.shape[0], dtype=bool)
    weights = np.ones(src.shape[0], dtype=np.float64)
    matrix = np.eye(4, 3, dtype=np.float64)
    for _iter in range(4):
        x = np.concatenate([src[keep], np.ones((int(keep.sum()), 1), dtype=np.float64)], axis=1)
        y = tgt[keep]
        try:
            matrix, *_ = np.linalg.lstsq(x * weights[keep, None], y * weights[keep, None], rcond=None)
        except np.linalg.LinAlgError:
            break
        pred = np.concatenate([src, np.ones((src.shape[0], 1), dtype=np.float64)], axis=1) @ matrix
        residual = np.mean(np.abs(pred - tgt), axis=1)
        threshold = float(np.percentile(residual, 68.0))
        keep = residual <= max(threshold, 1e-4)
        sigma = max(float(np.median(np.abs(residual - np.median(residual)))) * 1.4826, 1e-4)
        weights = 1.0 / np.maximum(1.0, residual / (2.5 * sigma))

    h, w = source.shape[:2]
    x_full = np.concatenate([source.reshape(-1, 3).astype(np.float64), np.ones((h * w, 1), dtype=np.float64)], axis=1)
    matched = (x_full @ matrix).reshape(h, w, 3)
    return np.clip(matched, 0.0, 1.0).astype(np.float32), {
        "status": "ok",
        "fit_pixels": int(keep.sum()),
        "valid_pixels": int(src.shape[0]),
        "matrix": matrix.astype(float).tolist(),
    }


def robust_change_score(clean_img: np.ndarray, clutter_img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    clean = np.clip(clean_img.astype(np.float32), 0.0, 1.0)
    clutter = np.clip(clutter_img.astype(np.float32), 0.0, 1.0)
    small_clean = cv2.GaussianBlur(clean, (0, 0), 1.0)
    small_clutter = cv2.GaussianBlur(clutter, (0, 0), 1.0)
    large_clean = cv2.GaussianBlur(clean, (0, 0), 4.0)
    large_clutter = cv2.GaussianBlur(clutter, (0, 0), 4.0)
    small = np.mean(np.abs(small_clean - small_clutter), axis=-1)
    large = np.mean(np.abs(large_clean - large_clutter), axis=-1)

    clean_lab = cv2.cvtColor((clean * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    clutter_lab = cv2.cvtColor((clutter * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    chroma = np.linalg.norm(clean_lab[..., 1:3] - clutter_lab[..., 1:3], axis=-1) / np.sqrt(2.0)

    clean_gray = cv2.cvtColor((clean * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    clutter_gray = cv2.cvtColor((clutter * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    clean_gray_s = cv2.GaussianBlur(clean_gray, (0, 0), 0.8)
    clutter_gray_s = cv2.GaussianBlur(clutter_gray, (0, 0), 0.8)
    gx_clean = cv2.Sobel(clean_gray_s, cv2.CV_32F, 1, 0, ksize=3)
    gy_clean = cv2.Sobel(clean_gray_s, cv2.CV_32F, 0, 1, ksize=3)
    gx_clutter = cv2.Sobel(clutter_gray_s, cv2.CV_32F, 1, 0, ksize=3)
    gy_clutter = cv2.Sobel(clutter_gray_s, cv2.CV_32F, 0, 1, ksize=3)
    grad_clean = np.sqrt(gx_clean * gx_clean + gy_clean * gy_clean)
    grad_clutter = np.sqrt(gx_clutter * gx_clutter + gy_clutter * gy_clutter)
    grad_gain = np.maximum(grad_clutter - 0.75 * grad_clean, 0.0)
    grad_gain = cv2.dilate(grad_gain, np.ones((7, 7), np.uint8))
    grad_gain = cv2.GaussianBlur(grad_gain, (0, 0), 2.0)
    grad_valid = valid.astype(bool) & np.isfinite(grad_gain)
    if grad_valid.any():
        lo = float(np.percentile(grad_gain[grad_valid], 60.0))
        hi = float(np.percentile(grad_gain[grad_valid], 98.5))
        grad_support = np.clip((grad_gain - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        grad_support = np.zeros_like(grad_gain, dtype=np.float32)

    texture = np.maximum(np.sqrt(gx_clean * gx_clean + gy_clean * gy_clean), np.sqrt(gx_clutter * gx_clutter + gy_clutter * gy_clutter))
    texture = cv2.GaussianBlur(texture, (0, 0), 1.2)
    texture_penalty = 1.0 / (1.0 + 1.5 * texture)
    photometric = (0.35 * small + 0.45 * large + 0.20 * chroma) * texture_penalty
    score = photometric * (0.08 + 0.92 * grad_support.astype(np.float32))
    return np.where(valid.astype(bool), score, np.nan).astype(np.float32)


def robust_pairdiff_pixel_mask(
    clean_img: np.ndarray,
    clutter_img: np.ndarray,
    valid_full: np.ndarray,
    cfg: PseudoLabelConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    valid = valid_full.astype(bool)
    aligned_clean, aligned_valid, align_meta = align_clean_to_clutter(clean_img, clutter_img, valid, cfg.label_align)
    matched_clean, color_meta = robust_color_match(aligned_clean, clutter_img, aligned_valid)
    score = robust_change_score(matched_clean, clutter_img, aligned_valid)
    score_valid = aligned_valid & np.isfinite(score)
    vals = score[score_valid].astype(np.float32)
    if vals.size == 0:
        return np.zeros_like(valid, dtype=bool), score, {"status": "empty", "alignment": align_meta, "color_match": color_meta}

    median = float(np.median(vals))
    mad = max(float(np.median(np.abs(vals - median))) * 1.4826, 1e-6)
    strong_thr = max(float(np.percentile(vals, cfg.label_strong_pct)), median + 3.0 * mad)
    weak_thr = max(float(np.percentile(vals, cfg.label_weak_pct)), median + 1.5 * mad)
    strong = score_valid & (score >= strong_thr)
    weak = score_valid & (score >= weak_thr)
    weak = cv2.morphologyEx(weak.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    weak = cv2.morphologyEx(weak.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    strong = cv2.dilate(strong.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & score_valid

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(weak.astype(np.uint8), connectivity=8)
    valid_area = max(int(score_valid.sum()), 1)
    image_h, image_w = valid.shape
    border_x = int(round(float(cfg.label_border_margin_frac) * float(image_w)))
    border_y = int(round(float(cfg.label_border_margin_frac) * float(image_h)))
    kept = np.zeros_like(valid, dtype=bool)
    components = []
    for comp_id in range(1, num_labels):
        comp = labels == comp_id
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        width = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        height = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        strong_count = int((comp & strong).sum())
        area_frac = float(area / valid_area)
        aspect = float(max(width / max(height, 1), height / max(width, 1)))
        touches_border = (
            border_x > 0
            and border_y > 0
            and (x <= border_x or y <= border_y or x + width >= image_w - border_x or y + height >= image_h - border_y)
        )
        interior_strong = strong_count
        if touches_border:
            interior = np.zeros_like(comp, dtype=bool)
            interior[border_y : max(border_y, image_h - border_y), border_x : max(border_x, image_w - border_x)] = True
            interior_strong = int((comp & strong & interior).sum())

        keep = True
        reason = "kept"
        if area < 12:
            keep, reason = False, "small_area"
        elif area_frac > float(cfg.label_max_component_frac):
            keep, reason = False, "too_large"
        elif strong_count <= 0:
            keep, reason = False, "no_strong_seed"
        elif strong_count / max(area, 1) < 0.03 and strong_count < 64:
            keep, reason = False, "weak_seed_density"
        elif touches_border and area_frac <= float(cfg.label_border_component_max_frac) and interior_strong < max(64, int(0.25 * strong_count)):
            keep, reason = False, "border_artifact"
        elif float(cfg.label_max_aspect) > 0.0 and aspect > float(cfg.label_max_aspect) and area_frac <= float(cfg.label_thin_component_max_frac):
            keep, reason = False, "thin_artifact"

        if keep:
            kept |= comp
        components.append(
            {
                "area": area,
                "area_frac": area_frac,
                "bbox": [x, y, width, height],
                "aspect": aspect,
                "touches_border": bool(touches_border),
                "strong_count": strong_count,
                "interior_strong_count": interior_strong,
                "score_mean": float(np.nanmean(score[comp])) if comp.any() else 0.0,
                "kept": bool(keep),
                "reason": reason,
            }
        )

    kept = cv2.morphologyEx(kept.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    return kept.astype(bool), score.astype(np.float32), {
        "status": "ok",
        "alignment": align_meta,
        "color_match": color_meta,
        "score_median": median,
        "score_mad": mad,
        "strong_threshold": strong_thr,
        "weak_threshold": weak_thr,
        "num_components": int(num_labels - 1),
        "kept_components": int(sum(1 for c in components if c["kept"])),
        "components": sorted(components, key=lambda c: c["area"], reverse=True)[:20],
    }


def build_pseudo_labels(
    clean_img: np.ndarray,
    clutter_img: np.ndarray,
    valid_full: np.ndarray,
    patch_shape: tuple[int, int],
    cfg: PseudoLabelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cfg = cfg or PseudoLabelConfig()
    ph, pw = patch_shape
    valid = valid_full.astype(bool)
    pixel_changed, diff, pairdiff_meta = robust_pairdiff_pixel_mask(clean_img, clutter_img, valid, cfg)
    change_frac = block_mean_2d(pixel_changed.astype(np.float32), ph, pw)
    diff_patch = block_mean_2d(np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0), ph, pw)
    patch_valid = block_any_2d(valid, ph, pw)

    label = np.full(int(ph) * int(pw), -1, dtype=np.int8)
    label[patch_valid & (change_frac >= float(cfg.positive_min_frac))] = 1
    label[patch_valid & (change_frac <= float(cfg.negative_max_frac))] = 0
    pos = int(np.sum(label == 1))
    neg = int(np.sum(label == 0))
    mode = "pixel_pairdiff_fraction"
    patch_otsu_meta = None

    if pos < int(cfg.min_pos_patches) or neg < int(cfg.min_neg_patches):
        patch_threshold, patch_otsu_meta = otsu_threshold(diff_patch[patch_valid], bins=128)
        label = np.full(int(ph) * int(pw), -1, dtype=np.int8)
        label[patch_valid & (diff_patch >= float(patch_threshold))] = 1
        label[patch_valid & (diff_patch < float(patch_threshold))] = 0
        pos = int(np.sum(label == 1))
        neg = int(np.sum(label == 0))
        mode = "fallback_patch_otsu"

    return label, patch_valid, {
        "mode": mode,
        "robust_pairdiff": pairdiff_meta,
        "patch_otsu": patch_otsu_meta,
        "positive_patches": pos,
        "negative_patches": neg,
        "ignored_patches": int(np.sum(label < 0)),
        "positive_min_frac": float(cfg.positive_min_frac),
        "negative_max_frac": float(cfg.negative_max_frac),
        "patch_diff_mean": float(diff_patch[patch_valid].mean()) if patch_valid.any() else 0.0,
        "patch_diff_p95": float(np.percentile(diff_patch[patch_valid], 95)) if patch_valid.any() else 0.0,
    }


def orient_metric(y_true: np.ndarray, score: np.ndarray) -> dict:
    finite = np.isfinite(score)
    y = y_true[finite].astype(np.int32)
    s = score[finite].astype(np.float64)
    if y.size == 0 or np.unique(y).size < 2:
        return {"auroc": float("nan"), "ap": float("nan"), "orientation": "+", "n": int(y.size)}
    auc_pos = float(roc_auc_score(y, s))
    auc_neg = float(roc_auc_score(y, -s))
    if auc_neg > auc_pos:
        s_eval = -s
        orientation = "-"
        auroc = auc_neg
    else:
        s_eval = s
        orientation = "+"
        auroc = auc_pos
    ap = float(average_precision_score(y, s_eval))
    return {"auroc": auroc, "ap": ap, "orientation": orientation, "n": int(y.size)}


def patch_ranking_metrics(label_patch: np.ndarray, patch_valid: np.ndarray, scores: dict[str, np.ndarray]) -> dict[str, dict]:
    mask = patch_valid.astype(bool) & (label_patch >= 0)
    y = label_patch[mask].astype(np.int32)
    return {name: orient_metric(y, values[mask]) for name, values in scores.items()}
