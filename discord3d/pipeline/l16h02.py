from __future__ import annotations

import cv2
import numpy as np
import torch

from .common import PATCH_SIZE
from .head_signals import run_attention_signal_pass, run_attention_signal_pass_from_tensor, single_head_scores
from .pca_otsu import (
    EPS,
    confidence_nonfloor_mask,
    guided_filter_multichannel,
    otsu_threshold,
    resize_2d,
    resize_hwc_masked,
    resize_scalar_masked,
    resize_valid,
)


DEFAULT_LAYER = 16
DEFAULT_HEAD = 2


def _as_valid_np(valid_masks: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(valid_masks, torch.Tensor):
        return valid_masks.detach().cpu().numpy().astype(bool)
    return np.asarray(valid_masks).astype(bool)


def resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (int(w), int(h)), interpolation=cv2.INTER_NEAREST).astype(bool)


def guided_otsu_from_patch_trust(
    trust_patch: np.ndarray,
    patch_valid: np.ndarray,
    images: np.ndarray,
    valid_masks: np.ndarray,
    conf: np.ndarray,
    conf_eps: float = 1e-5,
    guided_radius: int = 8,
    guided_eps: float = 1e-3,
    otsu_bins: int = 256,
    method_name: str = "l16h02_entropy_guided_otsu_conf",
) -> dict:
    """Lift a coarse trust score to image space with RGB guidance, then Otsu + confidence floor."""

    valid_masks = valid_masks.astype(bool)
    views, full_h, full_w = valid_masks.shape
    ph = full_h // PATCH_SIZE
    pw = full_w // PATCH_SIZE
    half_h = full_h // 2
    half_w = full_w // 2

    valid_half = np.stack([resize_valid(valid_masks[i], half_h, half_w) for i in range(views)])
    rgb_half = np.stack([resize_hwc_masked(images[i], valid_masks[i], half_h, half_w) for i in range(views)])
    patch_grid = trust_patch.reshape(views, ph, pw)

    bilinear_half = np.zeros((views, half_h, half_w), dtype=np.float32)
    guided_half = np.zeros_like(bilinear_half)
    otsu_half = np.zeros((views, half_h, half_w), dtype=bool)
    view_meta: list[dict] = []

    for view_idx in range(views):
        src = resize_2d(patch_grid[view_idx], half_h, half_w, cv2.INTER_LINEAR)
        source = np.where(valid_half[view_idx], src, np.nan).astype(np.float32)
        bilinear_half[view_idx] = source
        guided_half[view_idx] = guided_filter_multichannel(
            rgb_half[view_idx],
            source,
            valid_half[view_idx],
            int(guided_radius),
            float(guided_eps),
        )
        valid = valid_half[view_idx].astype(bool) & np.isfinite(guided_half[view_idx])
        threshold, diag = otsu_threshold(guided_half[view_idx, valid], int(otsu_bins))
        keep = valid & (guided_half[view_idx] >= float(threshold))
        otsu_half[view_idx] = keep
        view_meta.append(
            {
                "mask_method": method_name,
                "otsu": diag,
                "threshold": float(threshold),
                "otsu_keep_frac_half": float(keep[valid_half[view_idx]].mean()) if valid_half[view_idx].any() else 0.0,
            }
        )

    guided_full = np.stack([resize_scalar_masked(guided_half[i], valid_half[i], full_h, full_w) for i in range(views)])
    bilinear_full = np.stack([resize_scalar_masked(bilinear_half[i], valid_half[i], full_h, full_w) for i in range(views)])
    otsu_mask = np.stack([resize_mask(otsu_half[i], full_h, full_w) for i in range(views)])
    floor_keep, floor_meta = confidence_nonfloor_mask(conf, valid_masks, conf_eps)
    final_mask = otsu_mask & floor_keep & valid_masks

    for view_idx in range(views):
        view_meta[view_idx]["floor"] = floor_meta[view_idx]
        view_meta[view_idx]["otsu_keep_frac"] = (
            float(otsu_mask[view_idx, valid_masks[view_idx]].mean()) if valid_masks[view_idx].any() else 0.0
        )
        view_meta[view_idx]["final_keep_frac"] = (
            float(final_mask[view_idx, valid_masks[view_idx]].mean()) if valid_masks[view_idx].any() else 0.0
        )

    return {
        "trust_patch": trust_patch.astype(np.float32),
        "patch_valid": patch_valid.astype(bool),
        "trust_bilinear": np.where(valid_masks, bilinear_full, np.nan).astype(np.float32),
        "trust_guided": np.where(valid_masks, guided_full, np.nan).astype(np.float32),
        "floor_keep": floor_keep.astype(bool),
        "otsu_mask": otsu_mask.astype(bool),
        "final_mask": final_mask.astype(bool),
        "meta": view_meta,
        "diagnostics": {"half_shape": [int(half_h), int(half_w)], "patch_shape": [int(ph), int(pw)]},
    }


def compute_l16h02_entropy_guided_otsu_outputs(
    model,
    imgs: torch.Tensor,
    valid_masks: torch.Tensor | np.ndarray,
    conf: np.ndarray,
    device: str,
    layer: int = DEFAULT_LAYER,
    head: int = DEFAULT_HEAD,
    conf_eps: float = 1e-5,
    guided_radius: int = 8,
    guided_eps: float = 1e-3,
    otsu_bins: int = 256,
) -> dict:
    """Phototourism-facing entropy branch: L16 h02 low-entropy trust + RGB-guided Otsu."""

    valid_np = _as_valid_np(valid_masks)
    images = imgs.detach().cpu().float().numpy().transpose(0, 2, 3, 1).astype(np.float32)
    pass_data = run_attention_signal_pass_from_tensor(model, imgs, valid_np, [int(layer)], device)
    scores = single_head_scores(pass_data, int(layer), int(head))
    out = guided_otsu_from_patch_trust(
        scores["entropy_trust"],
        pass_data["patch_valid"],
        images,
        valid_np,
        conf,
        conf_eps=conf_eps,
        guided_radius=guided_radius,
        guided_eps=guided_eps,
        otsu_bins=otsu_bins,
        method_name=f"l{int(layer):02d}h{int(head):02d}_entropy_guided_otsu_conf",
    )
    out["head_scores"] = scores
    out["attention_pass"] = pass_data
    for meta in out["meta"]:
        meta["entropy_layer"] = int(layer)
        meta["head"] = int(head)
    return out


def compute_l16h02_mass_scores(
    model,
    paths: list[str],
    device: str,
    layer: int = DEFAULT_LAYER,
    head: int = DEFAULT_HEAD,
) -> tuple[dict, dict[str, np.ndarray]]:
    """RobustNeRF-facing mass branch: return L16 h02 self/cross scores for image paths."""

    pass_data = run_attention_signal_pass(model, paths, [int(layer)], device)
    return pass_data, single_head_scores(pass_data, int(layer), int(head))


def mass_score_alias(name: str) -> str:
    name = str(name).strip().lower().replace("-", "_")
    aliases = {
        "1_minus_cross": "one_minus_cross",
        "one_minus_cross_mass": "one_minus_cross",
        "low_cross": "one_minus_cross",
        "log": "mass_log_ratio",
        "log_ratio": "mass_log_ratio",
        "self_cross_ratio": "mass_log_ratio",
        "self_minus_cross": "mass_diff",
        "diff": "mass_diff",
    }
    return aliases.get(name, name)


def score_gap(values: np.ndarray, labels: np.ndarray, valid: np.ndarray) -> dict:
    mask = valid.astype(bool) & (labels >= 0) & np.isfinite(values)
    if not mask.any():
        return {"pos_mean": float("nan"), "neg_mean": float("nan"), "pos_minus_neg": float("nan")}
    pos = mask & (labels == 1)
    neg = mask & (labels == 0)
    pos_mean = float(np.mean(values[pos])) if pos.any() else float("nan")
    neg_mean = float(np.mean(values[neg])) if neg.any() else float("nan")
    return {
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "pos_minus_neg": float(pos_mean - neg_mean) if np.isfinite(pos_mean) and np.isfinite(neg_mean) else float("nan"),
    }
