from __future__ import annotations

import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import PATCH_SIZE, PATCH_START, maybe_autocast


EPS = 1e-8


def _sync_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def patch_valid_from_masks(valid_masks: np.ndarray, ph: int, pw: int) -> np.ndarray:
    num_views = valid_masks.shape[0]
    cropped = valid_masks[:, : ph * PATCH_SIZE, : pw * PATCH_SIZE].astype(bool)
    grid = cropped.reshape(num_views, ph, PATCH_SIZE, pw, PATCH_SIZE)
    return grid.any(axis=(2, 4)).reshape(num_views, ph * pw)


def _restore_attn_forward(attn, original_forward) -> None:
    if original_forward is None:
        attn.__dict__.pop("forward", None)
    else:
        attn.forward = original_forward


def extract_entropy_heads_and_features(model, imgs: torch.Tensor, device: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Return patch-level cross-view entropy heads [V,H,P] and normalized latent features [V,P,D]."""
    num_views, _, full_h, full_w = imgs.shape
    ph = full_h // PATCH_SIZE
    pw = full_w // PATCH_SIZE
    p_patch = ph * pw
    p_total = PATCH_START + p_patch

    block = model.aggregator.global_blocks[int(layer)]
    attn = block.attn
    original_forward = attn.__dict__.get("forward", None)
    storage: dict[str, np.ndarray] = {}

    def view_patch_range(view_idx: int) -> tuple[int, int]:
        start = view_idx * p_total + PATCH_START
        return start, start + p_patch

    def compute_entropy_heads(probs: torch.Tensor) -> np.ndarray:
        probs = probs[0].float()
        heads = int(probs.shape[0])
        out = torch.zeros((num_views, heads, p_patch), dtype=torch.float32, device=probs.device)
        log_k = float(np.log(max((num_views - 1) * p_patch, 2)))
        for view_idx in range(num_views):
            q0, q1 = view_patch_range(view_idx)
            cross_idx = []
            for other_idx in range(num_views):
                if other_idx == view_idx:
                    continue
                k0, k1 = view_patch_range(other_idx)
                cross_idx.extend(range(k0, k1))
            cross_idx_t = torch.tensor(cross_idx, dtype=torch.long, device=probs.device)
            cross = probs[:, q0:q1, cross_idx_t]
            cross_mass = cross.sum(dim=-1)
            cross_norm = cross / cross_mass.unsqueeze(-1).clamp_min(EPS)
            cross_norm = cross_norm.clamp_min(EPS)
            out[view_idx] = -(cross_norm * cross_norm.log()).sum(dim=-1) / log_k
        return out.detach().cpu().float().numpy().astype(np.float32)

    def forward_with_entropy_capture(x_, pos=None, attn_mask=None, v_proj_cfg=None, **kwargs):
        batch, n_tok, cdim = x_.shape
        qkv = attn.qkv(x_).reshape(batch, n_tok, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)

        if attn.rope is not None:
            q = attn.rope(q, pos)
            k = attn.rope(k, pos)

        scale = getattr(attn, "scale", attn.head_dim**-0.5)
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * float(scale)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                logits = logits.masked_fill(attn_mask, float("-inf"))
            else:
                logits = logits + attn_mask.to(logits.dtype)
        probs = logits.softmax(dim=-1)
        storage["entropy_heads"] = compute_entropy_heads(probs)

        out = torch.matmul(probs.to(v.dtype), v)
        out = out.transpose(1, 2).reshape(batch, n_tok, cdim)
        out = attn.proj(out)
        out = attn.proj_drop(out)
        return out

    attn.forward = forward_with_entropy_capture
    try:
        with torch.inference_mode():
            with maybe_autocast(device):
                agg, _patch_start_idx = model.aggregator(imgs.unsqueeze(0).to(device))
    finally:
        _restore_attn_forward(attn, original_forward)

    feat = agg[int(layer)][0, :, PATCH_START : PATCH_START + p_patch].detach().float()
    if feat.shape[-1] > 1024:
        feat = feat[..., 1024:]
    feat = F.normalize(feat, p=2, dim=-1).cpu().numpy().astype(np.float32)
    return storage["entropy_heads"], feat


def standardize_entropy_heads(entropy_heads: np.ndarray, patch_valid: np.ndarray) -> np.ndarray:
    num_views, num_heads, p_patch = entropy_heads.shape
    out = np.full((num_views, num_heads, p_patch), np.nan, dtype=np.float32)
    for view_idx in range(num_views):
        valid = patch_valid[view_idx].astype(bool)
        for head_idx in range(num_heads):
            vals = entropy_heads[view_idx, head_idx, valid].astype(np.float32)
            if vals.size == 0:
                continue
            mu = float(vals.mean())
            sigma = max(float(vals.std()), 1e-6)
            out[view_idx, head_idx] = -((entropy_heads[view_idx, head_idx] - mu) / sigma).astype(np.float32)
    return out


def pca_common_factor(ztrust: np.ndarray, patch_valid: np.ndarray) -> np.ndarray:
    num_views, _num_heads, p_patch = ztrust.shape
    out = np.full((num_views, p_patch), np.nan, dtype=np.float32)
    for view_idx in range(num_views):
        valid = patch_valid[view_idx].astype(bool)
        x = ztrust[view_idx][:, valid].astype(np.float64)
        if x.shape[1] < 2:
            continue
        x = x - x.mean(axis=1, keepdims=True)
        try:
            _u, _s, vt = np.linalg.svd(x, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        component = vt[0].astype(np.float64)
        mean_trust = np.nanmean(ztrust[view_idx][:, valid], axis=0).astype(np.float64)
        if np.corrcoef(component, mean_trust)[0, 1] < 0:
            component = -component
        out_view = np.full(p_patch, np.nan, dtype=np.float32)
        out_view[valid] = component.astype(np.float32)
        out[view_idx] = out_view
    return out


def _row_softmax(sim: np.ndarray) -> np.ndarray:
    logits = sim.astype(np.float32)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits).astype(np.float32)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), EPS)


def full_softmax_smooth(trust: np.ndarray, features: np.ndarray, patch_valid: np.ndarray) -> np.ndarray:
    out = np.full_like(trust, np.nan, dtype=np.float32)
    for view_idx in range(trust.shape[0]):
        valid = patch_valid[view_idx].astype(bool) & np.isfinite(trust[view_idx])
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            continue
        feat = features[view_idx, idx].astype(np.float32)
        vals = trust[view_idx, idx].astype(np.float32)
        sim = np.maximum(feat @ feat.T, 0.0).astype(np.float32)
        weights = _row_softmax(sim)
        out[view_idx, idx] = (weights @ vals).astype(np.float32)
    return out


def compute_pca_full_softmax_trust(
    model,
    imgs: torch.Tensor,
    valid_masks: np.ndarray,
    device: str,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    full_h, full_w = imgs.shape[-2:]
    ph = full_h // PATCH_SIZE
    pw = full_w // PATCH_SIZE
    patch_valid = patch_valid_from_masks(valid_masks, ph, pw)
    entropy_heads, features = extract_entropy_heads_and_features(model, imgs, device, layer)
    pca_patch = pca_common_factor(standardize_entropy_heads(entropy_heads, patch_valid), patch_valid)
    trust = full_softmax_smooth(pca_patch, features, patch_valid)
    for view_idx in range(trust.shape[0]):
        trust[view_idx, ~patch_valid[view_idx].astype(bool)] = np.nan
    return trust.astype(np.float32), patch_valid, {"patch_shape": [int(ph), int(pw)]}


def resize_2d(values: np.ndarray, h: int, w: int, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(values.astype(np.float32), (int(w), int(h)), interpolation=interpolation).astype(np.float32)


def resize_valid(mask: np.ndarray, h: int, w: int, min_fraction: float = 0.5) -> np.ndarray:
    weight = cv2.resize(mask.astype(np.float32), (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
    return weight >= float(min_fraction)


def resize_hwc_masked(values: np.ndarray, valid: np.ndarray, h: int, w: int) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    weight = valid.astype(np.float32)
    num = cv2.resize(values * weight[..., None], (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
    den = cv2.resize(weight, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
    out = num / np.maximum(den[..., None], EPS)
    out[den <= EPS] = 0.0
    return out.astype(np.float32)


def resize_scalar_masked(values: np.ndarray, valid: np.ndarray, h: int, w: int) -> np.ndarray:
    finite = valid.astype(bool) & np.isfinite(values)
    clean = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    weight = finite.astype(np.float32)
    num = cv2.resize(clean * weight, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
    den = cv2.resize(weight, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
    out = num / np.maximum(den, EPS)
    out[den <= EPS] = np.nan
    return out.astype(np.float32)


def _box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * int(radius) + 1
    return cv2.boxFilter(values.astype(np.float32), ddepth=-1, ksize=(k, k), normalize=False, borderType=cv2.BORDER_REFLECT)


def guided_filter_multichannel(guide: np.ndarray, source: np.ndarray, valid: np.ndarray, radius: int, eps: float) -> np.ndarray:
    finite = np.isfinite(source) & np.all(np.isfinite(guide), axis=-1)
    valid = valid.astype(bool) & finite
    guide = np.nan_to_num(guide.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    source = np.nan_to_num(source.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid_f = valid.astype(np.float32)
    h, w, channels = guide.shape
    n = np.maximum(_box_sum(valid_f, radius), EPS)

    source_f = np.where(valid, source, 0.0).astype(np.float32)
    guide_f = np.where(valid[..., None], guide, 0.0).astype(np.float32)

    mean_i = np.zeros_like(guide_f, dtype=np.float32)
    for c in range(channels):
        mean_i[..., c] = _box_sum(guide_f[..., c] * valid_f, radius) / n
    mean_p = _box_sum(source_f * valid_f, radius) / n

    cov_ip = np.zeros((h, w, channels), dtype=np.float32)
    cov_ii = np.zeros((h, w, channels, channels), dtype=np.float32)
    for c in range(channels):
        cov_ip[..., c] = _box_sum(guide_f[..., c] * source_f * valid_f, radius) / n - mean_i[..., c] * mean_p
        for d in range(c, channels):
            cov = _box_sum(guide_f[..., c] * guide_f[..., d] * valid_f, radius) / n - mean_i[..., c] * mean_i[..., d]
            cov_ii[..., c, d] = cov
            cov_ii[..., d, c] = cov

    cov_ii = cov_ii + (float(eps) * np.eye(channels, dtype=np.float32))[None, None, :, :]
    a = np.linalg.solve(cov_ii.reshape(-1, channels, channels), cov_ip.reshape(-1, channels, 1)).reshape(h, w, channels)
    b = mean_p - np.sum(a * mean_i, axis=-1)

    mean_a = np.zeros_like(a, dtype=np.float32)
    for c in range(channels):
        mean_a[..., c] = _box_sum(a[..., c] * valid_f, radius) / n
    mean_b = _box_sum(b * valid_f, radius) / n
    q = np.sum(mean_a * guide, axis=-1) + mean_b
    return np.where(valid, q, np.nan).astype(np.float32)


def confidence_nonfloor_mask(conf: np.ndarray, valid_masks: np.ndarray, conf_eps: float) -> tuple[np.ndarray, list[dict]]:
    out = np.zeros_like(conf, dtype=bool)
    meta = []
    for view_idx in range(conf.shape[0]):
        valid = valid_masks[view_idx].astype(bool)
        vals = conf[view_idx, valid].astype(np.float32)
        if vals.size == 0:
            meta.append({"floor_threshold": None, "floor_keep_frac": 0.0})
            continue
        floor_thr = float(vals.min()) + float(conf_eps)
        nonfloor = valid & (conf[view_idx] > floor_thr)
        out[view_idx] = nonfloor
        meta.append(
            {
                "floor_threshold": floor_thr,
                "floor_keep_frac": float(nonfloor[valid].mean()) if valid.any() else 0.0,
                "floor_pixel_frac": float((valid & ~nonfloor)[valid].mean()) if valid.any() else 0.0,
            }
        )
    return out, meta


def otsu_threshold(values: np.ndarray, bins: int) -> tuple[float, dict]:
    vals = values[np.isfinite(values)].astype(np.float64)
    if vals.size == 0:
        return 0.0, {"status": "empty", "num_values": 0, "threshold": 0.0}
    lo = float(vals.min())
    hi = float(vals.max())
    if hi - lo <= EPS:
        threshold = float(np.median(vals))
        return threshold, {"status": "flat", "num_values": int(vals.size), "threshold": threshold, "min": lo, "max": hi}

    hist, edges = np.histogram(vals, bins=int(bins), range=(lo, hi))
    prob = hist.astype(np.float64)
    prob = prob / max(float(prob.sum()), EPS)
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_total = float(mu[-1])
    denom = omega * (1.0 - omega)
    scores = np.full_like(denom, -np.inf, dtype=np.float64)
    good = denom > EPS
    scores[good] = ((mu_total * omega[good] - mu[good]) ** 2) / denom[good]
    scores = np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf, posinf=-np.inf)
    idx = int(np.argmax(scores))
    threshold = float(centers[idx])
    return threshold, {
        "status": "ok",
        "num_values": int(vals.size),
        "threshold": threshold,
        "hist_bins": int(bins),
        "bin_index": idx,
        "between_class_variance": float(scores[idx]),
        "min": lo,
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "max": hi,
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
    }


def compute_pca_otsu_outputs(
    model,
    imgs: torch.Tensor,
    valid_masks: torch.Tensor,
    conf: np.ndarray,
    device: str,
    layer: int = 8,
    conf_eps: float = 1e-5,
    guided_radius: int = 8,
    guided_eps: float = 1e-3,
    otsu_bins: int = 256,
) -> dict:
    _sync_device(device)
    t_total0 = time.time()
    imgs_np = imgs.detach().cpu().float().numpy().transpose(0, 2, 3, 1).astype(np.float32)
    valid_np = valid_masks.detach().cpu().numpy().astype(bool)
    views, full_h, full_w = valid_np.shape
    ph = full_h // PATCH_SIZE
    pw = full_w // PATCH_SIZE
    half_h = full_h // 2
    half_w = full_w // 2

    t_signal0 = time.time()
    trust_patch, patch_valid, signal_meta = compute_pca_full_softmax_trust(model, imgs, valid_np, device, layer)
    _sync_device(device)
    signal_s = time.time() - t_signal0

    t_guided0 = time.time()
    valid_half = np.stack([resize_valid(valid_np[i], half_h, half_w) for i in range(views)])
    rgb_half = np.stack([resize_hwc_masked(imgs_np[i], valid_np[i], half_h, half_w) for i in range(views)])
    patch_grid = trust_patch.reshape(views, ph, pw)
    guided_half = np.zeros((views, half_h, half_w), dtype=np.float32)
    for view_idx in range(views):
        src = resize_2d(patch_grid[view_idx], half_h, half_w, cv2.INTER_LINEAR)
        source = np.where(valid_half[view_idx], src, np.nan).astype(np.float32)
        guided_half[view_idx] = guided_filter_multichannel(
            rgb_half[view_idx],
            source,
            valid_half[view_idx],
            guided_radius,
            guided_eps,
        )
    guided_full = np.stack([resize_scalar_masked(guided_half[i], valid_half[i], full_h, full_w) for i in range(views)])
    guided_full = np.where(valid_np, guided_full, np.nan).astype(np.float32)
    guided_s = time.time() - t_guided0

    t_mask0 = time.time()
    floor_keep, floor_meta = confidence_nonfloor_mask(conf, valid_np, conf_eps)
    otsu_masks = np.zeros((views, full_h, full_w), dtype=bool)
    final_masks = np.zeros_like(otsu_masks)
    view_meta = []
    for view_idx in range(views):
        valid = valid_np[view_idx].astype(bool) & np.isfinite(guided_full[view_idx])
        threshold, diag = otsu_threshold(guided_full[view_idx, valid], otsu_bins)
        otsu_mask = valid & (guided_full[view_idx] >= float(threshold))
        final_mask = otsu_mask & floor_keep[view_idx]
        otsu_masks[view_idx] = otsu_mask
        final_masks[view_idx] = final_mask
        view_meta.append(
            {
                "entropy_layer": int(layer),
                "mask_method": "pca_full_softmax_rgb_guided_otsu_conf",
                "otsu": diag,
                "floor": floor_meta[view_idx],
                "otsu_keep_frac": float(otsu_mask[valid_np[view_idx]].mean()) if valid_np[view_idx].any() else 0.0,
                "final_keep_frac": float(final_mask[valid_np[view_idx]].mean()) if valid_np[view_idx].any() else 0.0,
                "guided_radius": int(guided_radius),
                "guided_eps": float(guided_eps),
            }
        )
    mask_s = time.time() - t_mask0

    return {
        "images": imgs_np,
        "valid_masks": valid_np,
        "trust_patch": trust_patch.astype(np.float32),
        "patch_valid": patch_valid.astype(bool),
        "trust_guided": guided_full.astype(np.float32),
        "floor_keep": floor_keep.astype(bool),
        "otsu_mask": otsu_masks.astype(bool),
        "final_mask": final_masks.astype(bool),
        "meta": view_meta,
        "timing": {
            "signal_compute_s": float(signal_s),
            "guided_filter_s": float(guided_s),
            "mask_s": float(mask_s),
            "total_s": float(time.time() - t_total0),
        },
        "diagnostics": {
            **signal_meta,
            "half_shape": [int(half_h), int(half_w)],
            "otsu_bins": int(otsu_bins),
        },
    }
