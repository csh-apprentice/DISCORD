from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .common import PATCH_SIZE, PATCH_START, load_crop_images_with_valid_masks, maybe_autocast
from .pca_otsu import EPS, patch_valid_from_masks, standardize_entropy_heads


def sync_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def restore_attention_forward(attn, original_forward) -> None:
    if original_forward is None:
        attn.__dict__.pop("forward", None)
    else:
        attn.forward = original_forward


def _view_patch_range(view_idx: int, p_total: int, p_patch: int) -> tuple[int, int]:
    start = int(view_idx) * int(p_total) + PATCH_START
    return start, start + int(p_patch)


def install_attention_stat_hooks(model, layers: list[int], num_views: int, p_total: int):
    """Capture per-head cross entropy plus self/cross attention mass.

    The returned storage is filled by a normal VGGT aggregator forward pass.
    Statistics are patch-query tensors with shape [views, heads, patches].
    """

    layers_set = set(int(layer) for layer in layers)
    p_patch = int(p_total) - PATCH_START
    storage: dict[int, dict[str, np.ndarray]] = {}
    originals = []

    cross_indices: list[list[int]] = []
    self_indices: list[list[int]] = []
    special_indices: list[list[int]] = []
    for view_idx in range(int(num_views)):
        self_start, self_end = _view_patch_range(view_idx, p_total, p_patch)
        self_idx = list(range(self_start, self_end))
        cross_idx = []
        for other_idx in range(int(num_views)):
            if other_idx == view_idx:
                continue
            k0, k1 = _view_patch_range(other_idx, p_total, p_patch)
            cross_idx.extend(range(k0, k1))
        patch_idx = set(self_idx) | set(cross_idx)
        self_indices.append(self_idx)
        cross_indices.append(cross_idx)
        special_indices.append([idx for idx in range(int(num_views) * int(p_total)) if idx not in patch_idx])

    def compute_stats(probs: torch.Tensor) -> dict[str, np.ndarray]:
        probs = probs[0].float()
        heads = int(probs.shape[0])
        entropy = torch.zeros((num_views, heads, p_patch), dtype=torch.float32, device=probs.device)
        cross_mass = torch.zeros_like(entropy)
        self_mass = torch.zeros_like(entropy)
        special_mass = torch.zeros_like(entropy)
        cross_peak = torch.zeros_like(entropy)
        log_k = float(np.log(max((int(num_views) - 1) * int(p_patch), 2)))

        for view_idx in range(int(num_views)):
            q0, q1 = _view_patch_range(view_idx, p_total, p_patch)
            cross_idx_t = torch.tensor(cross_indices[view_idx], dtype=torch.long, device=probs.device)
            self_idx_t = torch.tensor(self_indices[view_idx], dtype=torch.long, device=probs.device)
            special_idx_t = torch.tensor(special_indices[view_idx], dtype=torch.long, device=probs.device)

            cross = probs[:, q0:q1, cross_idx_t]
            cmass = cross.sum(dim=-1)
            cross_mass[view_idx] = cmass
            cross_norm = cross / cmass.unsqueeze(-1).clamp_min(EPS)
            cross_norm = cross_norm.clamp_min(EPS)
            entropy[view_idx] = -(cross_norm * cross_norm.log()).sum(dim=-1) / log_k
            cross_peak[view_idx] = cross.max(dim=-1).values / cmass.clamp_min(EPS)

            self_mass[view_idx] = probs[:, q0:q1, self_idx_t].sum(dim=-1)
            special_mass[view_idx] = probs[:, q0:q1, special_idx_t].sum(dim=-1)

        return {
            "entropy": entropy.detach().cpu().numpy().astype(np.float32),
            "cross_mass": cross_mass.detach().cpu().numpy().astype(np.float32),
            "self_mass": self_mass.detach().cpu().numpy().astype(np.float32),
            "special_mass": special_mass.detach().cpu().numpy().astype(np.float32),
            "cross_peak": cross_peak.detach().cpu().numpy().astype(np.float32),
        }

    for layer_idx, block in enumerate(model.aggregator.global_blocks):
        if int(layer_idx) not in layers_set:
            continue
        attn = block.attn
        originals.append((attn, attn.__dict__.get("forward", None)))

        def make_forward(attn_, layer_id: int):
            def forward_with_capture(x_, pos=None, attn_mask=None, v_proj_cfg=None, **kwargs):
                batch, n_tok, cdim = x_.shape
                qkv = attn_.qkv(x_).reshape(batch, n_tok, 3, attn_.num_heads, attn_.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q, k = attn_.q_norm(q), attn_.k_norm(k)
                if attn_.rope is not None:
                    q = attn_.rope(q, pos)
                    k = attn_.rope(k, pos)
                scale = getattr(attn_, "scale", attn_.head_dim**-0.5)
                logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * float(scale)
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        logits = logits.masked_fill(attn_mask, float("-inf"))
                    else:
                        logits = logits + attn_mask.to(logits.dtype)
                probs = logits.softmax(dim=-1)
                storage[int(layer_id)] = compute_stats(probs)
                out = torch.matmul(probs.to(v.dtype), v)
                out = out.transpose(1, 2).reshape(batch, n_tok, cdim)
                out = attn_.proj(out)
                out = attn_.proj_drop(out)
                return out

            return forward_with_capture

        attn.forward = make_forward(attn, int(layer_idx))

    return storage, originals


def run_attention_signal_pass_from_tensor(
    model,
    imgs: torch.Tensor,
    valid_masks: torch.Tensor | np.ndarray,
    layers: list[int],
    device: str,
) -> dict:
    """Run VGGT once and return per-layer attention statistics."""

    if isinstance(valid_masks, torch.Tensor):
        valid_np = valid_masks.detach().cpu().numpy().astype(bool)
    else:
        valid_np = np.asarray(valid_masks).astype(bool)
    images = imgs.detach().cpu().float().numpy().transpose(0, 2, 3, 1).astype(np.float32)
    views, _channels, full_h, full_w = imgs.shape
    ph = int(full_h // PATCH_SIZE)
    pw = int(full_w // PATCH_SIZE)
    p_total = PATCH_START + ph * pw

    storage, originals = install_attention_stat_hooks(model, [int(layer) for layer in layers], int(views), int(p_total))
    try:
        with torch.inference_mode():
            with maybe_autocast(device):
                model.aggregator(imgs.unsqueeze(0).to(device))
    finally:
        for attn, original in originals:
            restore_attention_forward(attn, original)
    sync_device(device)

    patch_valid = patch_valid_from_masks(valid_np, ph, pw)
    return {
        "images": images,
        "valid": valid_np,
        "patch_valid": patch_valid,
        "patch_shape": (int(ph), int(pw)),
        "stats": {int(layer): storage[int(layer)] for layer in layers},
    }


def run_attention_signal_pass(model, paths: list[str | Path], layers: list[int], device: str) -> dict:
    imgs, valid_masks = load_crop_images_with_valid_masks([Path(path) for path in paths])
    return run_attention_signal_pass_from_tensor(model, imgs, valid_masks, layers, device)


def patch_to_full(values: np.ndarray, ph: int, pw: int, full_h: int, full_w: int) -> np.ndarray:
    grid = values.astype(np.float32).reshape(int(ph), int(pw))
    return np.repeat(np.repeat(grid, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1)[: int(full_h), : int(full_w)].astype(np.float32)


def patch_stack_to_full(values: np.ndarray, ph: int, pw: int, full_h: int, full_w: int) -> np.ndarray:
    return np.stack([patch_to_full(values[i], ph, pw, full_h, full_w) for i in range(values.shape[0])]).astype(np.float32)


def single_head_scores(pass_data: dict, layer: int = 16, head: int = 2) -> dict[str, np.ndarray]:
    """Return canonical per-view patch scores for one attention head."""

    stat = pass_data["stats"][int(layer)]
    num_heads = int(stat["entropy"].shape[1])
    if int(head) < 0 or int(head) >= num_heads:
        raise ValueError(f"Head {head} is outside [0, {num_heads - 1}]")
    entropy = stat["entropy"][:, int(head)].astype(np.float32)
    self_mass = stat["self_mass"][:, int(head)].astype(np.float32)
    cross_mass = stat["cross_mass"][:, int(head)].astype(np.float32)
    entropy_trust = standardize_entropy_heads(stat["entropy"], pass_data["patch_valid"])[:, int(head)].astype(np.float32)
    patch_valid = pass_data["patch_valid"].astype(bool)

    scores = {
        "entropy_high": entropy,
        "entropy_low": (-entropy).astype(np.float32),
        "entropy_trust": entropy_trust,
        "cross_mass": cross_mass,
        "one_minus_cross": (1.0 - cross_mass).astype(np.float32),
        "low_cross": (-cross_mass).astype(np.float32),
        "self_mass": self_mass,
        "mass_diff": (self_mass - cross_mass).astype(np.float32),
        "mass_log_ratio": np.log((self_mass + EPS) / (cross_mass + EPS)).astype(np.float32),
    }
    for name, values in scores.items():
        clean = values.astype(np.float32).copy()
        clean[~patch_valid] = np.nan
        scores[name] = clean
    return scores


def robust_selectivity(values: np.ndarray, valid: np.ndarray) -> dict:
    """Unsupervised compact-tail diagnostic for heatmaps."""

    mask = valid.astype(bool) & np.isfinite(values)
    vals = values[mask].astype(np.float32)
    if vals.size < 8:
        return {
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "tail_gap": float("nan"),
            "concentration": float("nan"),
            "tail_score": float("nan"),
            "argmax_enrichment": float("nan"),
        }

    med = float(np.median(vals))
    p90 = float(np.percentile(vals, 90))
    p95 = float(np.percentile(vals, 95))
    vmax = float(np.max(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    denom = max(mad, float(np.std(vals)), 1e-6)
    z = (vals - med) / denom
    energy = np.maximum(z, 0.0) ** 2
    if float(np.sum(energy)) > EPS:
        prob = energy / float(np.sum(energy))
        ent = float(-np.sum(prob[prob > 0] * np.log(prob[prob > 0])))
        concentration = float(np.clip(1.0 - ent / float(np.log(max(vals.size, 2))), 0.0, 1.0))
    else:
        concentration = 0.0

    tail_gap = float(p95 - med)
    return {
        "median": med,
        "p90": p90,
        "p95": p95,
        "max": vmax,
        "tail_gap": tail_gap,
        "concentration": concentration,
        "tail_score": float(tail_gap * concentration),
        "argmax_enrichment": float((vmax - med) / denom),
    }
