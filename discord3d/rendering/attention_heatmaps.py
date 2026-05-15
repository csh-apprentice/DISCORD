from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-8


def normalize_map(values: np.ndarray, valid: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float32)
    mask = valid.astype(bool) & np.isfinite(values)
    vals = values[mask].astype(np.float32)
    if vals.size == 0:
        return out
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if hi - lo <= EPS:
        out[mask] = 0.5
    else:
        out[mask] = np.clip((values[mask] - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return out.astype(np.float32)


def normalize_stack(values: np.ndarray, valid: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    return np.stack([normalize_map(values[i], valid[i], lo_pct, hi_pct) for i in range(values.shape[0])]).astype(np.float32)


def heat_rgb(values: np.ndarray, valid: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    rgb = np.ones((*values.shape, 3), dtype=np.float32)
    mapped = plt.get_cmap(cmap_name)(np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0))[..., :3].astype(np.float32)
    rgb[valid.astype(bool)] = mapped[valid.astype(bool)]
    return rgb


def mask_rgb(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rgb = np.ones((*mask.shape, 3), dtype=np.float32)
    rgb[valid.astype(bool)] = np.array([0.04, 0.04, 0.06], dtype=np.float32)
    rgb[mask.astype(bool) & valid.astype(bool)] = np.array([0.98, 0.88, 0.18], dtype=np.float32)
    return rgb


def overlay_mask(image: np.ndarray, mask: np.ndarray, valid: np.ndarray, alpha: float = 0.56) -> np.ndarray:
    color = mask_rgb(mask, valid)
    out = np.clip(image, 0.0, 1.0).copy()
    valid = valid.astype(bool)
    out[valid] = ((1.0 - float(alpha)) * out[valid] + float(alpha) * color[valid]).astype(np.float32)
    out[~valid] = 1.0
    return np.clip(out, 0.0, 1.0)


def overlay_top_fraction(
    image: np.ndarray,
    score: np.ndarray,
    valid: np.ndarray,
    top_frac: float = 0.10,
    color: np.ndarray | None = None,
) -> np.ndarray:
    if color is None:
        color = np.array([1.0, 0.82, 0.05], dtype=np.float32)
    out = np.clip(image, 0.0, 1.0).copy()
    mask = valid.astype(bool) & np.isfinite(score)
    vals = score[mask].astype(np.float32)
    if vals.size == 0:
        return out
    threshold = float(np.quantile(vals, 1.0 - float(top_frac)))
    chosen = mask & (score >= threshold)
    out[chosen] = 0.50 * out[chosen] + 0.50 * color.astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def diff_rgb(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> np.ndarray:
    valid = valid.astype(bool)
    a = a.astype(bool) & valid
    b = b.astype(bool) & valid
    rgb = np.ones((*valid.shape, 3), dtype=np.float32)
    rgb[valid] = np.array([0.04, 0.04, 0.06], dtype=np.float32)
    rgb[a & b] = np.array([0.98, 0.88, 0.18], dtype=np.float32)
    rgb[a & ~b] = np.array([0.10, 0.70, 0.95], dtype=np.float32)
    rgb[b & ~a] = np.array([0.95, 0.20, 0.72], dtype=np.float32)
    return rgb


def save_grid(
    path: str | Path,
    rows: list[tuple[str, list[np.ndarray]]],
    col_titles: list[str],
    title: str,
    dpi: int = 160,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(rows), len(col_titles), figsize=(3.2 * len(col_titles), 2.35 * len(rows)))
    axes = np.asarray(axes).reshape(len(rows), len(col_titles))
    for row_idx, (row_label, panels) in enumerate(rows):
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(np.clip(panel, 0.0, 1.0))
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title, fontsize=12, y=0.998)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.982))
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
