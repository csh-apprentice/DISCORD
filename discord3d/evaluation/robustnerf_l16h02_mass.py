#!/usr/bin/env python3
"""Evaluate and visualize L16 h02 mass scores on RobustNeRF clean/clutter pairs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from discord3d.evaluation.robustnerf_pairs import (  # noqa: E402
    PseudoLabelConfig,
    build_pseudo_labels,
    clean_clutter_pairs,
    load_single_image,
    parse_csv,
    patch_ranking_metrics,
    select_clean_context,
    select_pairs,
)
from discord3d.pipeline.common import VGGT  # noqa: E402
from discord3d.pipeline.head_signals import patch_to_full, robust_selectivity  # noqa: E402
from discord3d.pipeline.l16h02 import compute_l16h02_mass_scores  # noqa: E402
from discord3d.rendering.attention_heatmaps import heat_rgb, mask_rgb, normalize_map, overlay_top_fraction, save_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset_root", default="/data/shihan/robustnerf")
    ap.add_argument("--scenes", default="android,crab2,statue,yoda")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--head", type=int, default=2)
    ap.add_argument("--n_pairs", type=int, default=8, help="Pairs per scene. Negative means all available pairs.")
    ap.add_argument("--n_views", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sampling", choices=["pose_spread", "index_even", "random"], default="pose_spread")
    ap.add_argument("--context_mode", choices=["nearest", "pose_spread"], default="nearest")
    ap.add_argument("--max_pair_center_dist", type=float, default=0.02)
    ap.add_argument("--support_context", choices=["clean", "clutter"], default="clean")
    ap.add_argument("--top_frac", type=float, default=0.10)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out_dir", default="outputs/experiments/l16h02_robustnerf_mass")
    ap.add_argument("--visualize_max_per_scene", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=160)
    return ap.parse_args()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def full_score(values: np.ndarray, pass_data: dict, view_idx: int = 0) -> np.ndarray:
    ph, pw = pass_data["patch_shape"]
    full_h, full_w = pass_data["valid"].shape[1:]
    return patch_to_full(values[int(view_idx)].astype(np.float32), ph, pw, full_h, full_w)


def label_panel(label_patch: np.ndarray, valid_full: np.ndarray, patch_shape: tuple[int, int]) -> np.ndarray:
    ph, pw = patch_shape
    full = patch_to_full((label_patch == 1).astype(np.float32), ph, pw, valid_full.shape[0], valid_full.shape[1]) >= 0.5
    return mask_rgb(full, valid_full)


def save_case_sheet(
    path: Path,
    scene: str,
    pair_id: str,
    clean_img: np.ndarray,
    pass_data: dict,
    label_patch: np.ndarray,
    scores: dict[str, np.ndarray],
    metrics: dict[str, dict],
    top_frac: float,
    dpi: int,
) -> None:
    image = np.clip(pass_data["images"][0], 0.0, 1.0)
    valid = pass_data["valid"][0]
    ph, pw = pass_data["patch_shape"]
    rows = [
        ("Clean target", [np.clip(clean_img, 0.0, 1.0)]),
        ("Clutter target", [image]),
        ("Pseudo clutter label", [label_panel(label_patch, valid, (ph, pw))]),
    ]
    for key, label in [
        ("one_minus_cross", "L16 h02 1 - cross mass"),
        ("mass_log_ratio", "L16 h02 log self/cross"),
        ("self_mass", "L16 h02 self mass"),
        ("entropy_high", "L16 h02 entropy"),
    ]:
        full = full_score(scores[key], pass_data, 0)
        metric = metrics.get(key, {})
        metric_txt = ""
        if np.isfinite(metric.get("auroc", np.nan)):
            metric_txt = f"  AUC {metric['auroc']:.3f} AP {metric['ap']:.3f}"
        rows.append((label + metric_txt, [heat_rgb(normalize_map(full, valid), valid, "magma")]))
        if key in {"one_minus_cross", "mass_log_ratio"}:
            rows.append((label + " top overlay", [overlay_top_fraction(image, full, valid, top_frac)]))
    save_grid(path, rows, [f"{scene} pair {pair_id}"], f"RobustNeRF L16 h02 mass: {scene} pair {pair_id}", dpi=dpi)


def summarize(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for score_name in sorted({row["score_name"] for row in rows}):
        vals = [row for row in rows if row["score_name"] == score_name]
        auc = np.asarray([row["auroc"] for row in vals if np.isfinite(row["auroc"])], dtype=np.float32)
        ap = np.asarray([row["ap"] for row in vals if np.isfinite(row["ap"])], dtype=np.float32)
        out[score_name] = {
            "n": len(vals),
            "auroc_mean": float(np.mean(auc)) if auc.size else float("nan"),
            "auroc_median": float(np.median(auc)) if auc.size else float("nan"),
            "ap_mean": float(np.mean(ap)) if ap.size else float("nan"),
            "ap_median": float(np.median(ap)) if ap.size else float("nan"),
        }
    return out


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    case_dir = ensure_dir(out_dir / "by_case")
    rng = random.Random(int(args.seed))
    label_cfg = PseudoLabelConfig()

    print(f"Loading VGGT-1B on {args.device}...")
    model = VGGT.from_pretrained("facebook/VGGT-1B", local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)

    metric_rows: list[dict] = []
    stat_rows: list[dict] = []
    cases: list[dict] = []
    for scene in parse_csv(args.scenes):
        pairs = clean_clutter_pairs(args.dataset_root, scene)
        if float(args.max_pair_center_dist) >= 0.0:
            before = len(pairs)
            pairs = [
                (clean, clutter)
                for clean, clutter in pairs
                if float(np.linalg.norm(np.asarray(clean["center"], dtype=np.float64) - np.asarray(clutter["center"], dtype=np.float64)))
                <= float(args.max_pair_center_dist)
            ]
            print(f"[{scene}] kept {len(pairs)}/{before} near-pose pairs")
        clutter_by_pair_id = {str(clean["pair_id"]): clutter for clean, clutter in pairs}
        selected = select_pairs(pairs, int(args.n_pairs), str(args.sampling), rng)
        for case_idx, (clean, clutter) in enumerate(selected):
            context = select_clean_context(pairs, clean, int(args.n_views), str(args.context_mode))
            if args.support_context == "clean":
                support_entries = context[1:]
            else:
                support_entries = [clutter_by_pair_id[str(entry["pair_id"])] for entry in context[1:]]
            paths = [str(clutter["path"])] + [str(entry["path"]) for entry in support_entries]
            print(f"[{scene} {case_idx + 1}/{len(selected)}] pair {clean['pair_id']}")

            clean_img, clean_valid = load_single_image(clean["path"])
            pass_data, scores = compute_l16h02_mass_scores(model, paths, str(args.device), int(args.layer), int(args.head))
            valid_full = clean_valid & pass_data["valid"][0]
            label_patch, label_valid, label_meta = build_pseudo_labels(
                clean_img,
                pass_data["images"][0],
                valid_full,
                pass_data["patch_shape"],
                label_cfg,
            )
            eval_valid = label_valid & pass_data["patch_valid"][0]
            selected_scores = {
                "one_minus_cross": scores["one_minus_cross"][0],
                "mass_log_ratio": scores["mass_log_ratio"][0],
                "self_mass": scores["self_mass"][0],
                "mass_diff": scores["mass_diff"][0],
                "entropy_high": scores["entropy_high"][0],
                "entropy_trust": scores["entropy_trust"][0],
            }
            metrics = patch_ranking_metrics(label_patch, eval_valid, selected_scores)
            for score_name, metric in metrics.items():
                metric_rows.append(
                    {
                        "scene": scene,
                        "pair_id": str(clean["pair_id"]),
                        "clean": clean["stem"],
                        "clutter": clutter["stem"],
                        "score_name": score_name,
                        **metric,
                        "positive_patches": int(np.sum(label_patch == 1)),
                        "negative_patches": int(np.sum(label_patch == 0)),
                    }
                )
                stat_rows.append(
                    {
                        "scene": scene,
                        "pair_id": str(clean["pair_id"]),
                        "score_name": score_name,
                        **robust_selectivity(selected_scores[score_name], pass_data["patch_valid"][0]),
                    }
                )

            if case_idx < int(args.visualize_max_per_scene):
                save_case_sheet(
                    case_dir / f"{scene}_pair{clean['pair_id']}_L{args.layer:02d}h{args.head:02d}_mass.png",
                    scene,
                    str(clean["pair_id"]),
                    clean_img,
                    pass_data,
                    label_patch,
                    scores,
                    metrics,
                    float(args.top_frac),
                    int(args.dpi),
                )

            cases.append(
                {
                    "scene": scene,
                    "pair_id": str(clean["pair_id"]),
                    "clean": clean["stem"],
                    "clutter": clutter["stem"],
                    "support": [entry["stem"] for entry in support_entries],
                    "label_meta": label_meta,
                    "metrics": metrics,
                }
            )
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    metric_csv = out_dir / "robustnerf_l16h02_mass_metrics.csv"
    stat_csv = out_dir / "robustnerf_l16h02_mass_selectivity.csv"
    write_csv(metric_csv, metric_rows)
    write_csv(stat_csv, stat_rows)
    summary = {
        "config": vars(args),
        "metric_summary": summarize(metric_rows),
        "case_dir": str(case_dir.resolve()),
        "metric_csv": str(metric_csv.resolve()),
        "selectivity_csv": str(stat_csv.resolve()),
        "cases": cases,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved heatmaps to {case_dir}")
    print(f"Saved metrics to {metric_csv}")
    print(json.dumps(summary["metric_summary"], indent=2))


if __name__ == "__main__":
    main()
