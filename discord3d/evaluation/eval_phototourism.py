#!/usr/bin/env python3
"""Evaluate DISCORD and natural confidence baselines on reconstruction metrics."""

from __future__ import annotations

import argparse
import csv
import json
import random
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from discord3d.pipeline.common import (  # noqa: E402
    PATCH_SIZE,
    PATCH_START,
    VGGT,
    _install_entropy_hooks,
    _restore_hooks,
    activate_entropy_valid,
    build_region_map,
    fit_gmm_labels,
    load_crop_images_with_valid_masks,
    maybe_autocast,
    find_curvature_peak,
    split_thin_bridges,
    fill_small_holes,
    fill_component_holes,
)
from discord3d.pipeline.runtime import compute_final_outputs  # noqa: E402
from discord3d.vggt_support import cameras_from_pred, run_pass1_with_layers, read_colmap_poses  # noqa: E402


PHOTOTOURISM_SCENES = [
    "brandenburg_gate",
    "buckingham_palace",
    "colosseum_exterior",
    "pantheon_exterior",
    "taj_mahal",
    "temple_nara_japan",
]
ROBUSTNERF_SCENES = ["android", "crab1", "crab2", "statue", "yoda"]
LLFF_SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]

POINT_METRIC_FIELDS = [
    "acc",
    "acc_median",
    "acc_p90",
    "comp",
    "fscore_5mm",
    "fscore_1cm",
    "fscore_2cm",
    "precision_5mm",
    "precision_1cm",
    "precision_2cm",
    "recall_5mm",
    "recall_1cm",
    "recall_2cm",
    "outlier_5cm",
]


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", choices=["phototourism", "robustnerf", "llff"], default="phototourism")
    ap.add_argument("--scene", default="taj_mahal")
    ap.add_argument("--all_scenes", action="store_true")
    ap.add_argument("--bundle_root", default=None, help="Optional root of curated trial bundles to evaluate instead of random sampling.")
    ap.add_argument("--n_views", type=int, default=5)
    ap.add_argument("--n_trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--img_root", default=None)
    ap.add_argument("--colmap_root", default=None)
    ap.add_argument("--entropy_layer", type=int, default=8)
    ap.add_argument("--entropy_gain", type=float, default=3.0)
    ap.add_argument("--conf_eps", type=float, default=1e-5)
    ap.add_argument("--curve_points", type=int, default=200)
    ap.add_argument("--min_k", type=int, default=2)
    ap.add_argument("--max_k", type=int, default=6)
    ap.add_argument("--min_region_area_frac", type=float, default=0.001)
    ap.add_argument("--max_hole_area_frac", type=float, default=0.0025)
    ap.add_argument("--bridge_kernel", type=int, default=3)
    ap.add_argument("--min_bridge_residue_area", type=int, default=196)
    ap.add_argument("--split_region_bridges", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--trust_stat", choices=["mean", "quantile"], default="quantile")
    ap.add_argument("--trust_quantile", type=float, default=0.9)
    ap.add_argument(
        "--gmm_max_fit_points",
        type=int,
        default=0,
        help="0 or negative means exact full-fit GMM; positive enables approximate fit on a deterministic subsample.",
    )
    ap.add_argument("--photo_min_obs", type=int, default=1)
    ap.add_argument("--out_csv", default="outputs/eval/results.csv")
    return ap.parse_args()


def _default_roots(dataset: str) -> tuple[str, str]:
    if dataset == "phototourism":
        root = "/data/shihan/phototourism"
        return root, root
    if dataset == "robustnerf":
        root = "/data/shihan/robustnerf"
        return root, root
    if dataset == "llff":
        root = "/data/shihan/llff_full"
        return root, root
    raise ValueError(dataset)


def _align_trajectories_sim3(pred: np.ndarray, gt: np.ndarray):
    assert pred.shape == gt.shape
    mu_pred = pred.mean(axis=0)
    mu_gt = gt.mean(axis=0)
    pred_c = pred - mu_pred
    gt_c = gt - mu_gt

    cov = pred_c.T @ gt_c / pred.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3, dtype=np.float32)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[-1, -1] = -1.0
    r = vt.T @ s_mat @ u.T

    var_pred = float(np.mean(np.sum(pred_c ** 2, axis=1)))
    scale = 1.0 if var_pred < 1e-12 else float(np.trace(np.diag(d) @ s_mat) / var_pred)
    t = mu_gt - scale * (r @ mu_pred)
    aligned = (scale * (r @ pred.T)).T + t
    return aligned.astype(np.float32), float(scale), r.astype(np.float32), t.astype(np.float32)


def _extract_points(pred: dict, conf_map: np.ndarray, conf_thresh: float) -> np.ndarray:
    pts = pred["world_points"][0].detach().cpu().numpy()
    mask = conf_map >= conf_thresh
    return pts[mask].astype(np.float32)


def _align_points(points: np.ndarray, pred_centers: np.ndarray, gt_centers: np.ndarray) -> np.ndarray:
    _, scale, rot, trans = _align_trajectories_sim3(pred_centers, gt_centers)
    return (scale * (rot @ points.T).T + trans).astype(np.float32)


def _point_metrics(pred_pts: np.ndarray, ref_pts: np.ndarray) -> dict:
    if len(pred_pts) == 0 or len(ref_pts) == 0:
        return {k: float("nan") for k in POINT_METRIC_FIELDS}
    tree_ref = cKDTree(ref_pts)
    tree_pred = cKDTree(pred_pts)
    d_pred_ref, _ = tree_ref.query(pred_pts, k=1)
    d_ref_pred, _ = tree_pred.query(ref_pts, k=1)
    acc = float(np.mean(d_pred_ref))
    acc_median = float(np.median(d_pred_ref))
    acc_p90 = float(np.percentile(d_pred_ref, 90.0))
    comp = float(np.mean(d_ref_pred))
    outlier_5cm = float(np.mean(d_pred_ref > 0.05))

    def _prf(thr: float) -> tuple[float, float, float]:
        prec = float(np.mean(d_pred_ref < thr))
        reca = float(np.mean(d_ref_pred < thr))
        f1 = 0.0 if (prec + reca) <= 1e-8 else 2.0 * prec * reca / (prec + reca)
        return prec, reca, float(f1)

    p5, r5, f5 = _prf(0.005)
    p10, r10, f10 = _prf(0.01)
    p20, r20, f20 = _prf(0.02)
    return {
        "acc": acc,
        "acc_median": acc_median,
        "acc_p90": acc_p90,
        "comp": comp,
        "fscore_5mm": f5,
        "fscore_1cm": f10,
        "fscore_2cm": f20,
        "precision_5mm": p5,
        "precision_1cm": p10,
        "precision_2cm": p20,
        "recall_5mm": r5,
        "recall_1cm": r10,
        "recall_2cm": r20,
        "outlier_5cm": outlier_5cm,
    }


def _load_robustnerf_entries(img_root: str, scene: str):
    scene_dir = Path(img_root) / scene
    img_dir = scene_dir / "images"
    gt_poses = read_colmap_poses(str(scene_dir / "sparse" / "0" / "images.bin"))
    entries = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = p.stem
        if stem not in gt_poses:
            continue
        low = stem.lower()
        if low.startswith("0clean"):
            state = "clean"
        elif low.startswith("1extra"):
            state = "extra"
        elif low.startswith("2clutter"):
            state = "clutter"
        else:
            state = "unknown"
        pair_id = stem[-3:] if stem[-3:].isdigit() else None
        entries.append({"stem": stem, "path": p, "center": gt_poses[stem], "state": state, "pair_id": pair_id})
    return entries


def _matched_pairs(img_root: str, scene: str):
    entries = _load_robustnerf_entries(img_root, scene)
    clean = {e["pair_id"]: e for e in entries if e["state"] == "clean" and e["pair_id"] is not None}
    clutter = {e["pair_id"]: e for e in entries if e["state"] == "clutter" and e["pair_id"] is not None}
    pair_ids = sorted(set(clean) & set(clutter))
    return [(clean[i], clutter[i]) for i in pair_ids]


def _read_images_bin(path: Path) -> dict[int, dict]:
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<i", f.read(4))[0]
            qvec = np.frombuffer(f.read(32), dtype=np.float64).copy()
            tvec = np.frombuffer(f.read(24), dtype=np.float64).copy()
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")
            n_pts = struct.unpack("<Q", f.read(8))[0]
            raw = np.frombuffer(f.read(n_pts * 24), dtype=np.float64).reshape(n_pts, 3)
            point3d_ids = raw[:, 2].view(np.int64).copy()
            images[img_id] = {"name": name, "camera_id": cam_id, "qvec": qvec, "tvec": tvec, "point3D_ids": point3d_ids}
    return images


def _read_points3d_bin(path: Path) -> dict[int, dict]:
    points = {}
    with open(path, "rb") as f:
        num_pts = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_pts):
            pt_id = struct.unpack("<Q", f.read(8))[0]
            xyz = np.frombuffer(f.read(24), dtype=np.float64).copy()
            f.read(3)
            f.read(8)
            n_obs = struct.unpack("<Q", f.read(8))[0]
            track = np.frombuffer(f.read(n_obs * 8), dtype=np.int32).reshape(n_obs, 2)
            points[pt_id] = {"xyz": xyz.astype(np.float32), "image_ids": track[:, 0].copy()}
    return points


def _load_phototourism_entries(img_root: str, scene: str):
    scene_dir = Path(img_root) / scene / "dense"
    img_dir = scene_dir / "images"
    images_bin = scene_dir / "sparse" / "images.bin"
    gt_poses = read_colmap_poses(str(images_bin))
    entries = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = p.stem
        if stem not in gt_poses:
            continue
        entries.append({"stem": stem, "path": p, "center": gt_poses[stem]})
    return entries


def _load_llff_entries(img_root: str, scene: str):
    scene_dir = Path(img_root) / scene
    img_dir = scene_dir / "images"
    images_bin = scene_dir / "sparse" / "0" / "images.bin"
    gt_poses = read_colmap_poses(str(images_bin))
    entries = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = p.stem
        if stem not in gt_poses:
            continue
        entries.append({"stem": stem, "path": p, "center": gt_poses[stem]})
    return entries


def _visible_sparse_points(colmap_root: str, scene: str, sample_stems: list[str], min_obs: int) -> np.ndarray:
    model_dir = Path(colmap_root) / scene / "dense" / "sparse"
    images = _read_images_bin(model_dir / "images.bin")
    points = _read_points3d_bin(model_dir / "points3D.bin")
    stem_to_id = {Path(info["name"]).stem: img_id for img_id, info in images.items()}
    sample_ids = [stem_to_id[s] for s in sample_stems if s in stem_to_id]
    sample_id_set = set(sample_ids)
    xyz = []
    for rec in points.values():
        hits = 0
        for image_id in rec["image_ids"]:
            if image_id in sample_id_set:
                hits += 1
                if hits >= min_obs:
                    xyz.append(rec["xyz"])
                    break
    return np.zeros((0, 3), dtype=np.float32) if not xyz else np.stack(xyz, axis=0).astype(np.float32)


def _visible_sparse_points_llff(colmap_root: str, scene: str, sample_stems: list[str], min_obs: int) -> np.ndarray:
    model_dir = Path(colmap_root) / scene / "sparse" / "0"
    images = _read_images_bin(model_dir / "images.bin")
    points = _read_points3d_bin(model_dir / "points3D.bin")
    stem_to_id = {Path(info["name"]).stem: img_id for img_id, info in images.items()}
    sample_ids = [stem_to_id[s] for s in sample_stems if s in stem_to_id]
    sample_id_set = set(sample_ids)
    xyz = []
    for rec in points.values():
        hits = 0
        for image_id in rec["image_ids"]:
            if image_id in sample_id_set:
                hits += 1
                if hits >= min_obs:
                    xyz.append(rec["xyz"])
                    break
    return np.zeros((0, 3), dtype=np.float32) if not xyz else np.stack(xyz, axis=0).astype(np.float32)


def _scenes_for_dataset(dataset: str, all_scenes: bool, scene: str) -> list[str]:
    if not all_scenes:
        return [scene]
    if dataset == "phototourism":
        return PHOTOTOURISM_SCENES
    if dataset == "robustnerf":
        return ROBUSTNERF_SCENES
    if dataset == "llff":
        return LLFF_SCENES
    raise ValueError(dataset)


def _list_bundle_dirs(bundle_root: str | Path, dataset: str, scene: str | None = None) -> list[Path]:
    root = Path(bundle_root)
    bundle_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "bundle_meta.json").exists())
    out = []
    for p in bundle_dirs:
        meta = json.loads((p / "bundle_meta.json").read_text())
        if meta.get("dataset") != dataset:
            continue
        if scene is not None and meta.get("scene") != scene:
            continue
        out.append(p)
    return out


def _extract_entropy_map(imgs: torch.Tensor, model, device: str, layer: int) -> np.ndarray:
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


def _threshold_for_target_count(vals: np.ndarray, target_count: int) -> float:
    flat = vals.reshape(-1).astype(np.float32)
    total = flat.size
    if target_count <= 0:
        return float("inf")
    if target_count >= total:
        return float("-inf")
    kth_index = max(total - target_count, 0)
    kth = np.partition(flat, kth_index)[kth_index]
    return float(kth)


def _compute_final_mask(conf_i: np.ndarray, ent_raw_i: np.ndarray, valid: np.ndarray, args) -> tuple[np.ndarray, dict]:
    floor_thr = float(conf_i[valid].min()) + float(args.conf_eps)
    floor_keep = valid & (conf_i > floor_thr)

    ent_act_i = activate_entropy_valid(ent_raw_i, valid, args.entropy_gain)
    masked_entropy = ent_act_i[floor_keep]
    masked_entropy = masked_entropy[np.isfinite(masked_entropy)]

    thresholds = np.linspace(float(masked_entropy.min()), float(masked_entropy.max()), args.curve_points, dtype=np.float32)
    counts = np.array([(masked_entropy > t).sum() for t in thresholds], dtype=np.int32)
    tol, idx = find_curvature_peak(thresholds, counts)

    log_conf = np.log(np.clip(conf_i[floor_keep] - 1.0 + args.conf_eps, 1e-8, None)).astype(np.float32)
    labels, best_k, _ = fit_gmm_labels(log_conf, args.min_k, args.max_k)
    label_map = np.full(conf_i.shape, -1, dtype=np.int32)
    label_map[floor_keep] = labels

    min_region_area = int(args.min_region_area_frac * conf_i.shape[0] * conf_i.shape[1])
    region_map = build_region_map(label_map, floor_keep, min_region_area)

    raw_mask = np.zeros_like(floor_keep, dtype=bool)
    for rid in sorted(int(r) for r in np.unique(region_map) if r >= 0):
        region = region_map == rid
        vals = ent_act_i[region]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if float(vals.mean()) <= float(tol):
            raw_mask |= region

    max_hole_area = int(args.max_hole_area_frac * conf_i.shape[0] * conf_i.shape[1])
    bridge_cut_mask, _ = split_thin_bridges(raw_mask & valid, valid, args.bridge_kernel, args.min_bridge_residue_area)
    bridge_filled_mask, _ = fill_small_holes(bridge_cut_mask, valid, max_hole_area)
    merged_mask = (raw_mask & valid) | bridge_filled_mask
    final_mask, _ = fill_component_holes(merged_mask, valid, max_hole_area)

    return final_mask.astype(bool), {
        "floor_threshold": floor_thr,
        "curvature_tolerance": float(tol),
        "curve_index": int(idx),
        "floor_keep_frac": float(floor_keep.sum() / valid.sum()),
        "final_keep_frac": float(final_mask.sum() / valid.sum()),
        "best_k": int(best_k),
    }


def _prepare_phototourism(scene: str, entries: list[dict], model, args, rng: random.Random):
    sample = rng.sample(entries, args.n_views)
    paths = [e["path"] for e in sample]
    stems = [e["stem"] for e in sample]
    gt_centers = np.array([e["center"] for e in sample], dtype=np.float32)
    ref_pts = _visible_sparse_points(args.colmap_root, scene, stems, args.photo_min_obs)
    imgs, valid_masks = load_crop_images_with_valid_masks(paths)
    pred, _ = run_pass1_with_layers(imgs, model, args.device, feat_layers=[])
    pred_centers = cameras_from_pred(pred, imgs.shape[2], imgs.shape[3])
    conf = pred["depth_conf"][0].detach().cpu().numpy().astype(np.float32)
    return sample, imgs, valid_masks.cpu().numpy().astype(bool), pred, pred_centers, gt_centers, conf, ref_pts


def _prepare_phototourism_bundle(bundle_dir: Path, model, args):
    meta = json.loads((bundle_dir / "bundle_meta.json").read_text())
    scene = str(meta["scene"])
    sample_stems = list(meta["sample_stems"])
    img_paths = []
    for stem in sample_stems:
        matches = sorted((bundle_dir / "images").glob(f"{stem}.*"))
        if not matches:
            raise FileNotFoundError(f"Could not find image for stem {stem} in {bundle_dir / 'images'}")
        img_paths.append(matches[0])

    gt_pose_map = read_colmap_poses(str(Path(args.colmap_root) / scene / "dense" / "sparse" / "images.bin"))
    gt_centers = np.array([gt_pose_map[stem] for stem in sample_stems], dtype=np.float32)
    ref_pts = _visible_sparse_points(args.colmap_root, scene, sample_stems, args.photo_min_obs)
    imgs, valid_masks = load_crop_images_with_valid_masks(img_paths)
    pred, _ = run_pass1_with_layers(imgs, model, args.device, feat_layers=[])
    pred_centers = cameras_from_pred(pred, imgs.shape[2], imgs.shape[3])
    conf = pred["depth_conf"][0].detach().cpu().numpy().astype(np.float32)
    return meta, imgs, valid_masks.cpu().numpy().astype(bool), pred, pred_centers, gt_centers, conf, ref_pts


def _prepare_llff(scene: str, entries: list[dict], model, args, rng: random.Random):
    sample = rng.sample(entries, args.n_views)
    paths = [e["path"] for e in sample]
    stems = [e["stem"] for e in sample]
    gt_centers = np.array([e["center"] for e in sample], dtype=np.float32)
    ref_pts = _visible_sparse_points_llff(args.colmap_root, scene, stems, args.photo_min_obs)
    imgs, valid_masks = load_crop_images_with_valid_masks(paths)
    pred, _ = run_pass1_with_layers(imgs, model, args.device, feat_layers=[])
    pred_centers = cameras_from_pred(pred, imgs.shape[2], imgs.shape[3])
    conf = pred["depth_conf"][0].detach().cpu().numpy().astype(np.float32)
    return sample, imgs, valid_masks.cpu().numpy().astype(bool), pred, pred_centers, gt_centers, conf, ref_pts


def _prepare_robustnerf(scene: str, pairs: list[tuple[dict, dict]], model, args, rng: random.Random):
    sample = rng.sample(pairs, args.n_views)
    clean_paths = [c["path"] for c, _ in sample]
    clutter_paths = [k["path"] for _, k in sample]
    gt_centers = np.array([c["center"] for c, _ in sample], dtype=np.float32)

    clean_imgs, _ = load_crop_images_with_valid_masks(clean_paths)
    clutter_imgs, valid_masks = load_crop_images_with_valid_masks(clutter_paths)
    clean_pred, _ = run_pass1_with_layers(clean_imgs, model, args.device, feat_layers=[])
    clutter_pred, _ = run_pass1_with_layers(clutter_imgs, model, args.device, feat_layers=[])

    clean_conf = clean_pred["depth_conf"][0].detach().cpu().numpy().astype(np.float32)
    clutter_conf = clutter_pred["depth_conf"][0].detach().cpu().numpy().astype(np.float32)
    clean_centers = cameras_from_pred(clean_pred, clean_imgs.shape[2], clean_imgs.shape[3])
    clutter_centers = cameras_from_pred(clutter_pred, clutter_imgs.shape[2], clutter_imgs.shape[3])

    ref_pts = _extract_points(clean_pred, clean_conf, float("-inf"))
    ref_pts = _align_points(ref_pts, clean_centers, gt_centers)
    return sample, clutter_imgs, valid_masks.cpu().numpy().astype(bool), clutter_pred, clutter_centers, gt_centers, clutter_conf, ref_pts


def _evaluate_masks(
    pred: dict,
    pred_centers: np.ndarray,
    gt_centers: np.ndarray,
    conf: np.ndarray,
    valid_masks: np.ndarray,
    floor_masks: np.ndarray,
    final_masks: np.ndarray,
    ref_pts: np.ndarray,
):
    raw_conf = np.where(valid_masks, conf, float("-inf")).astype(np.float32)
    final_count = int(final_masks.sum())
    topk_thr = _threshold_for_target_count(conf[valid_masks], final_count)
    topk_mask = valid_masks & (conf >= topk_thr)

    pts_raw = _extract_points(pred, raw_conf, float("-inf"))
    pts_floor = _extract_points(pred, floor_masks.astype(np.float32), 0.5)
    pts_topk = _extract_points(pred, topk_mask.astype(np.float32), 0.5)
    pts_final = _extract_points(pred, final_masks.astype(np.float32), 0.5)

    pts_raw = _align_points(pts_raw, pred_centers, gt_centers)
    pts_floor = _align_points(pts_floor, pred_centers, gt_centers)
    pts_topk = _align_points(pts_topk, pred_centers, gt_centers)
    pts_final = _align_points(pts_final, pred_centers, gt_centers)

    return {
        "raw": {"n_points": int(len(pts_raw)), **_point_metrics(pts_raw, ref_pts)},
        "floor_only": {"n_points": int(len(pts_floor)), **_point_metrics(pts_floor, ref_pts)},
        "confidence_topk": {"n_points": int(len(pts_topk)), **_point_metrics(pts_topk, ref_pts)},
        "discord": {"n_points": int(len(pts_final)), **_point_metrics(pts_final, ref_pts)},
        "topk_threshold": float(topk_thr),
    }


def main():
    args = parse_args()
    default_img_root, default_colmap_root = _default_roots(args.dataset)
    args.img_root = args.img_root or default_img_root
    args.colmap_root = args.colmap_root or default_colmap_root
    args.gmm_max_fit_points = None if int(args.gmm_max_fit_points) <= 0 else int(args.gmm_max_fit_points)

    scenes = _scenes_for_dataset(args.dataset, args.all_scenes, args.scene)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("Loading VGGT-1B...")
    model = VGGT.from_pretrained("facebook/VGGT-1B", local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)

    rng = random.Random(args.seed)
    rows = []
    per_trial_dump = []

    if args.bundle_root:
        bundle_scene = None if args.all_scenes else args.scene
        bundle_dirs = _list_bundle_dirs(args.bundle_root, args.dataset, bundle_scene)
        if not bundle_dirs:
            raise RuntimeError(f"No bundle directories found under {args.bundle_root} for dataset={args.dataset}")
        bundle_dirs = sorted(bundle_dirs)
        trial_items = [("bundle", bundle_dir) for bundle_dir in bundle_dirs]
    else:
        trial_items = []
        for scene in scenes:
            if args.dataset == "phototourism":
                entries = _load_phototourism_entries(args.img_root, scene)
                if len(entries) < args.n_views:
                    print(f"{scene}: only {len(entries)} images, skipping")
                    continue
                trial_items.extend([("phototourism", (scene, entries, trial)) for trial in range(args.n_trials)])
            elif args.dataset == "llff":
                entries = _load_llff_entries(args.img_root, scene)
                if len(entries) < args.n_views:
                    print(f"{scene}: only {len(entries)} images, skipping")
                    continue
                trial_items.extend([("llff", (scene, entries, trial)) for trial in range(args.n_trials)])
            else:
                pairs = _matched_pairs(args.img_root, scene)
                if len(pairs) < args.n_views:
                    print(f"{scene}: only {len(pairs)} matched pairs, skipping")
                    continue
                trial_items.extend([("robustnerf", (scene, pairs, trial)) for trial in range(args.n_trials)])

    for kind, payload in trial_items:
        if kind == "bundle":
            bundle_dir = payload
            meta, imgs, valid_masks, pred, pred_centers, gt_centers, conf, ref_pts = _prepare_phototourism_bundle(bundle_dir, model, args)
            scene = str(meta["scene"])
            trial = int(meta["trial"])
            sample_stems = list(meta["sample_stems"])
            n_views = int(meta.get("n_views", len(sample_stems)))
        elif kind == "phototourism":
            scene, entries, trial = payload
            sample, imgs, valid_masks, pred, pred_centers, gt_centers, conf, ref_pts = _prepare_phototourism(scene, entries, model, args, rng)
            sample_stems = [e["stem"] for e in sample]
            n_views = int(args.n_views)
        elif kind == "llff":
            scene, entries, trial = payload
            sample, imgs, valid_masks, pred, pred_centers, gt_centers, conf, ref_pts = _prepare_llff(scene, entries, model, args, rng)
            sample_stems = [e["stem"] for e in sample]
            n_views = int(args.n_views)
        else:
            scene, pairs, trial = payload
            sample, imgs, valid_masks, pred, pred_centers, gt_centers, conf, ref_pts = _prepare_robustnerf(scene, pairs, model, args, rng)
            sample_stems = [k["stem"] for _, k in sample]
            n_views = int(args.n_views)

        outputs = compute_final_outputs(
            model=model,
            imgs=imgs,
            valid_masks=torch.from_numpy(valid_masks),
            conf=conf,
            device=args.device,
            entropy_layer=args.entropy_layer,
            entropy_gain=args.entropy_gain,
            conf_eps=args.conf_eps,
            curve_points=args.curve_points,
            min_k=args.min_k,
            max_k=args.max_k,
            min_region_area_frac=args.min_region_area_frac,
            max_hole_area_frac=args.max_hole_area_frac,
            gmm_max_fit_points=args.gmm_max_fit_points,
            bridge_kernel=args.bridge_kernel,
            min_bridge_residue_area=args.min_bridge_residue_area,
            split_region_bridges=args.split_region_bridges,
            trust_stat=args.trust_stat,
            trust_quantile=args.trust_quantile,
        )

        floor_masks = outputs["floor_keep"].astype(bool)
        final_masks = outputs["final_mask"].astype(bool)
        view_meta = outputs["meta"]

        metrics = _evaluate_masks(pred, pred_centers, gt_centers, conf, valid_masks, floor_masks, final_masks, ref_pts)

        per_trial_dump.append(
            {
                "dataset": args.dataset,
                "scene": scene,
                "trial": int(trial),
                "sample_stems": sample_stems,
                "view_meta": view_meta,
                "timing": outputs["timing"],
                "metrics": metrics,
            }
        )

        for method, m in metrics.items():
            if method == "topk_threshold":
                continue
            row = {
                "dataset": args.dataset,
                "scene": scene,
                "trial": int(trial),
                "method": method,
                "n_views": int(n_views),
                "n_ref_points": int(len(ref_pts)),
                "n_points": int(m["n_points"]),
                "topk_threshold": float(metrics["topk_threshold"]),
            }
            for key in POINT_METRIC_FIELDS:
                row[key] = float(m[key])
            rows.append(row)

        print(
            f"{scene} trial {int(trial)+1}: "
            f"floor f1={metrics['floor_only']['fscore_1cm']:.4f}, "
            f"conf-topk f1={metrics['confidence_topk']['fscore_1cm']:.4f}, "
            f"discord f1={metrics['discord']['fscore_1cm']:.4f}"
        )

    with out_csv.open("w", newline="") as f:
        fieldnames = ["dataset", "scene", "trial", "method", "n_views", "n_ref_points", "n_points", *POINT_METRIC_FIELDS, "topk_threshold"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for scene in sorted(set((r["dataset"], r["scene"], r["method"]) for r in rows)):
        dataset, scene_name, method = scene
        subset = [r for r in rows if r["dataset"] == dataset and r["scene"] == scene_name and r["method"] == method]
        summary_rows.append(
            {
                "dataset": dataset,
                "scene": scene_name,
                "method": method,
                "n_trials": len(subset),
                "n_points": float(np.mean([r["n_points"] for r in subset])),
                **{key: float(np.mean([r[key] for r in subset])) for key in POINT_METRIC_FIELDS},
            }
        )

    summary_path = out_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"rows": summary_rows, "trials": per_trial_dump}, indent=2))
    print(f"Saved results to {out_csv}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
