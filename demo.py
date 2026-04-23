"""
DISCORD Gradio Demo
===================
Interactive reconstruction demo for the public DISCORD code release.

Supported modes:
  - Baseline VGGT: plain VGGT reconstruction
  - Floor-only: confidence floor filtering without entropy
  - DISCORD: the final confidence+entropy trust-segmentation pipeline
"""

import base64
import json
import os
import cv2
import math
import shutil
import glob
import gc
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Gradio stages uploads into its own temp cache before our app copies files into
_HERE = os.path.dirname(os.path.abspath(__file__))
_GRADIO_TEMP_DIR = os.path.join(_HERE, "gradio_tmp")
os.makedirs(_GRADIO_TEMP_DIR, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", _GRADIO_TEMP_DIR)
_OUTPUT_ROOT = Path(_HERE) / "outputs"
_UPLOAD_ROOT = _OUTPUT_ROOT / "uploads"
_DEMO_RUN_ROOT = _OUTPUT_ROOT / "demo_runs"
_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
_DEMO_RUN_ROOT.mkdir(parents=True, exist_ok=True)

import gradio as gr

from discord3d.third_party import setup_vggt_paths

setup_vggt_paths()

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from discord3d.pipeline.common import image_overlay, load_crop_images_with_valid_masks
from discord3d.pipeline.runtime import (
    apply_final_masks_to_predictions,
    build_method_gallery,
    compute_final_outputs,
    parse_entropy_layer,
)

# device / dtype 
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)

print("Loading VGGT model (facebook/VGGT-1B)…")
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()
model.requires_grad_(False)
print("Model ready.")

TRIAL_BUNDLE_ROOT = Path(_HERE) / "examples" / "phototourism_nv5_t3"


CAPTURE_VIEW_JS = r"""
(runDir, _snapshotData, _snapshotPose) => {
  const root = document.querySelector("#reconstruction_viewer");
  const canvas = root?.querySelector("canvas");
  if (!canvas) {
    return [runDir, "", JSON.stringify({ capture_error: "No 3D canvas found. Please reconstruct a scene first." })];
  }

  let dataUrl = "";
  try {
    dataUrl = canvas.toDataURL("image/png");
  } catch (err) {
    return [runDir, "", JSON.stringify({ capture_error: `Canvas capture failed: ${String(err)}` })];
  }

  let pose = {
    width: canvas.width || null,
    height: canvas.height || null,
  };

  try {
    const B = window.BABYLON || window.babylonjs;
    const scene =
      B?.EngineStore?.LastCreatedScene ||
      B?.EngineStore?._LastCreatedScene ||
      null;
    const camera = scene?.activeCamera || null;
    if (camera) {
      const toDeg = (rad) => (rad == null ? null : rad * 180.0 / Math.PI);
      pose = {
        ...pose,
        alpha_deg: toDeg(camera.alpha),
        beta_deg: toDeg(camera.beta),
        radius: camera.radius ?? null,
        fov_deg: toDeg(camera.fov),
        target: camera.target
          ? { x: camera.target.x, y: camera.target.y, z: camera.target.z }
          : null,
        position: camera.position
          ? { x: camera.position.x, y: camera.position.y, z: camera.position.z }
          : null,
        name: camera.name || null,
      };
    }
  } catch (err) {
    pose = { ...pose, pose_error: String(err) };
  }

  return [runDir, dataUrl, JSON.stringify(pose)];
}
"""


LIVE_CAMERA_POSE_JS = r"""
(prevPoseJson) => {
  let pose = {
    available: false,
  };

  try {
    const B = window.BABYLON || window.babylonjs;
    const scene =
      B?.EngineStore?.LastCreatedScene ||
      B?.EngineStore?._LastCreatedScene ||
      null;
    const camera = scene?.activeCamera || null;
    if (camera) {
      const toDeg = (rad) => (rad == null ? null : rad * 180.0 / Math.PI);
      pose = {
        available: true,
        alpha_deg: toDeg(camera.alpha),
        beta_deg: toDeg(camera.beta),
        radius: camera.radius ?? null,
        fov_deg: toDeg(camera.fov),
        target: camera.target
          ? { x: camera.target.x, y: camera.target.y, z: camera.target.z }
          : null,
        position: camera.position
          ? { x: camera.position.x, y: camera.position.y, z: camera.position.z }
          : null,
        camera_position_deg: [
          toDeg(camera.alpha),
          toDeg(camera.beta),
          camera.radius ?? null,
        ],
        name: camera.name || null,
      };
    }
  } catch (err) {
    pose = {
      available: false,
      pose_error: String(err),
    };
  }

  return [JSON.stringify(pose)];
}
"""


# Core helpers
def _compute_view_scores(images_dev, attn_a=0.5, cos_a=0.5):
    """
    First forward pass. Returns combined_scores (N,) and raw predictions.
    Anchor = image index 0.
    """
    aggregator  = model.aggregator
    patch_size  = aggregator.patch_size
    patch_start = aggregator.patch_start_idx
    H, W        = images_dev.shape[-2:]
    h_p, w_p    = H // patch_size, W // patch_size
    num_patch   = h_p * w_p
    tok_per_img = patch_start + num_patch
    N = images_dev.shape[0]

    q_out, k_out = {}, {}
    handles = []
    blk_attn = model.aggregator.global_blocks[23].attn
    handles.append(blk_attn.q_norm.register_forward_hook(
        lambda m, i, o: q_out.__setitem__(23, o.detach())))
    handles.append(blk_attn.k_norm.register_forward_hook(
        lambda m, i, o: k_out.__setitem__(23, o.detach())))

    with torch.inference_mode():
        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, _ = model.aggregator(images_dev.unsqueeze(0))
            predictions = model(images_dev.unsqueeze(0))

    for h in handles:
        h.remove()

    feat_all = aggregated_tokens_list[23][0, :, patch_start:, 1024:].float()
    feat_norm = F.normalize(feat_all, p=2, dim=-1)
    ref_norm  = feat_norm[0:1]
    cos_scores = (feat_norm * ref_norm).sum(-1).mean(-1)

    Q = q_out[23]
    K = k_out[23]
    q_anchor = Q[:, :, patch_start:tok_per_img, :]
    scale    = 1.0 / math.sqrt(float(q_anchor.shape[-1]))
    logits   = torch.einsum("bhqd,bhtd->bhqt", q_anchor, K) * scale
    probs    = torch.softmax(logits, dim=-1).mean(1).mean(1)[0]

    attn_scores = []
    for i in range(N):
        start = i * tok_per_img + patch_start
        end   = min(start + num_patch, probs.shape[-1])
        attn_scores.append(probs[start:end].mean().item())
    attn_scores = torch.tensor(attn_scores, dtype=torch.float32)

    norm01 = lambda t: (t - t.min()) / (t.max() - t.min() + 1e-6)
    combined = attn_a * norm01(attn_scores) + cos_a * norm01(cos_scores.cpu())

    del aggregated_tokens_list, feat_all, feat_norm, ref_norm
    del q_out, k_out, Q, K, q_anchor, logits, probs
    torch.cuda.empty_cache()

    return combined, predictions


def save_canvas_snapshot(run_dir, snapshot_data, snapshot_pose_json):
    if not run_dir or run_dir == "None":
        return "No active run directory found. Reconstruct a scene first."

    pose = None
    if snapshot_pose_json:
        try:
            pose = json.loads(snapshot_pose_json)
        except Exception:
            pose = {"raw": snapshot_pose_json}

    if isinstance(pose, dict) and pose.get("capture_error"):
        return pose["capture_error"]
    if not snapshot_data:
        return "No canvas snapshot was captured."

    run_path = Path(run_dir)
    snap_dir = run_path / "camera_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = snap_dir / f"view_{timestamp}.png"
    meta_path = snap_dir / f"view_{timestamp}.json"

    try:
        if snapshot_data.startswith("data:image"):
            _, payload = snapshot_data.split(",", 1)
        else:
            payload = snapshot_data
        image_path.write_bytes(base64.b64decode(payload))
    except Exception as exc:
        return f"Failed to save canvas snapshot: {exc}"

    meta = {
        "timestamp": timestamp,
        "image_path": str(image_path),
        "pose": pose,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[camera-snapshot] saved image={image_path} meta={meta_path}", flush=True)
    return f"Saved current view to {image_path}"


def _gallery_from_paths(image_paths):
    names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    return [(str(p), f"[{i}] {n}") for i, (p, n) in enumerate(zip(image_paths, names))]


def _trial_bundle_dirs():
    if not TRIAL_BUNDLE_ROOT.exists():
        return []
    return [p for p in sorted(TRIAL_BUNDLE_ROOT.iterdir()) if (p / "bundle_meta.json").exists()]


def _read_trial_bundle_meta(bundle_name):
    if not bundle_name:
        return None, None
    bundle_dir = TRIAL_BUNDLE_ROOT / bundle_name
    meta_path = bundle_dir / "bundle_meta.json"
    if not meta_path.exists():
        return None, None
    return bundle_dir, json.loads(meta_path.read_text())


def _trial_bundle_info_text(meta):
    metrics = meta.get("metrics", {})
    stems = meta.get("sample_stems", [])
    stem_text = ", ".join(stems)
    return (
        f"**{meta.get('scene', 'unknown')}**, trial `{meta.get('trial', '?')}`  \n"
        f"raw F-score `{metrics.get('raw_fscore_1cm', float('nan')):.6f}` | "
        f"confidence-only (matched budget) `{metrics.get('confidence_topk_fscore_1cm', float('nan')):.6f}` | "
        f"DISCORD `{metrics.get('discord_fscore_1cm', float('nan')):.6f}` | "
        f"Δ(DISCORD-conf) `{metrics.get('discord_minus_conf_fscore_1cm', float('nan')):+.6f}`  \n"
        f"Views: {stem_text}"
    )


def refresh_trial_bundle_choices():
    choices = [p.name for p in _trial_bundle_dirs()]
    value = choices[0] if choices else None
    info = (
        f"Found {len(choices)} prepared trial bundles under `{TRIAL_BUNDLE_ROOT}`."
        if choices
        else f"No prepared trial bundles found under `{TRIAL_BUNDLE_ROOT}`."
    )
    return gr.Dropdown(choices=choices, value=value), info


def preview_trial_bundle(bundle_name):
    bundle_dir, meta = _read_trial_bundle_meta(bundle_name)
    if bundle_dir is None or meta is None:
        return [], "No trial bundle selected."
    image_paths = sorted(glob.glob(os.path.join(bundle_dir, "images", "*")))
    return _gallery_from_paths(image_paths), _trial_bundle_info_text(meta)


def load_trial_bundle(bundle_name):
    bundle_dir, meta = _read_trial_bundle_meta(bundle_name)
    if bundle_dir is None or meta is None:
        return None, None, [], [], "Select a valid trial bundle first.", gr.CheckboxGroup(choices=[])

    image_paths = sorted(glob.glob(os.path.join(bundle_dir, "images", "*")))
    gallery = _gallery_from_paths(image_paths)
    names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    log = (
        f"Loaded trial bundle `{bundle_name}` with {len(image_paths)} images. "
        "Select a method and click **Reconstruct**."
    )
    return (
        None,
        str(bundle_dir),
        gallery,
        [],
        log,
        gr.CheckboxGroup(choices=names, value=[]),
    )


def format_live_camera_pose(pose_json):
    if not pose_json:
        return "Current camera pose: unavailable.", pose_json
    try:
        pose = json.loads(pose_json)
    except Exception:
        return f"Current camera pose: raw={pose_json}", pose_json

    if not isinstance(pose, dict) or not pose.get("available"):
        if isinstance(pose, dict) and pose.get("pose_error"):
            return f"Current camera pose: unavailable ({pose['pose_error']}).", pose_json
        return "Current camera pose: unavailable.", pose_json

    alpha = pose.get("alpha_deg")
    beta = pose.get("beta_deg")
    radius = pose.get("radius")
    fov = pose.get("fov_deg")
    pos = pose.get("position")
    target = pose.get("target")

    def _fmt(v):
        return "None" if v is None else f"{float(v):.2f}"

    msg = (
        f"Current camera pose: "
        f"`alpha={_fmt(alpha)} deg`, "
        f"`beta={_fmt(beta)} deg`, "
        f"`radius={_fmt(radius)}`, "
        f"`fov={_fmt(fov)} deg`"
    )
    if isinstance(pos, dict):
        msg += (
            f" | pos=(`{_fmt(pos.get('x'))}`, `{_fmt(pos.get('y'))}`, `{_fmt(pos.get('z'))}`)"
        )
    if isinstance(target, dict):
        msg += (
            f" | target=(`{_fmt(target.get('x'))}`, `{_fmt(target.get('y'))}`, `{_fmt(target.get('z'))}`)"
        )
    return msg, pose_json


def _postprocess_predictions(predictions, image_hw):
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], image_hw)
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().float().numpy().squeeze(0)

    predictions["pose_enc_list"] = None
    depth = predictions["depth"]
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        depth, predictions["extrinsic"], predictions["intrinsic"])

    # Diagnostic: log shapes so we can catch mismatches
    for key in ["world_points", "world_points_conf", "depth", "depth_conf",
                "world_points_from_depth", "images", "extrinsic"]:
        if key in predictions and isinstance(predictions[key], np.ndarray):
            print(f"  predictions[{key!r}].shape = {predictions[key].shape}")

    return predictions


def _align_prediction_shapes(predictions):
    """Ensure world_points/depth and their conf arrays have matching view counts.

    If a mismatch is detected (e.g. from mixed forward-pass state), log a warning
    and regenerate the conf array as ones so GLB rendering can proceed.
    """
    for pts_key, conf_key in [("world_points", "world_points_conf"),
                               ("world_points_from_depth", "depth_conf")]:
        pts = predictions.get(pts_key)
        conf = predictions.get(conf_key)
        if pts is None or conf is None:
            continue
        # pts is (S, H, W, 3) or similar; conf is (S, H, W)
        # Their total pixels must match: pts.size/3 == conf.size
        n_pts = pts.size // 3 if pts.shape[-1] == 3 else pts.size // pts.shape[-1]
        n_conf = conf.size
        if n_pts != n_conf:
            print(f"  WARNING: shape mismatch {pts_key} {pts.shape} vs "
                  f"{conf_key} {conf.shape} — regenerating conf as ones")
            # Use pts shape to create matching conf
            if pts.shape[-1] == 3:
                predictions[conf_key] = np.ones(pts.shape[:-1], dtype=pts.dtype)
            else:
                predictions[conf_key] = np.ones_like(pts[..., 0])


def _parse_layers(layer_str):
    """Parse layer string like '20-23' or '23' or '20,21,22,23' into list of ints."""
    layer_str = layer_str.strip()
    if not layer_str:
        return [23]
    if "-" in layer_str and "," not in layer_str:
        parts = layer_str.split("-")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(x.strip()) for x in layer_str.split(",")]


def _apply_entropy_method(
    predictions,
    images,
    valid_masks,
    names,
    run_dir,
    entropy_layers,
    bridge_kernel=3,
    trust_quantile=0.9,
    fast_region_mode=True,
):
    entropy_layer = parse_entropy_layer(entropy_layers)
    bridge_kernel = max(3, int(round(float(bridge_kernel))))
    if bridge_kernel % 2 == 0:
        bridge_kernel += 1
    trust_quantile = float(np.clip(trust_quantile, 0.5, 0.99))
    gmm_max_fit_points = 20000 if fast_region_mode else None
    outputs = compute_final_outputs(
        model,
        images,
        valid_masks,
        predictions["depth_conf"].astype(np.float32),
        device,
        entropy_layer=entropy_layer,
        split_region_bridges=True,
        bridge_kernel=bridge_kernel,
        trust_stat="quantile",
        trust_quantile=trust_quantile,
        gmm_max_fit_points=gmm_max_fit_points,
    )
    predictions = apply_final_masks_to_predictions(predictions, outputs["final_mask"])
    predictions["entropy_layer"] = np.array([int(entropy_layer)], dtype=np.int32)
    predictions["final_mask"] = outputs["final_mask"].astype(np.uint8)

    gallery = build_method_gallery(run_dir, names, outputs)
    meta = {
        "entropy_layer": int(entropy_layer),
        "split_region_bridges": True,
        "bridge_kernel": int(bridge_kernel),
        "trust_stat": "quantile",
        "trust_quantile": float(trust_quantile),
        "fast_region_mode": bool(fast_region_mode),
        "timing": outputs.get("timing", {}),
        "views": [
            {"view": n, **m} for n, m in zip(names, outputs["meta"])
        ],
    }
    with open(os.path.join(run_dir, "final_pipeline_meta.json"), "w") as f:
        import json
        json.dump(meta, f, indent=2)

    return predictions, outputs, gallery


def _apply_floor_only_method(predictions, images, valid_masks, names, run_dir, conf_eps=1e-5):
    conf = predictions["depth_conf"].astype(np.float32)
    valid_np = valid_masks.detach().cpu().numpy().astype(bool)
    final_mask = np.zeros_like(conf, dtype=np.uint8)
    gallery = []
    gallery_dir = Path(run_dir) / "method_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(names):
        valid = valid_np[i]
        floor_thr = float(conf[i][valid].min()) + float(conf_eps)
        keep = valid & (conf[i] > floor_thr)
        final_mask[i] = keep.astype(np.uint8)
        overlay = image_overlay(np.clip(images[i].cpu().numpy().transpose(1, 2, 0), 0.0, 1.0), keep)
        out_path = gallery_dir / f"{i:02d}_{name}_floor_only.png"
        cv2.imwrite(str(out_path), cv2.cvtColor((overlay * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))
        gallery.append((str(out_path), f"[{i}] {name} — floor-only"))

    predictions = apply_final_masks_to_predictions(predictions, final_mask.astype(bool))
    predictions["final_mask"] = final_mask
    return predictions, gallery



# File handling
def handle_uploads(input_video, input_images):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = str(_UPLOAD_ROOT / f"input_images_{timestamp}")
    target_dir_images = os.path.join(target_dir, "images")
    os.makedirs(target_dir_images, exist_ok=True)

    image_paths = []
    if input_images is not None:
        for file_data in input_images:
            file_path = file_data["name"] if isinstance(file_data, dict) else file_data
            dst = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst)
            image_paths.append(dst)

    if input_video is not None:
        video_path = input_video["name"] if isinstance(input_video, dict) else input_video
        vs  = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps))
        count = frame_num = 0
        while True:
            ok, frame = vs.read()
            if not ok:
                break
            count += 1
            if count % frame_interval == 0:
                p = os.path.join(target_dir_images, f"{frame_num:06d}.png")
                cv2.imwrite(p, frame)
                image_paths.append(p)
                frame_num += 1

    return target_dir, sorted(image_paths)


def on_upload(input_video, input_images):
    """When the user uploads media, populate the gallery and target directory."""
    if not input_video and not input_images:
        return None, None, [], [], "Upload images first.", gr.CheckboxGroup(choices=[])

    target_dir, image_paths = handle_uploads(input_video, input_images)
    names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    gallery = [(p, f"[{i}] {n}") for i, (p, n) in enumerate(zip(image_paths, names))]

    return (
        None,  # clear 3D viewer
        target_dir,
        gallery,
        [],
        f"Uploaded {len(image_paths)} images. Select a method and click **Reconstruct**.",
        gr.CheckboxGroup(choices=names, value=[]),
    )


# Main reconstruction callback
def gradio_reconstruct(
    target_dir,
    method,
    cond_views_selected,
    layer_str,
    entropy_layer_str,
    bridge_kernel,
    trust_quantile,
    fast_region_mode,
    entropy_action,
    entropy_thresh,
    proj_mode,
    beta,
    svd_rank,
    rej_thresh,
    attn_a,
    conf_thres,
    frame_filter,
    mask_black_bg,
    mask_white_bg,
    show_cam,
    mask_sky,
    prediction_mode,
):
    if not target_dir or not os.path.isdir(target_dir):
        return None, "No valid directory. Upload images first.", None, None, None, None

    gc.collect()
    torch.cuda.empty_cache()
    t0 = time.time()

    # Each combination of settings gets its own subdirectory — no cache collisions
    slug = _run_slug(
        method,
        cond_views_selected or [],
        layer_str,
        entropy_layer_str,
        bridge_kernel,
        trust_quantile,
        fast_region_mode,
        entropy_action,
        entropy_thresh,
        proj_mode,
        beta,
        svd_rank,
        rej_thresh,
        attn_a,
    )
    target_path = Path(target_dir).resolve()
    try:
        is_curated_bundle = TRIAL_BUNDLE_ROOT.resolve() in target_path.parents
    except Exception:
        is_curated_bundle = False
    if is_curated_bundle:
        run_dir = str(_DEMO_RUN_ROOT / target_path.name / slug)
    else:
        run_dir = os.path.join(target_dir, slug)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[run_dir] {run_dir}")

    image_paths = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    if not image_paths:
        return None, "No images found.", None, None, None, None

    names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    N = len(image_paths)
    print(f"\n[{method}] {N} images: {names}")

    images, valid_masks = load_crop_images_with_valid_masks([Path(p) for p in image_paths])
    image_hw = tuple(int(d) for d in images.shape[-2:])
    images_dev = images.to(device, dtype=dtype)

    # Route by method
    if method == "Baseline VGGT":
        # Plain VGGT — no intervention
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images_dev.unsqueeze(0))
        predictions = _postprocess_predictions(predictions, image_hw)
        predictions["image_files"] = np.array([os.path.basename(p) for p in image_paths])
        gallery = [(p, f"[{i}] {n}") for i, (p, n) in enumerate(zip(image_paths, names))]
        entropy_gallery = []
        log = f"Baseline VGGT — {N} views, {time.time()-t0:.1f}s"

    elif method == "Floor-only":
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images_dev.unsqueeze(0))

        predictions = _postprocess_predictions(predictions, image_hw)
        predictions["image_files"] = np.array([os.path.basename(p) for p in image_paths])
        predictions, entropy_gallery = _apply_floor_only_method(predictions, images, valid_masks, names, run_dir)
        gallery = [(p, f"[{i}] {n}") for i, (p, n) in enumerate(zip(image_paths, names))]
        log = f"Floor-only — confidence floor filtering, {time.time()-t0:.1f}s"

    elif method == "DISCORD":
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images_dev.unsqueeze(0))

        predictions = _postprocess_predictions(predictions, image_hw)
        predictions["image_files"] = np.array([os.path.basename(p) for p in image_paths])
        predictions, pipeline_outputs, entropy_gallery = _apply_entropy_method(
            predictions, images, valid_masks, names, run_dir, entropy_layer_str, bridge_kernel, trust_quantile, fast_region_mode
        )
        gallery = [(p, f"[{i}] {n}") for i, (p, n) in enumerate(zip(image_paths, names))]
        mean_keep = float(np.mean([m["final_keep_frac"] for m in pipeline_outputs["meta"]]))
        mean_tol = float(np.mean([m["curvature_tolerance"] for m in pipeline_outputs["meta"]]))
        timing = pipeline_outputs.get("timing", {})
        log = (
            f"DISCORD — final pipeline, L{parse_entropy_layer(entropy_layer_str)}, "
            f"bridge=on(k={int(round(float(bridge_kernel)))}, q={float(trust_quantile):.2f}), "
            f"mode={'fast' if fast_region_mode else 'exact'}, "
            f"mean keep={mean_keep:.3f}, mean tol={mean_tol:.3f}, "
            f"entropy={timing.get('entropy_compute_s', 0.0):.1f}s, "
            f"region={timing.get('region_post_s', 0.0):.1f}s, "
            f"total={time.time()-t0:.1f}s"
        )

    else:
        return None, f"Unsupported method in the public DISCORD demo: {method}", None, None, None, None

    # Defensive: ensure world_points and world_points_conf have matching S
    _align_prediction_shapes(predictions)

    # Save predictions into run_dir (settings-specific, no collision)
    np.savez(os.path.join(run_dir, "predictions.npz"), **{
        k: v for k, v in predictions.items() if isinstance(v, np.ndarray)
    })

    S_pred = predictions["world_points"].shape[0] if "world_points" in predictions else N
    ff_choices = ["All"] + [f"{i}: view {i}" for i in range(S_pred)]

    glbfile = _make_glb(run_dir, target_dir, predictions, conf_thres,
                        frame_filter or "All",
                        mask_black_bg, mask_white_bg,
                        show_cam, mask_sky, prediction_mode)

    gc.collect()
    torch.cuda.empty_cache()

    return (
        glbfile,
        log,
        gr.Dropdown(choices=ff_choices, value="All", interactive=True),
        gallery,
        entropy_gallery,
        run_dir,   # stored in hidden box for re-visualization
    )


def _run_slug(method, cond_views_selected, layer_str, entropy_layer_str, bridge_kernel, trust_quantile, fast_region_mode, entropy_action,
              entropy_thresh, proj_mode, beta, svd_rank, rej_thresh, attn_a):
    """Build a short, filesystem-safe string that uniquely identifies a run's settings."""
    safe = lambda s: str(s).replace(".", "p").replace(" ", "").replace(",", "-").replace("/", "")
    if method == "Baseline VGGT":
        return "baseline"
    elif method == "Floor-only":
        return "floor_only"
    elif method == "DISCORD":
        kernel_slug = int(round(float(bridge_kernel)))
        q_slug = int(round(float(trust_quantile) * 100.0))
        fast_slug = "gf1" if fast_region_mode else "gf0"
        return f"entropy_final_L{safe(entropy_layer_str)}_rb1_bk{kernel_slug}_q{q_slug}_{fast_slug}"
    return f"legacy_{safe(method)}"


def _make_glb(run_dir, target_dir, predictions, conf_thres, frame_filter,
              mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode):
    """Render GLB into run_dir; filename encodes only visualization settings.

    target_dir is the upload root (contains images/ subfolder) — only needed
    when mask_sky=True so visual_util can find the original image files.
    """
    safe = lambda s: str(s).replace(".", "_").replace(":", "").replace(" ", "_")
    glbfile = os.path.join(
        run_dir,
        f"glb_{conf_thres}_{safe(frame_filter)}_b{mask_black_bg}_w{mask_white_bg}"
        f"_cam{show_cam}_sky{mask_sky}_{safe(prediction_mode)}.glb",
    )
    if not os.path.exists(glbfile):
        scene = predictions_to_glb(
            predictions, conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg, mask_white_bg=mask_white_bg,
            show_cam=show_cam, mask_sky=mask_sky,
            target_dir=target_dir, prediction_mode=prediction_mode,
        )
        scene.export(file_obj=glbfile)
    return glbfile


def update_visualization(
    run_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg,
    show_cam, mask_sky, prediction_mode, is_example,
):
    if is_example == "True":
        return None, "No reconstruction yet."
    if not run_dir or run_dir == "None" or not os.path.isdir(run_dir):
        return None, "No reconstruction yet — click **Reconstruct** first."
    preds_path = os.path.join(run_dir, "predictions.npz")
    if not os.path.exists(preds_path):
        return None, "Run **Reconstruct** first."

    key_list = [
        "pose_enc", "depth", "depth_conf", "world_points", "world_points_conf",
        "images", "extrinsic", "intrinsic", "world_points_from_depth", "image_files",
        "final_mask", "entropy_layer",
    ]
    loaded      = np.load(preds_path, allow_pickle=True)
    predictions = {k: np.array(loaded[k]) for k in key_list if k in loaded}

    target_dir = os.path.dirname(run_dir)
    glbfile = _make_glb(run_dir, target_dir, predictions, conf_thres, frame_filter,
                        mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode)
    return glbfile, "Visualization updated."


def on_method_change(method):
    """Show/hide controls based on selected method."""
    is_robust = False
    is_cond = False
    is_entropy = method == "DISCORD"
    return (
        gr.update(visible=is_robust),   # rej_thresh
        gr.update(visible=is_robust),   # attn_a
        gr.update(visible=is_cond),     # cond_views
        gr.update(visible=is_cond),     # layer_str
        gr.update(visible=is_entropy),  # entropy_layer_str
        gr.update(visible=is_entropy),  # bridge_kernel
        gr.update(visible=is_entropy),  # trust_quantile
        gr.update(visible=is_entropy),  # fast_region_mode
        gr.update(visible=False),       # entropy_action
        gr.update(visible=False),       # entropy_thresh
        gr.update(visible=is_cond),     # proj_mode
        gr.update(visible=is_cond),     # beta
        gr.update(visible=is_cond),     # svd_rank
    )



# Gradio UI
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(
    theme=theme,
    css="""
    .log-text * {
        font-style: italic; font-size: 18px !important; font-weight: bold !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent !important;
        text-align: center !important;
    }
    """,
) as demo:

    is_example     = gr.Textbox(visible=False, value="None")
    target_dir_box = gr.Textbox(visible=False, value="None")
    run_dir_box    = gr.Textbox(visible=False, value="None")  # set per-run, encodes all settings
    snapshot_data_box = gr.Textbox(visible=False, value="")
    snapshot_pose_box = gr.Textbox(visible=False, value="")
    live_pose_box = gr.Textbox(visible=False, value="")
    camera_pose_timer = gr.Timer(value=0.75, active=True)

    gr.HTML("""
    <h1>DISCORD — Cross-View Geometric Disagreement for Robust Feed-Forward 3D Reconstruction</h1>
    <p>
      Upload a multi-view image collection, inspect a baseline VGGT reconstruction,
      and compare it against floor-only confidence filtering and the final DISCORD pipeline.
    </p>
    <p style="font-size: 0.85em; color: #666;">
      <b>DISCORD</b> = floor-only confidence preprocessing + entropy-guided region trust + bridge-aware filling
    </p>
    """)

    with gr.Row():
        # Left column: upload + controls
        with gr.Column(scale=2):
            input_video  = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images",
                                   interactive=True)

            gr.Markdown("### Curated Trial Bundles")
            with gr.Row():
                trial_bundle = gr.Dropdown(
                    choices=[p.name for p in _trial_bundle_dirs()],
                    value=None,
                    label="Prepared Eval Trial",
                    scale=5,
                )
                refresh_bundle_btn = gr.Button("Refresh", scale=1)
                load_bundle_btn = gr.Button("Load", scale=1)
            bundle_info = gr.Markdown(
                f"Prepared bundles live under `{TRIAL_BUNDLE_ROOT}`.",
            )

            image_gallery = gr.Gallery(
                label="Input frames",
                columns=4, height="280px", object_fit="contain", preview=True,
            )

            gr.Markdown("---")
            gr.Markdown("### Method")
            method = gr.Radio(
                ["Baseline VGGT", "Floor-only", "DISCORD"],
                value="DISCORD",
                label="Reconstruction Method",
            )

            # ── Legacy hidden controls kept only for callback compatibility ──
            rej_thresh = gr.Slider(
                0.0, 1.0, value=0.4, step=0.05,
                label="Rejection Threshold τ", visible=False,
            )
            attn_a = gr.Slider(
                0.0, 1.0, value=0.5, step=0.05,
                label="Attention Weight α", visible=False,
            )

            # Legacy hidden controls kept only for callback compatibility 
            cond_views = gr.CheckboxGroup(
                choices=[], value=[],
                label="Conditioning Views (C) — select target appearance",
                visible=False,
            )
            with gr.Row(visible=False) as cond_row1:
                proj_mode = gr.Radio(
                    ["AdaIN", "SVD"],
                    value="AdaIN",
                    label="Projection Mode",
                )
                layer_str = gr.Textbox(
                    value="20-23",
                    label="Layers (e.g. 23, 20-23, 12,16,20,23)",
                )
            with gr.Row(visible=False) as cond_row2:
                beta = gr.Slider(
                    0.0, 1.0, value=1.0, step=0.05,
                    label="Beta (0=no change, 1=full projection)",
                )
                svd_rank = gr.Number(
                    value=16, label="SVD Rank (-1=full)",
                    precision=0,
                )
            entropy_layer_str = gr.Textbox(value="8", label="Entropy Layer", visible=True)
            bridge_kernel = gr.Slider(
                3, 9, value=3, step=1,
                label="Bridge Erosion Kernel",
                visible=True,
            )
            trust_quantile = gr.Slider(
                0.5, 0.98, value=0.90, step=0.02,
                label="Region Strictness q",
                visible=True,
            )
            fast_region_mode = gr.Checkbox(
                label="Fast Confidence Clustering (approx.)",
                value=True,
                visible=True,
            )
            entropy_action = gr.Radio(
                ["Hard Filter", "Confidence Reweight"],
                value="Hard Filter",
                label="Entropy Action",
                visible=False,
            )
            entropy_thresh = gr.Slider(
                0.0, 1.0, value=0.4, step=0.05,
                label="Entropy Trust Threshold",
                visible=False,
            )

        # Right column: 3-D viewer
        with gr.Column(scale=4):
            log_output = gr.Markdown(
                "Upload images, select a method, then click **Reconstruct**.",
                elem_classes=["log-text"],
            )
            reconstruction_output = gr.Model3D(
                height=520, zoom_speed=0.5, pan_speed=0.5,
                elem_id="reconstruction_viewer",
            )
            entropy_gallery = gr.Gallery(
                label="Method Visualizations",
                columns=3, height="220px", object_fit="contain", preview=True,
            )
            camera_pose_status = gr.Markdown("Current camera pose: unavailable.")
            snapshot_status = gr.Markdown("No camera snapshot saved yet.")

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", variant="primary", scale=2)
                save_view_btn = gr.Button("Save Current View", variant="secondary", scale=1)
                clear_btn  = gr.ClearButton(
                    [input_video, input_images, reconstruction_output,
                     log_output, target_dir_box, image_gallery, entropy_gallery, camera_pose_status, snapshot_status],
                    scale=1,
                )

            gr.Markdown("### Visualization")
            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Pointmap Branch"],
                    value="Pointmap Branch",
                    label="Prediction Mode",
                )
            with gr.Row():
                conf_thres   = gr.Slider(0, 100, value=50, step=0.1,
                                         label="Confidence Threshold (%)")
                frame_filter = gr.Dropdown(choices=["All"], value="All",
                                           label="Show Points from Frame")
                with gr.Column():
                    show_cam      = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky      = gr.Checkbox(label="Filter Sky",  value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background",
                                                value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background",
                                                value=False)

    # Method toggle → show/hide relevant controls
    method.change(
        fn=on_method_change,
        inputs=[method],
        outputs=[rej_thresh, attn_a, cond_views, layer_str, entropy_layer_str, bridge_kernel, trust_quantile, fast_region_mode, entropy_action, entropy_thresh, proj_mode, beta, svd_rank],
    )

    # Upload → populate gallery + conditioning view choices
    for upload_comp in [input_video, input_images]:
        upload_comp.change(
            fn=on_upload,
            inputs=[input_video, input_images],
            outputs=[reconstruction_output, target_dir_box, image_gallery,
                     entropy_gallery, log_output, cond_views],
        )

    refresh_bundle_btn.click(
        fn=refresh_trial_bundle_choices,
        inputs=[],
        outputs=[trial_bundle, bundle_info],
    )

    trial_bundle.change(
        fn=preview_trial_bundle,
        inputs=[trial_bundle],
        outputs=[image_gallery, bundle_info],
    )

    load_bundle_btn.click(
        fn=load_trial_bundle,
        inputs=[trial_bundle],
        outputs=[reconstruction_output, target_dir_box, image_gallery,
                 entropy_gallery, log_output, cond_views],
    ).then(fn=lambda: "False", inputs=[], outputs=[is_example])

    # Reconstruct button
    submit_btn.click(
        fn=lambda: (None, "Reconstructing…"),
        inputs=[], outputs=[reconstruction_output, log_output],
    ).then(
        fn=gradio_reconstruct,
        inputs=[
            target_dir_box, method, cond_views, layer_str, entropy_layer_str, bridge_kernel, trust_quantile, fast_region_mode, entropy_action, entropy_thresh, proj_mode, beta,
            svd_rank, rej_thresh, attn_a,
            conf_thres, frame_filter,
            mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode,
        ],
        outputs=[reconstruction_output, log_output, frame_filter, image_gallery, entropy_gallery, run_dir_box],
    ).then(fn=lambda: "False", inputs=[], outputs=[is_example])

    save_view_btn.click(
        fn=save_canvas_snapshot,
        inputs=[run_dir_box, snapshot_data_box, snapshot_pose_box],
        outputs=[snapshot_status],
        js=CAPTURE_VIEW_JS,
        queue=False,
        show_progress="hidden",
    )

    camera_pose_timer.tick(
        fn=format_live_camera_pose,
        inputs=[live_pose_box],
        outputs=[camera_pose_status, live_pose_box],
        js=LIVE_CAMERA_POSE_JS,
        queue=False,
        show_progress="hidden",
    )

    # Visualization controls → re-render using the current run_dir (settings already baked in)
    viz_inputs = [
        run_dir_box, conf_thres, frame_filter, mask_black_bg,
        mask_white_bg, show_cam, mask_sky, prediction_mode, is_example,
    ]
    for component in [conf_thres, frame_filter, mask_black_bg, mask_white_bg,
                      show_cam, mask_sky, prediction_mode]:
        component.change(
            fn=update_visualization,
            inputs=viz_inputs,
            outputs=[reconstruction_output, log_output],
        )

if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        show_error=True,
        share=False,
        allowed_paths=[str(TRIAL_BUNDLE_ROOT.resolve())],
    )
