from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .third_party import setup_vggt_paths


setup_vggt_paths()

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


PATCH_SIZE = 14
PATCH_START = 5
DTYPE = torch.bfloat16


def run_pass1_with_layers(imgs: torch.Tensor, model, device: str, feat_layers: list[int]):
    x = imgs.unsqueeze(0).to(device)
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=DTYPE):
        agg, patch_start_idx = model.aggregator(x)
        with torch.cuda.amp.autocast(enabled=False):
            pose_enc_list = model.camera_head(agg)
            depth, depth_conf = model.depth_head(agg, images=x, patch_start_idx=patch_start_idx)
            pts3d, pts3d_conf = model.point_head(agg, images=x, patch_start_idx=patch_start_idx)

    selected = {}
    for layer in feat_layers:
        feat = agg[layer][0, :, PATCH_START:, 1024:].detach().cpu().float()
        selected[layer] = F.normalize(feat, p=2, dim=-1)

    pred = {
        "pose_enc": pose_enc_list[-1],
        "depth": depth,
        "depth_conf": depth_conf,
        "world_points": pts3d,
        "world_points_conf": pts3d_conf,
        "images": x,
    }
    return pred, selected


def cameras_from_pred(pred, h: int, w: int) -> np.ndarray:
    pe = pred["pose_enc"]
    ext, _ = pose_encoding_to_extri_intri(pe.float(), (h, w))
    ext = ext[0].cpu().numpy()
    return np.array([-ext[i, :3, :3].T @ ext[i, :3, 3] for i in range(ext.shape[0])])


def read_colmap_poses(images_bin: str | Path) -> dict[str, np.ndarray]:
    images_bin = Path(images_bin)
    poses = {}
    with images_bin.open("rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            _image_id = struct.unpack("<i", f.read(4))[0]
            qvec = np.frombuffer(f.read(32), dtype=np.float64).copy()
            tvec = np.frombuffer(f.read(24), dtype=np.float64).copy()
            _camera_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = Path(name.decode("utf-8")).stem
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts * 24)

            qw, qx, qy, qz = qvec
            rot = np.array(
                [
                    [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                    [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
                    [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
                ],
                dtype=np.float64,
            )
            center = -rot.T @ tvec
            poses[name] = center.astype(np.float32)
    return poses

