from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-discord")

import cv2
import numpy as np
import trimesh

from discord3d.third_party import setup_vggt_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
setup_vggt_paths()

from visual_util import predictions_to_glb  # noqa: E402


def load_predictions(run_dir: Path) -> dict[str, np.ndarray]:
    preds_path = run_dir / "predictions.npz"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {preds_path}")
    loaded = np.load(preds_path, allow_pickle=True)
    return {k: np.array(loaded[k]) for k in loaded.files}


def load_camera_pose(camera_json: Path) -> dict:
    data = json.loads(camera_json.read_text())
    pose = data.get("pose", data)
    if not isinstance(pose, dict):
        raise ValueError(f"Invalid camera pose json: {camera_json}")
    return pose


def scene_to_world_geometry(scene: trimesh.Scene, include_meshes: bool = True):
    point_vertices = []
    point_colors = []
    meshes = []

    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry[geom_name]
        world_transform = np.asarray(transform, dtype=np.float32)

        if isinstance(geom, trimesh.points.PointCloud):
            verts = trimesh.transform_points(np.asarray(geom.vertices), world_transform)
            colors = np.asarray(geom.colors)[:, :3].astype(np.uint8)
            point_vertices.append(verts.astype(np.float32))
            point_colors.append(colors)
        elif include_meshes and isinstance(geom, trimesh.Trimesh):
            verts = trimesh.transform_points(np.asarray(geom.vertices), world_transform).astype(np.float32)
            faces = np.asarray(geom.faces).astype(np.int32)
            face_colors = np.asarray(geom.visual.face_colors)[:, :3].astype(np.uint8)
            meshes.append({
                "vertices": verts,
                "faces": faces,
                "face_colors": face_colors,
            })

    if point_vertices:
        points = np.concatenate(point_vertices, axis=0)
        colors = np.concatenate(point_colors, axis=0)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.uint8)
    return points, colors, meshes


def make_camera_frame(position: np.ndarray, target: np.ndarray, up_hint: np.ndarray | None = None):
    if up_hint is None:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    forward = target - position
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError("Camera position and target are identical.")
    forward = forward / forward_norm

    right = np.cross(forward, up_hint)
    if np.linalg.norm(right) < 1e-8:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    return right.astype(np.float32), up.astype(np.float32), forward.astype(np.float32)


def project_points(points: np.ndarray, position: np.ndarray, target: np.ndarray, width: int, height: int, fov_deg: float):
    right, up, forward = make_camera_frame(position, target)
    rel = points - position[None, :]
    cam_x = rel @ right
    cam_y = rel @ up
    cam_z = rel @ forward

    aspect = float(width) / float(height)
    f = 1.0 / np.tan(np.deg2rad(fov_deg) * 0.5)

    eps = 1e-6
    x_ndc = (cam_x / np.clip(cam_z, eps, None)) * (f / aspect)
    y_ndc = (cam_y / np.clip(cam_z, eps, None)) * f

    x_pix = (x_ndc + 1.0) * 0.5 * width
    y_pix = (1.0 - y_ndc) * 0.5 * height
    return x_pix, y_pix, cam_z


def render_point_cloud(
    image: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    position: np.ndarray,
    target: np.ndarray,
    fov_deg: float,
    point_size_scale: float = 1.0,
    max_points: int = 300_000,
):
    if len(points) == 0:
        return image

    if len(points) > max_points:
        stride = max(1, len(points) // max_points)
        points = points[::stride]
        colors = colors[::stride]

    h, w = image.shape[:2]
    xs, ys, z = project_points(points, position, target, w, h, fov_deg)
    valid = (z > 1e-4) & (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not np.any(valid):
        return image

    xs = xs[valid]
    ys = ys[valid]
    z = z[valid]
    cols = colors[valid]

    order = np.argsort(z)[::-1]  # far to near
    xs = xs[order]
    ys = ys[order]
    z = z[order]
    cols = cols[order]

    median_z = float(np.median(z)) if len(z) > 0 else 1.0
    for x, y, depth, color in zip(xs, ys, z, cols):
        radius = int(np.clip(point_size_scale * (median_z / max(float(depth), 1e-4)) * 1.3, 1, 4))
        cv2.circle(
            image,
            (int(round(x)), int(round(y))),
            radius,
            tuple(int(c) for c in color.tolist()),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return image


def render_meshes(
    image: np.ndarray,
    meshes: list[dict],
    position: np.ndarray,
    target: np.ndarray,
    fov_deg: float,
):
    if not meshes:
        return image

    h, w = image.shape[:2]
    triangles = []
    for mesh in meshes:
        verts = mesh["vertices"]
        faces = mesh["faces"]
        face_colors = mesh["face_colors"]
        xs, ys, z = project_points(verts, position, target, w, h, fov_deg)
        for face_idx, face in enumerate(faces):
            face_z = z[face]
            if np.any(face_z <= 1e-4):
                continue
            pts = np.stack([xs[face], ys[face]], axis=1).astype(np.int32)
            color = face_colors[min(face_idx, len(face_colors) - 1)]
            triangles.append((float(face_z.mean()), pts, color))

    triangles.sort(key=lambda item: item[0], reverse=True)
    for _, pts, color in triangles:
        cv2.fillConvexPoly(image, pts, color=tuple(int(c) for c in color.tolist()), lineType=cv2.LINE_AA)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return image


def render_scene(
    scene: trimesh.Scene,
    pose: dict,
    width: int,
    height: int,
    show_cam: bool,
    background: str,
    point_size_scale: float,
):
    points, colors, meshes = scene_to_world_geometry(scene, include_meshes=show_cam)

    bg_color = (255, 255, 255) if background == "white" else (0, 0, 0)
    image = np.full((height, width, 3), bg_color, dtype=np.uint8)

    position = np.array([
        pose["position"]["x"],
        pose["position"]["y"],
        pose["position"]["z"],
    ], dtype=np.float32)
    target = np.array([
        pose["target"]["x"],
        pose["target"]["y"],
        pose["target"]["z"],
    ], dtype=np.float32)
    fov_deg = float(pose.get("fov_deg", 45.0))

    image = render_point_cloud(
        image,
        points,
        colors,
        position=position,
        target=target,
        fov_deg=fov_deg,
        point_size_scale=point_size_scale,
    )
    if show_cam:
        image = render_meshes(
            image,
            meshes,
            position=position,
            target=target,
            fov_deg=fov_deg,
        )
    return image


def parse_args():
    parser = argparse.ArgumentParser(description="Render a static 3D view from a saved Gradio camera snapshot.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing predictions.npz")
    parser.add_argument("--camera-json", type=Path, required=True, help="Saved camera snapshot json")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--width", type=int, default=None, help="Render width. Defaults to saved snapshot width.")
    parser.add_argument("--height", type=int, default=None, help="Render height. Defaults to saved snapshot height.")
    parser.add_argument("--conf-thres", type=float, default=50.0, help="Confidence percentile threshold")
    parser.add_argument("--frame-filter", type=str, default="All", help="Frame filter passed to predictions_to_glb")
    parser.add_argument("--prediction-mode", type=str, default="Depthmap and Camera Branch", choices=["Depthmap and Camera Branch", "Pointmap Branch"])
    parser.add_argument("--show-cam", action="store_true", help="Render input camera meshes")
    parser.add_argument("--mask-black-bg", action="store_true")
    parser.add_argument("--mask-white-bg", action="store_true")
    parser.add_argument("--mask-sky", action="store_true")
    parser.add_argument("--background", choices=["black", "white"], default="black")
    parser.add_argument("--point-size-scale", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = load_predictions(args.run_dir)
    pose = load_camera_pose(args.camera_json)

    if "position" not in pose or "target" not in pose:
        raise ValueError("Camera json must contain position and target.")

    width = args.width or int(pose.get("width") or 1280)
    height = args.height or int(pose.get("height") or 720)

    scene = predictions_to_glb(
        predictions,
        conf_thres=float(args.conf_thres),
        filter_by_frames=args.frame_filter,
        mask_black_bg=bool(args.mask_black_bg),
        mask_white_bg=bool(args.mask_white_bg),
        show_cam=bool(args.show_cam),
        mask_sky=bool(args.mask_sky),
        target_dir=str(args.run_dir.parent),
        prediction_mode=args.prediction_mode,
    )

    image = render_scene(
        scene,
        pose=pose,
        width=width,
        height=height,
        show_cam=bool(args.show_cam),
        background=args.background,
        point_size_scale=float(args.point_size_scale),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved render to {args.output}")


if __name__ == "__main__":
    main()
