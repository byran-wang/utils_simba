import json
import pickle
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from tqdm import tqdm

import nvdiffrast.torch as dr

from .render import nvdiffrast_render


def ensure_cuda_available():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nvdiffrast visualization")


def load_pickle_compat(path: Path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as exc:
            if "numpy._core" not in str(exc):
                raise
            f.seek(0)

            class _NumpyCompatUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module.startswith("numpy._core"):
                        module = module.replace("numpy._core", "numpy.core", 1)
                    return super().find_class(module, name)

            return _NumpyCompatUnpickler(f).load()


def normalize_intrinsics(raw_k):
    k = np.asarray(raw_k, dtype=np.float32)
    if k.shape == (3, 3):
        return k
    if k.shape == (1, 3, 3):
        return k[0]
    if k.shape == (9,):
        return k.reshape(3, 3)
    if k.shape == (4,):
        fx, fy, cx, cy = k.tolist()
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    raise ValueError(f"Unsupported intrinsics shape: {k.shape}")


def load_mesh_as_trimesh(mesh_path: Path):
    loaded = trimesh.load(str(mesh_path), process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph[node_name]
            geom = loaded.geometry[geom_name].copy()
            geom.apply_transform(transform)
            meshes.append(geom)
        if len(meshes) == 0:
            return None
        return trimesh.util.concatenate(meshes)
    return None


def find_pose_dir(result_folder: Path, pose_dir_name: str = "ob_in_cam"):
    candidate = result_folder / pose_dir_name
    if candidate.is_dir():
        return candidate
    if result_folder.name == pose_dir_name and result_folder.is_dir():
        return result_folder
    raise FileNotFoundError(
        f"Could not find pose folder '{pose_dir_name}' under: {result_folder}"
    )


def load_pose_map(result_folder: Path, pose_dir_name: str = "ob_in_cam"):
    pose_dir = find_pose_dir(result_folder, pose_dir_name=pose_dir_name)
    pose_map = {}
    for txt in sorted(pose_dir.glob("*.txt")):
        pose_map[txt.stem] = np.loadtxt(txt).reshape(4, 4).astype(np.float32)
    if len(pose_map) == 0:
        raise RuntimeError(f"No pose txt files found in {pose_dir}")
    return pose_map, pose_dir


def find_rgb_meta_dirs(data_dir: Path):
    candidates = [
        (data_dir / "pipeline_preprocess" / "rgb", data_dir / "pipeline_preprocess" / "meta"),
        (data_dir / "rgb", data_dir / "meta"),
        (data_dir.parent.parent / "rgb", data_dir.parent.parent / "meta"),
    ]
    for rgb_dir, meta_dir in candidates:
        if rgb_dir.exists() and meta_dir.exists():
            return rgb_dir, meta_dir
    raise FileNotFoundError(f"Could not locate rgb/meta dirs from {data_dir}")


def find_image_for_id(rgb_dir: Path, frame_id: str):
    for ext in (".png", ".jpg", ".jpeg", ".JPG", ".PNG"):
        image_path = rgb_dir / f"{frame_id}{ext}"
        if image_path.exists():
            return image_path
    return None


def load_gt_valid_frame_ids(data_dir: Path, frame_ids_int):
    import vggt.utils.gt as gt

    seq_name = data_dir.name

    def get_image_fids():
        return frame_ids_int

    data_gt = gt.load_data(seq_name, get_image_fids)
    gt_is_valid = data_gt["is_valid"]
    if torch.is_tensor(gt_is_valid):
        gt_is_valid = gt_is_valid.detach().cpu().numpy()
    gt_is_valid = np.asarray(gt_is_valid).astype(bool)
    if len(gt_is_valid) != len(frame_ids_int):
        raise RuntimeError(
            f"GT validity length mismatch: {len(gt_is_valid)} vs {len(frame_ids_int)}"
        )
    return {fid for fid, ok in zip(frame_ids_int, gt_is_valid) if bool(ok)}


def load_pose_rgb_meta_frames(
    data_dir: Path,
    result_folder: Path,
    *,
    pose_dir_name: str = "ob_in_cam",
    vis_gt: bool = False,
):
    pose_map, pose_dir = load_pose_map(result_folder, pose_dir_name=pose_dir_name)
    rgb_dir, meta_dir = find_rgb_meta_dirs(data_dir)

    gt_valid_frame_ids_int = None
    if vis_gt:
        frame_ids_int = []
        for frame_id in sorted(pose_map.keys()):
            try:
                frame_ids_int.append(int(frame_id))
            except ValueError:
                continue
        if len(frame_ids_int) == 0:
            raise RuntimeError("Pose ids are not numeric; cannot align with GT validity")
        gt_valid_frame_ids_int = load_gt_valid_frame_ids(data_dir, sorted(set(frame_ids_int)))

    frames = []
    for frame_id in sorted(pose_map.keys()):
        frame_idx = None
        try:
            frame_idx = int(frame_id)
        except ValueError:
            pass

        if gt_valid_frame_ids_int is not None and frame_idx not in gt_valid_frame_ids_int:
            print(f"[skip] frame {frame_id}: invalid for evaluation according to GT")
            continue

        image_path = find_image_for_id(rgb_dir, frame_id)
        meta_path = meta_dir / f"{frame_id}.pkl"
        if image_path is None or not meta_path.exists():
            continue

        meta = load_pickle_compat(meta_path)
        if isinstance(meta, dict):
            if "camMat" in meta:
                raw_k = meta["camMat"]
            elif "intrinsics" in meta:
                raw_k = meta["intrinsics"]
            else:
                continue
        else:
            raw_k = meta

        frame = {
            "frame_id": frame_id,
            "image": np.array(Image.open(image_path).convert("RGB")),
            "K": normalize_intrinsics(raw_k),
            "pose_o2c": pose_map[frame_id],
        }
        if frame_idx is not None:
            frame["frame_idx"] = frame_idx
        frames.append(frame)

    if len(frames) == 0:
        if vis_gt:
            raise RuntimeError("No frames remain after GT-valid filtering")
        raise RuntimeError("No overlapping frames between poses and rgb/meta data")

    return {
        "frames": frames,
        "pose_map": pose_map,
        "pose_dir": pose_dir,
        "rgb_dir": rgb_dir,
        "meta_dir": meta_dir,
        "gt_valid_frame_ids_int": gt_valid_frame_ids_int,
    }


def overlay_normal(raw_img, normal_tensor, depth_tensor, alpha):
    normal = normal_tensor[0].detach().cpu().numpy()
    depth = depth_tensor[0].detach().cpu().numpy()
    mask = depth > 1e-6

    normal_vis = 1.0 - ((normal + 1.0) * 0.5).clip(0.0, 1.0)
    normal_vis = (normal_vis * 255.0).astype(np.uint8)

    out = raw_img.astype(np.float32).copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * normal_vis[mask]
    return out.clip(0, 255).astype(np.uint8), normal_vis


def create_video(frame_dir: Path, output_video: Path, fps: int):
    cmd = [
        "/usr/bin/ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "%06d.png"),
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(output_video),
    ]
    print(f"Running command:\n{shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


def prepare_output_dirs(
    out_dir: Path,
    *,
    rebuild: bool,
    overlay_dir_name: str,
    normal_dir_name: str,
):
    if rebuild and out_dir.exists():
        shutil.rmtree(out_dir)
    overlay_dir = out_dir / overlay_dir_name
    normal_dir = out_dir / normal_dir_name
    overlay_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    return overlay_dir, normal_dir


def _record_frame_value(record, key, value):
    if value is None:
        return
    if isinstance(value, np.generic):
        value = value.item()
    record[key] = value


def render_frames_with_nvdiffrast(
    *,
    frames,
    out_dir: Path,
    alpha: float,
    fps: int,
    rebuild: bool,
    default_mesh_tensors=None,
    desc: str = "Rendering normals with nvdiffrast",
    overlay_dir_name: str = "nvdiffrast_overlay_frames",
    normal_dir_name: str = "nvdiffrast_normal_frames",
    video_name: str = "nvdiffrast_overlay.mp4",
):
    ensure_cuda_available()
    overlay_dir, normal_dir = prepare_output_dirs(
        out_dir,
        rebuild=rebuild,
        overlay_dir_name=overlay_dir_name,
        normal_dir_name=normal_dir_name,
    )

    glctx = dr.RasterizeCudaContext()
    records = []

    for render_index, frame in enumerate(tqdm(frames, desc=desc)):
        image = frame["image"]
        h, w = image.shape[:2]
        pose = torch.as_tensor(frame["pose_o2c"][None], dtype=torch.float32, device="cuda")
        mesh_tensors = frame.get("mesh_tensors", default_mesh_tensors)
        if mesh_tensors is None:
            raise RuntimeError("No mesh tensors provided for rendering")

        _, depth, normal = nvdiffrast_render(
            K=normalize_intrinsics(frame["K"]),
            H=h,
            W=w,
            ob_in_cvcams=pose,
            glctx=glctx,
            context="cuda",
            get_normal=True,
            mesh_tensors=mesh_tensors,
            output_size=(h, w),
            use_light=False,
            extra={},
        )

        overlay_img, normal_img = overlay_normal(image, normal, depth, alpha=float(alpha))
        overlay_path = overlay_dir / f"{render_index:06d}.png"
        normal_path = normal_dir / f"{render_index:06d}.png"
        Image.fromarray(overlay_img).save(overlay_path)
        Image.fromarray(normal_img).save(normal_path)

        record = {
            "render_index": render_index,
            "overlay_path": overlay_path.name,
            "normal_path": normal_path.name,
        }
        _record_frame_value(record, "frame_idx", frame.get("frame_idx"))
        _record_frame_value(record, "frame_id", frame.get("frame_id"))
        record.update(frame.get("record", {}))
        records.append(record)

    if len(records) == 0:
        raise RuntimeError("No valid frames were rendered")

    with open(out_dir / "frame_map.json", "w") as f:
        json.dump(records, f, indent=2)

    output = {
        "records": records,
        "overlay_dir": overlay_dir,
        "normal_dir": normal_dir,
        "video_path": out_dir / video_name,
    }
    if fps > 0:
        create_video(overlay_dir, output["video_path"], fps=fps)
    return output
