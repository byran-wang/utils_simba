from pathlib import Path
from typing import List, Optional, Tuple

import rerun as rr
import numpy as np
import os
import cv2
import trimesh
import rerun.blueprint as rrb
from .vis import rotation_matrix_to_quaternion


# ---------------------------------------------------------------------------
# Standalone helpers (used by both Visualizer class and pipeline scripts)
# ---------------------------------------------------------------------------

def add_material(color: list) -> rr.Material:
    """Creates a ReRun material with the specified color."""
    return rr.Material(albedo_factor=color)


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals by averaging adjacent face normals (vectorized)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    vertex_normals = np.zeros_like(vertices, dtype=np.float64)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (vertex_normals / norms).astype(np.float32)


def load_mesh_as_trimesh(path_or_dir) -> Optional[trimesh.Trimesh]:
    """Load a mesh file (or first found in a directory) as a single Trimesh.

    If *path_or_dir* is a directory, tries ``scene.glb`` then ``mesh.obj``
    inside it. Supports ``trimesh.Scene`` (GLB) by concatenating geometries.
    """
    p = Path(path_or_dir)
    if p.is_dir():
        candidates = [p / "scene.glb", p / "mesh.obj"]
    else:
        candidates = [p]

    for candidate in candidates:
        if not candidate.exists():
            continue
        loaded = trimesh.load(str(candidate), process=False)
        if isinstance(loaded, trimesh.Trimesh):
            return loaded
        if isinstance(loaded, trimesh.Scene):
            meshes = []
            for node_name in loaded.graph.nodes_geometry:
                transform, geom_name = loaded.graph[node_name]
                geom = loaded.geometry[geom_name].copy()
                geom.apply_transform(transform)
                meshes.append(geom)
            if meshes:
                return trimesh.util.concatenate(meshes)
    return None


def get_vertex_colors(mesh: Optional[trimesh.Trimesh]) -> Optional[np.ndarray]:
    """Extract (N, 3) uint8 RGB vertex colors from a trimesh, or ``None``."""
    if mesh is None:
        return None
    n_verts = len(mesh.vertices)
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc.ndim == 2 and vc.shape[0] == n_verts and vc.shape[1] >= 3:
            return vc[:, :3].astype(np.uint8)
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "to_color") and callable(mesh.visual.to_color):
        vc = np.asarray(mesh.visual.to_color().vertex_colors)
        if vc.ndim == 2 and vc.shape[0] == n_verts and vc.shape[1] >= 3:
            return vc[:, :3].astype(np.uint8)
    return None


def compute_frustum_lines(
    K: np.ndarray,
    H: int,
    W: int,
    c2w: np.ndarray,
    depth: float = 0.05,
) -> List[np.ndarray]:
    """Compute camera frustum line segments in world space.

    Returns a list of (2, 3) arrays: 4 edges from the camera origin to the
    image-plane corners and 4 edges forming the rectangle.
    """
    K_inv = np.linalg.inv(K[:3, :3])
    corners_px = np.array(
        [[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], dtype=np.float64
    )
    corners_cam = (K_inv @ corners_px.T).T * depth
    R = c2w[:3, :3].astype(np.float64)
    t = c2w[:3, 3].astype(np.float64)
    origin_w = t.copy()
    corners_w = (R @ corners_cam.T).T + t

    segments: List[np.ndarray] = []
    for c in corners_w:
        segments.append(np.stack([origin_w, c]))
    for j in range(4):
        segments.append(np.stack([corners_w[j], corners_w[(j + 1) % 4]]))
    return segments


def stamp_frame_text(
    img: np.ndarray,
    text: str,
    font_size: int = 40,
    position: Tuple[int, int] = (10, 10),
    color: Tuple[int, int, int] = (255, 255, 0),
) -> np.ndarray:
    """Draw *text* on an RGB image and return the modified array."""
    from PIL import Image, ImageDraw, ImageFont

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except (IOError, OSError):
        font = ImageFont.load_default()
    draw.text(position, text, fill=color, font=font)
    return np.array(pil_img)


def backproject_depth_to_points(
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    image: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    max_points: int = 50000,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Back-project a depth map to a 3D point cloud in world space.

    Args:
        depth: (H, W) depth map (values <= 0 are ignored).
        K: (3, 3) camera intrinsics.
        c2w: (4, 4) camera-to-world transform.
        image: Optional (H, W, 3) RGB image for per-point colors.
        mask: Optional (H, W) binary mask; only masked pixels are kept.
        max_points: Sub-sample to at most this many points.

    Returns:
        ``(points_world, colors)`` where *colors* may be ``None``.
    """
    H, W = depth.shape[:2]
    K_inv = np.linalg.inv(K[:3, :3])
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    pixels = np.stack([uu, vv, np.ones_like(uu)], axis=-1).reshape(-1, 3).astype(np.float64)

    valid = depth.ravel() > 0
    if mask is not None:
        valid = valid & (mask.ravel() > 0)

    rays_cam = (K_inv @ pixels.T).T
    pts_cam = rays_cam * depth.ravel()[:, None]
    pts_cam = pts_cam[valid]

    c2w = np.asarray(c2w, dtype=np.float64)
    pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]

    colors = None
    if image is not None:
        colors = image.reshape(-1, 3)[valid]

    if len(pts_world) > max_points:
        sel = np.random.choice(len(pts_world), max_points, replace=False)
        pts_world = pts_world[sel]
        if colors is not None:
            colors = colors[sel]

    return pts_world.astype(np.float32), colors


def log_camera_frame(
    entity: str,
    K: np.ndarray,
    c2w: np.ndarray,
    image: Optional[np.ndarray] = None,
    image_plane_distance: float = 1.0,
    jpeg_quality: int = 85,
    static: bool = False,
) -> None:
    """Log a camera transform, pinhole model, and optional image to rerun.

    Combines the three-call pattern: ``rr.Transform3D`` + ``rr.Pinhole`` + ``rr.Image``.
    """
    c2w = np.asarray(c2w, dtype=np.float32)
    rr.log(
        entity,
        rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]),
        static=static,
    )
    if image is not None and K is not None:
        H, W = image.shape[:2]
        K = np.asarray(K, dtype=np.float64)
        rr.log(
            entity,
            rr.Pinhole(
                resolution=[W, H],
                focal_length=[float(K[0, 0]), float(K[1, 1])],
                principal_point=[float(K[0, 2]), float(K[1, 2])],
                image_plane_distance=image_plane_distance,
            ),
            static=static,
        )
        rr.log(entity, rr.Image(image).compress(jpeg_quality=jpeg_quality), static=static)


def log_mesh(
    label: str,
    mesh_file: str,
    material: rr.Material = None,
    colors: np.ndarray = None,
    normals: np.ndarray = None,
    static: bool = False,
    faces_downsample_ratio: float = 1,
    compute_normals_flag: bool = True,
) -> None:
    """Load a mesh file and log it to rerun."""
    assert os.path.exists(mesh_file), f"Mesh file {mesh_file} does not exist"
    mesh = trimesh.load(mesh_file, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces)

    if colors is None:
        colors = get_vertex_colors(mesh)
    if faces_downsample_ratio < 1:
        indices = np.random.choice(faces.shape[0], int(faces.shape[0] * faces_downsample_ratio), replace=False)
        faces = faces[indices]
    if normals is None and compute_normals_flag:
        normals = compute_vertex_normals(vertices, faces)

    mesh_kwargs = dict(
        vertex_positions=vertices,
        triangle_indices=faces.astype(np.int32),
    )
    if normals is not None:
        mesh_kwargs["vertex_normals"] = normals
    if colors is not None:
        mesh_kwargs["vertex_colors"] = colors.astype(np.uint8)
    elif material is not None:
        mesh_kwargs["mesh_material"] = material
    rr.log(label, rr.Mesh3D(**mesh_kwargs), static=static)


def vis_camera_image_pose_intrisic_in_rerun(all_images, focal, w, h, all_c2w, image_plane_distance=0.1):
    """Visualize camera intrinsics, poses and images in rerun."""
    rr.init("blender_dataset", spawn=True)

    for cam_idx in range(len(all_images)):
        K = np.array([
            [focal, 0, w / 2],
            [0, focal, h / 2],
            [0, 0, 1]
        ])
        c2w = all_c2w[cam_idx].cpu().numpy()
        img = (all_images[cam_idx].cpu().numpy() * 255).astype(np.uint8)
        log_camera_frame(
            f"world/camera_{cam_idx}", K, c2w, img,
            image_plane_distance=image_plane_distance,
        )


# ---------------------------------------------------------------------------
# Visualizer class (stateful, with world_transform support)
# ---------------------------------------------------------------------------

class Visualizer:
    def __init__(
        self,
        viewer_name: str = "trellis",
        jpeg_quality: int = 75,
        ImagePlaneDistance: float = 1.0,
        world_coordinate: str = "object",
    ) -> None:
        self._jpeg_quality = jpeg_quality
        rr.init(viewer_name, spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        self.world_coordinate = world_coordinate
        self.world_transform = np.eye(4)

    def log_mesh(self, label: str,
                 mesh_file: str,
                 material: rr.Material = None,
                 colors: np.ndarray = None,
                 normals: np.ndarray = None,
                 static=False,
                 faces_downsample_ratio: float = 1,
                 compute_normals_flag: bool = True,
                 ) -> None:
        assert os.path.exists(mesh_file), f"Mesh file {mesh_file} does not exist"
        mesh = trimesh.load(mesh_file, process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces)

        if colors is None:
            colors = get_vertex_colors(mesh)
        if faces_downsample_ratio < 1:
            indices = np.random.choice(faces.shape[0], int(faces.shape[0] * faces_downsample_ratio), replace=False)
            faces = faces[indices]
        vertices = (self.world_transform[:3, :3] @ vertices.T + self.world_transform[:3, 3:4]).T
        if normals is None and compute_normals_flag:
            normals = compute_vertex_normals(vertices, faces)

        mesh_kwargs = dict(
            vertex_positions=vertices,
            triangle_indices=faces.astype(np.int32),
        )
        if normals is not None:
            mesh_kwargs["vertex_normals"] = normals
        if colors is not None:
            mesh_kwargs["vertex_colors"] = colors.astype(np.uint8)
        elif material is not None:
            mesh_kwargs["mesh_material"] = material
        rr.log(label, rr.Mesh3D(**mesh_kwargs), static=static)

    def log_image(self, label: str,
                  image_file: str,
                  jpeg_quality: int = 75,
                  static=False,
                  ) -> None:
        assert os.path.exists(image_file), f"Image file {image_file} does not exist"
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rr.log(label, rr.Image(image).compress(jpeg_quality=jpeg_quality), static=static)

    def log_calibration(
        self,
        label: str,
        resolution: list[int],
        intrins: np.ndarray,
        image_plane_distance: float = 1.0,
        static=False,
    ) -> None:
        rr.log(
            label,
            rr.Pinhole(
                resolution=resolution,
                focal_length=[intrins[0, 0], intrins[1, 1]],
                principal_point=[intrins[0, 2], intrins[1, 2]],
                image_plane_distance=image_plane_distance,
            ),
            static=static,
        )

    def log_cam_pose(self, label: str,
                     c2w: np.ndarray,
                     axis_length: float = 0.1,
                     static=False,
                     ) -> None:
        c2w = self.world_transform @ c2w
        tvec = c2w[:3, 3]
        quat_xyzw = rotation_matrix_to_quaternion(c2w[:3, :3])
        rr.log(label, rr.Transform3D(translation=tvec, rotation=rr.Quaternion(xyzw=quat_xyzw), axis_length=axis_length, from_parent=False), static=static)

    def set_time_sequence(self, frame_index: int) -> None:
        rr.set_time_sequence("frame", frame_index)

    def log_points(self, label: str,
                   points: np.ndarray,
                   colors: np.ndarray = None,
                   sizes: np.ndarray = None,
                   radii: float = 0.001,
                   static=False,
                   ) -> None:
        points = (self.world_transform[:3, :3] @ points.T + self.world_transform[:3, 3:4]).T
        if colors is None:
            rr.log(label, rr.Points3D(positions=points, radii=radii), static=static)
        elif sizes is None:
            rr.log(label, rr.Points3D(positions=points, colors=colors, radii=radii), static=static)
        else:
            rr.log(label, rr.Points3D(positions=points, colors=colors, radii=sizes), static=static)

    def log_axis(self, label: str = "axis", scale: float = 1.0):
        origins = np.zeros((3, 3))
        ends = np.eye(3) * scale
        colors = np.eye(3, 4)
        colors[:, -1] = 1
        rr.log(f"{label}", rr.Arrows3D(origins=origins, vectors=ends, colors=colors), timeless=True)

    def log_3d_asset(self, label: str, mesh_path: str, static: bool = False):
        rr.log(label, rr.Asset3D(path=mesh_path), static=static)
