import argparse
import logging
from pathlib import Path

import numpy as np
import trimesh
import open3d as o3d
import os

import sys
sys.path.append("./third_party/utils_simba")
from utils_simba.visibility_test import mesh_to_voxel_grid, voxel_centers, determine_visibility
from utils_simba.visibility_test import create_colored_mesh_from_voxels

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_file", type=str, help="mesh file")
    parser.add_argument("--c2w", type=str, help="camera poses from camera to world")
    parser.add_argument("--out_dir", type=str, help="output directory")
    parser.add_argument("--mute", action="store_true")


    args = parser.parse_args()

    return args


def get_visibility_mesh(mesh, c2w_np_type, out_dir, idx):

    voxel_size = 0.02  # Adjust voxel size as needed
    num_views = 5  # Number of camera views
    max_distance = 1000  # Maximum distance for visibility checks
    # Load mesh if given a file path
    if isinstance(mesh, (str, os.PathLike)):
        mesh = trimesh.load(mesh)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()  # In case the mesh is a scene with multiple meshes
    # Voxelization

    voxels, voxel_origin, grid_size = mesh_to_voxel_grid(mesh, voxel_size)
    print(f"Voxel grid size: {grid_size}")

    camera_poses = c2w_np_type

    # Compute voxel centers
    voxel_centers_np, occupied_indices = voxel_centers(voxels, voxel_origin, voxel_size)
    print(f"Number of occupied voxels: {voxel_centers_np.shape[0]}")

    # Determine visibility
    try:
        # Try to use PyEmbree for faster ray tracing
        visibility = determine_visibility(mesh, voxel_centers_np, camera_poses, max_distance)
    except ImportError:
        print("PyEmbree not installed, falling back to slower ray tracing.")
        trimesh.ray.ray_pyembree.RayMeshIntersector = None
        visibility = determine_visibility(mesh, voxel_centers_np, camera_poses, max_distance)

    # # Example: Print visibility of first 10 voxels
    # for i in range(min(10, visibility.shape[0])):
    #     print(f"Voxel {i}: Visible from views {np.where(visibility[i])[0]}")
    
    # Create colored mesh from voxels
    mesh_with_visibility = create_colored_mesh_from_voxels(
        voxels, voxel_origin, voxel_size, visibility
    )
    # Optional: Save the mesh to a file
    o3d.io.write_triangle_mesh(f"{out_dir}/mesh_with_visibility_{idx}.ply", mesh_with_visibility)
    
    return visibility

def get_visibility_occ(mesh, c2o, debug_dir=None, voxel_size=0.01, render_size=512, depth_threshold=0.01):
    """Get visible occupancy voxels of a mesh from a given camera pose.

    Renders the mesh depth from c2o, voxelizes the mesh, projects each voxel
    into the camera, and compares voxel depth against rendered depth to
    determine visibility.

    Args:
        mesh: trimesh.Trimesh mesh in object space (or file path)
        c2o: (4,4) camera-to-object transform
        debug_dir: directory to save debug outputs (None to skip saving)
        voxel_size: size of each voxel
        render_size: resolution for rendering
        depth_threshold: tolerance for depth comparison

    Returns:
        visible_voxel_mesh: trimesh with visible voxels (red) and occluded (blue)
    """
    import torch
    from utils_simba.render import nvdiffrast_render

    if isinstance(mesh, (str, os.PathLike)):
        mesh = trimesh.load(mesh)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()

    if debug_dir is not None:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)

    # Fake intrinsics for rendering
    focal = render_size * 0.8
    K_fake = np.array([
        [focal, 0, render_size / 2],
        [0, focal, render_size / 2],
        [0, 0, 1],
    ], dtype=np.float64)

    # Render mesh depth from c2o pose (o2c = inv(c2o))
    o2c = np.linalg.inv(c2o)
    ob_in_cvcams = torch.tensor(o2c, dtype=torch.float32, device="cuda")[None]
    _, depth_render, _ = nvdiffrast_render(
        K=K_fake, H=render_size, W=render_size,
        ob_in_cvcams=ob_in_cvcams, mesh=mesh,
    )
    depth_map = depth_render[0].cpu().numpy()  # (H, W)

    # Convert mesh to occupancy grid
    voxels, voxel_origin, grid_size = mesh_to_voxel_grid(mesh, voxel_size)
    occupied = np.argwhere(voxels)  # (N, 3)
    voxel_centers_np = voxel_origin + (occupied + 0.5) * voxel_size  # (N, 3) in object space

    # Project each voxel center into the camera
    pts_h = np.hstack([voxel_centers_np, np.ones((len(voxel_centers_np), 1))])  # (N, 4)
    pts_cam = (o2c @ pts_h.T).T[:, :3]  # (N, 3) in camera space
    voxel_z = pts_cam[:, 2]  # depth of each voxel in camera

    # Project to pixel coordinates
    px = (K_fake[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K_fake[0, 2]).astype(int)
    py = (K_fake[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K_fake[1, 2]).astype(int)

    # Determine visibility: voxel is visible if its depth <= rendered depth + threshold
    in_bounds = (px >= 0) & (px < render_size) & (py >= 0) & (py < render_size) & (voxel_z > 0)
    visible = np.zeros(len(voxel_centers_np), dtype=bool)
    for i in np.where(in_bounds)[0]:
        rendered_depth = depth_map[py[i], px[i]]
        if rendered_depth > 0.01 and voxel_z[i] <= rendered_depth + depth_threshold:
            visible[i] = True

    logger.info(f"    Visibility: {visible.sum()}/{len(visible)} voxels visible")

    # Build a renderable mesh from all voxels (visible=red, occluded=blue)
    if len(voxel_centers_np) == 0:
        return trimesh.Trimesh()
    boxes = []
    for center, is_visible in zip(voxel_centers_np, visible):
        box = trimesh.creation.box(extents=[voxel_size] * 3)
        box.apply_translation(center)
        color = np.array([255, 0, 0, 255] if is_visible else [0, 0, 255, 255], dtype=np.uint8)
        box.visual.face_colors = np.tile(color, (len(box.faces), 1))
        boxes.append(box)
    visible_voxel_mesh = trimesh.util.concatenate(boxes)

    # Save the voxel mesh
    if debug_dir is not None:
        visible_voxel_mesh.export(str(debug_path / "visible_voxel_mesh.ply"))
    return visible_voxel_mesh


def combile_visibility_mesh(mesh_file, visibilities, out_dir):

    voxel_size = 0.02  # Adjust voxel size as needed
    # Load mesh
    mesh = trimesh.load(mesh_file)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()  # In case the mesh is a scene with multiple meshes
    # Voxelization

    
    voxels, voxel_origin, grid_size = mesh_to_voxel_grid(mesh, voxel_size)
    visibility = np.zeros_like((visibilities[0])).astype(bool)
    for vis in visibilities:
        visibility = np.logical_or(visibility, vis)

    # Create colored mesh from voxels
    mesh_with_visibility = create_colored_mesh_from_voxels(
        voxels, voxel_origin, voxel_size, visibility
    )
    # Optional: Save the mesh to a file
    o3d.io.write_triangle_mesh(f"{out_dir}/mesh_with_visibility.ply", mesh_with_visibility)

    return visibility

if __name__ == "__main__":
    args = parse_args()

    mesh_file = args.mesh_file
    c2w = args.c2w
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    get_visibility_mesh(mesh_file, c2w, out_dir)