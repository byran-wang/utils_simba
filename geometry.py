import numpy as np
import math
import torch
from plyfile import PlyData, PlyElement
import trimesh
from PIL import Image
import open3d as o3d

def fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    """
    Converts a depth map to 3D points in camera coordinates.

    Args:
        depth (torch.Tensor): Depth map with shape (N, P).
        pixel_grid (torch.Tensor): Pixel grid coordinates with shape (N, P, 2).
        focal (torch.Tensor): Focal length with shape (N, 1).
        pp (torch.Tensor): Principal point with shape (N, 2).

    Returns:
        torch.Tensor: 3D points in camera coordinates with shape (N, P, 3).
    """
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)
    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)

def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid

def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res

def depth_to_pts3d(focals, principal_points, c2w4x4, depth):
    """
    Convert depth map to 3D points in world frame.

    Args:
        focals (tensor): Focal lengths of the camera (N, 1).
        principal_points (tensor): Principal points of the camera (N, 2) with (cx, cy).
        c2w4x4 (tensor): Camera-to-world transformation matrix (N, 4, 4).
        depth (tensor): Depth map (N, H, W).

    Returns:
        numpy.ndarray: 3D points in world frame with shape of (N, H, W, 3).
    """

    # Get grid
    grid = xy_grid(depth.shape[-1], depth.shape[-2], device=depth.device, unsqueeze=0, cat_dim=-1)
    grid = grid.view(grid.shape[0], -1, 2)
    depth = depth.view(grid.shape[0], -1)
    # Get pointmaps in camera frame
    ptmaps_in_c = fast_depthmap_to_pts3d(depth, grid, focals, pp=principal_points) # (N, P, 3)
    # Project to world frame
    return geotrf(c2w4x4, ptmaps_in_c) # (N, P, 3)


def save_point_cloud_to_ply(pts_3d, filepath, colors=None):
    # Create a structured array for the points


    if colors is not None:
        colors = colors.astype(np.uint8)
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

        points_array = np.array(
            [(*point, *color) for point, color in zip(pts_3d, colors)],
            dtype=dtype
        )                

    else:
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        points_array = np.array([(point[0], point[1], point[2]) for point in pts_3d],
                            dtype=dtype)

    ply_element = PlyElement.describe(points_array, 'vertex')
    # Write to a PLY file
    PlyData([ply_element]).write(filepath)
    print(f"Point cloud saved to {filepath}")

# Function to read PLY file and extract point clouds
def read_point_cloud_from_ply(filepath):
    # Read the PLY file
    ply_data = PlyData.read(filepath)

    # Extract the vertex data (points) from the PLY file
    vertex_data = ply_data['vertex']
    try:
        red = np.array(vertex_data['red'])
        green = np.array(vertex_data['green'])
        blue = np.array(vertex_data['blue'])
        colors = np.vstack((red, green, blue)).T
    except:
        colors = None

    # Convert the data into a NumPy array
    positions = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T

    return positions, colors    

# Function to read PLY file and extract point clouds
def read_point_cloud_from_obj(obj_file, texture_file):
    # Step 1: Load the OBJ model using trimesh
    mesh = trimesh.load(obj_file, force='mesh')

    # Step 2: Get vertex positions and face indices
    positions = mesh.vertices            # numpy array of shape (n_vertices, 3)
    faces = mesh.faces                   # numpy array of shape (n_faces, 3)

    # Step 3: Load the texture image using PIL
    texture = Image.open(texture_file).convert('RGB')  # Ensure it's in RGB format
    texture_data = np.asarray(texture)    # Convert image to numpy array
    texture_height, texture_width = texture.size

    # Step 4: Get UV coordinates from the mesh
    uv_coords = mesh.visual.uv            # numpy array of shape (n_vertices, 2)

    # Ensure UV coordinates are within [0, 1]
    uv_coords = np.clip(uv_coords, 0, 1)

    # Step 5: Map UV coordinates to texture pixel indices
    u_indices = (uv_coords[:, 0] * (texture_width - 1)).astype(np.int32)
    v_indices = ((1 - uv_coords[:, 1]) * (texture_height - 1)).astype(np.int32)  # Invert V-axis

    # Step 6: Get colors from the texture image
    colors = texture_data[v_indices, u_indices] / 255.0  # Normalize RGB values to [0, 1]

    return positions, colors    

def convert_point_cloud(pc_f, pc_normalized_f, transformation_matrix):
    pcd = o3d.io.read_point_cloud(pc_f)
    if not pcd.has_points():
        raise ValueError(f"The point cloud at {pc_f} has no points.")
    
    print(f"Loaded point cloud with {len(pcd.points)} points.")

    # Check if point cloud has colors
    has_colors = pcd.has_colors()
    if has_colors:
        print("Point cloud has color information.")
    else:
        print("Point cloud does NOT have color information.")

    # Convert point cloud to numpy array for processing
    points = np.asarray(pcd.points)
    points_normalized = points @ transformation_matrix[:3, :3].T + transformation_matrix[:3, 3]
    pcd.points = o3d.utility.Vector3dVector(points_normalized)

    # If colors exist, ensure they are preserved
    if has_colors:
        colors = np.asarray(pcd.colors)
        # Normalize colors if they are in [0, 255]
        if colors.max() > 1.0:
            colors = colors / 255.0
            print("Normalized color values to [0, 1].")
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("Preserved color information in the transformed point cloud.")    

    success = o3d.io.write_point_cloud(pc_normalized_f, pcd, write_ascii=True)
    if success:
        print(f"Transformed point cloud saved to {pc_normalized_f}")
    else:
        raise IOError(f"Failed to save the transformed point cloud to {pc_normalized_f}")    

def get_revert_tvec(tvec, quat_xyzw):
    from scipy.spatial.transform import Rotation as R
    rotation = R.from_quat(quat_xyzw)
    rot_matrix = rotation.as_matrix()
    inverse_rot_matrix = rot_matrix.T
    reverse_tvec = -inverse_rot_matrix @ tvec
    return reverse_tvec

def transform_points(pts_c1: np.ndarray, c1Toc2_all: np.ndarray) -> np.ndarray:
    """
    Transforms 3D points from coordinates1 to coordinates2.

    Parameters:
        pts_w (np.ndarray): Points in  coordinates1 with shape (batch_size, num_points, 3).
        w2o_all (np.ndarray): Transformation matrices from world to camera coordinates with shape (batch_size, 4, 4).

    Returns:
        np.ndarray: Transformed points in camera coordinates2 with shape (batch_size, num_points, 3).
    """
    # Convert to homogeneous coordinates
    ones = np.ones((pts_c1.shape[0], pts_c1.shape[1], 1), dtype=pts_c1.dtype)
    pts_w_homo = np.concatenate([pts_c1, ones], axis=2)  # Shape: (batch_size, num_points, 4)

    # Expand dimensions for matrix multiplication
    pts_w_homo_expanded = pts_w_homo[..., np.newaxis]  # Shape: (batch_size, num_points, 4, 1)

    # Perform batch matrix multiplication
    pts_c2_homo = np.matmul(c1Toc2_all[:, np.newaxis, :, :], pts_w_homo_expanded)  # Shape: (batch_size, num_points, 4, 1)

    # Squeeze the last dimension and extract Cartesian coordinates
    pts_c2 = pts_c2_homo.squeeze(-1)[..., :3]  # Shape: (batch_size, num_points, 3)

    return pts_c2

def save_mesh(vertices, faces, out_f):
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)    

    mesh.export(out_f)