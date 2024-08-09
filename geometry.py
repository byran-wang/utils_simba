import numpy as np
import math
import torch
from plyfile import PlyData, PlyElement

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


def save_point_cloud_to_ply(pts_3d, filepath):
    # Create a structured array for the points
    points_array = np.array([(point[0], point[1], point[2]) for point in pts_3d],
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    # Create a PlyElement object
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

    # Convert the data into a NumPy array
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T

    return points    