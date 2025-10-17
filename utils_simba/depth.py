import numpy as np
import cv2
import torch

def save_depth(depth, fname, scale= 0.00012498664727900177):
    depth_scale = 1 / scale
    max_depth = 2**24 / (1/ scale)     # max 8.19112491607666 = 2**16 / (1/ 0.00012498664727900177)
    exceed_depth = depth[depth > max_depth]
    if len(exceed_depth) > 0:
        print(f"Warning: {len(exceed_depth)} depth values exceed the maximum depth of {max_depth}. Clipping to {max_depth}.")
    depth = depth.clip(0, max_depth)

    depth_scaled = (depth * depth_scale).astype(np.uint32)
    depth_lsb = np.bitwise_and(depth_scaled, 0xFF)  # Least significant byte
    depth_msb = np.bitwise_and(np.right_shift(depth_scaled, 8), 0xFF)  # Most significant byte
    depth_msb2 = np.bitwise_and(np.right_shift(depth_scaled, 16), 0xFF)  # Most significant byte
    depth_encoded = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    depth_encoded[..., 2] = depth_lsb
    depth_encoded[..., 1] = depth_msb
    depth_encoded[..., 0] = depth_msb2
    cv2.imwrite(fname, depth_encoded)
    # print(f"Saved depth to {fname}")

def get_depth(depth_file, zfar=np.inf, depth_scale = 0.00012498664727900177):
    # depth = cv2.imread(self.color_files[i].replace('.jpg','.png').replace('rgb','depth'), -1)
    depth = cv2.imread(depth_file, -1)
    depth = (depth[...,0]*256.0*256.0 + depth[...,1]*256.0 + depth[...,2])*depth_scale
    depth[(depth<0.01) | (depth>=zfar)] = 0
    return depth


def depth2xyzmap_cuda(depth, K, uvs=None):
    H, W = depth.shape[-2:]  # assume (H, W) or (1, H, W)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    device = depth.device
    if uvs is None:
        us, vs = torch.meshgrid(
            torch.arange(W, device=device),
            torch.arange(H, device=device),
            indexing='xy'
        )
        us = us.flatten()
        vs = vs.flatten()
    else:
        uvs = uvs.round().long()
        us = uvs[:, 0]
        vs = uvs[:, 1]

    zs = depth[vs, us]
    valid = zs >= 0.01
    xs = (us[valid].float() - cx) * zs[valid] / fx
    ys = (vs[valid].float() - cy) * zs[valid] / fy
    pts = torch.stack((xs, ys, zs[valid]), dim=1)

    xyz_map = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    xyz_map[vs[valid], us[valid]] = pts
    return xyz_map

def depth2xyzmap(depth, K, uvs=None):
    invalid_mask = (depth<0.01)
    H,W = depth.shape[:2]
    if uvs is None:
        vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:,0]
        vs = uvs[:,1]
    zs = depth[vs,us]
    xs = (us-K[0,2])*zs/K[0,0]
    ys = (vs-K[1,2])*zs/K[1,1]
    pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
    xyz_map = np.zeros((H,W,3), dtype=np.float32)
    xyz_map[vs,us] = pts
    xyz_map[invalid_mask] = 0
    return xyz_map

def xyz2depthmap(xyz, K, image_size):
    """
    Convert 3D points to a depth map.

    Parameters:
    - xyz: (N, 3) array of 3D points in camera coordinates.
    - K: (3, 3) camera intrinsic matrix.
    - image_size: Tuple (H, W) specifying the height and width of the depth map.

    Returns:
    - depth_map: (H, W) array representing the depth map.
    """
    H, W = image_size
    depth_map = np.zeros((H, W), dtype=np.float32)

    # Extract intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Extract X, Y, Z
    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]

    # Filter out points with non-positive depth
    valid = Z > 0
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    # Project to pixel coordinates
    u = (fx * X / Z + cx).round().astype(np.int32)
    v = (fy * Y / Z + cy).round().astype(np.int32)

    # Filter points that fall inside the image boundaries
    valid_idx = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid_idx]
    v = v[valid_idx]
    Z = Z[valid_idx]

    # Iterate through points and populate the depth map
    for idx in range(len(Z)):
        pixel_u = u[idx]
        pixel_v = v[idx]
        depth = Z[idx]

        # If the pixel is empty or the current depth is closer, update the depth map
        if depth_map[pixel_v, pixel_u] == 0 or depth < depth_map[pixel_v, pixel_u]:
            depth_map[pixel_v, pixel_u] = depth

    return depth_map