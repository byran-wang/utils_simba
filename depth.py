import numpy as np
import cv2

def save_depth(depth, fname, scale= 0.00012498664727900177):
    depth_scale = 1 / scale
    depth_scaled = (depth * depth_scale).astype(np.uint16)
    depth_lsb = np.bitwise_and(depth_scaled, 0xFF)  # Least significant byte
    depth_msb = np.right_shift(depth_scaled, 8)  # Most significant byte
    depth_encoded = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    depth_encoded[..., 2] = depth_lsb
    depth_encoded[..., 1] = depth_msb
    cv2.imwrite(fname, depth_encoded)
    print(f"Saved depth to {fname}")

def get_depth(depth_file, zfar=np.inf, depth_scale = 0.00012498664727900177):
    # depth = cv2.imread(self.color_files[i].replace('.jpg','.png').replace('rgb','depth'), -1)
    depth = cv2.imread(depth_file, -1)
    depth = (depth[...,2]+depth[...,1]*256)*depth_scale
    depth[(depth<0.1) | (depth>=zfar)] = 0
    return depth

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