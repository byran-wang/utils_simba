from PIL import Image
import numpy as np
import json
import os
from pathlib import Path
from typing import NamedTuple
from rich.console import Console
import math
from glob import glob
import pickle
import cv2
import os.path as op
import shutil
import sys
CONSOLE = Console(width=120)

class CameraInfo(NamedTuple):
    uid: int
    c2w4x4: np.array
    fl_x: np.array
    fl_y: np.array
    height: int
    width: int
    fov_x: float
    fov_y: float
    cx: float
    cy: float
    rgb_name: str
    rgba_name: str
    mask_name: str
    pts3d: np.array

def preprocessCamerasFromBlenderJson(config):  
    path = config.json_path
    transformsfile = config.json_file

    images_dir = os.path.join(path, "rgbs")
    masks_dir = os.path.join(path, "masks")

    os.makedirs(path, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)        
    with open(os.path.join(path, transformsfile)) as json_file:
        meta = json.load(json_file)
        for idx, frame in enumerate(meta["frames"]):
            rgba_f = os.path.join(path, frame["file_path"])
            if not rgba_f:
                num_skipped_image_filenames += 1
            else:
                rgba_image = cv2.imread(rgba_f, cv2.IMREAD_UNCHANGED)
                rgb_image = rgba_image[:, :, :3]
                mask_image = rgba_image[:, :, 3]
                _, binary_mask = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)
                rgba_base_name = os.path.basename(rgba_f)
                rgb_f = os.path.join(images_dir, rgba_base_name)
                mask_f = os.path.join(masks_dir, rgba_base_name)
                cv2.imwrite(rgb_f, rgb_image)
                cv2.imwrite(mask_f, binary_mask)
                print(f"Writing {rgb_f} and {mask_f}")



def readCamerasFromBlenderJson(config):
    # Read the camera information from the transforms.json file
    cam_type = config.cam_type.lower() # "cvc2cvw" or "cvc2blw" or "blc2blw"
    path = config.json_path
    transformsfile = config.json_file
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        meta = json.load(json_file)
        num_skipped_image_filenames = 0
        fl_x, fl_y = get_focal_lengths(meta)
        width = meta["w"]
        height = meta["h"]
        fov_x = math.atan(width / (2 * fl_y)) * 2
        fov_y = math.atan(height / (2 * fl_y)) * 2
        cx, cy = (width -1) * 0.5, (height -1) * 0.5
        cvc2blc = np.array([[1, 0, 0, 0],[0, -1 , 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        blw2cvw = np.array([[0, 1, 0, 0],[0, 0 , -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        for idx, frame in enumerate(meta["frames"]):
            rgba_f = os.path.join(path, frame["file_path"])
            rgba_base_name = os.path.basename(rgba_f)
            rgb_f = os.path.join(path, "rgbs", rgba_base_name)
            mask_f = os.path.join(path, "masks", rgba_base_name)
            if not rgba_f:
                num_skipped_image_filenames += 1
            else:
                blc2blw = np.array(frame["transform_matrix"])
                # c2w_cv = gl_to_cv_t @ c2w_gl
                if cam_type == "cvc2cvw":
                    cvc2cvw = blw2cvw @ blc2blw @ cvc2blc
                    c2w_final = cvc2cvw
                if cam_type == "cvc2blw":
                    cvc2blw = blc2blw @ cvc2blc
                    c2w_final = cvc2blw
                elif cam_type == "blc2blw":
                    c2w_final = blc2blw
                cam_infos.append(CameraInfo(uid=idx, c2w4x4=c2w_final, fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy, 
                                            rgb_name=rgb_f, rgba_name=rgba_f, mask_name=mask_f, pts3d=None,
                                            height=height, width=width, fov_x=fov_x, fov_y=fov_y))
        if num_skipped_image_filenames >= 0:
            CONSOLE.print(f"Skipping {num_skipped_image_filenames} files in dataset {path}.")
        assert (
            len(cam_infos) != 0
        ), """
        No image files found. 
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """               
            
    return cam_infos

def get_focal_lengths(meta):
    """Reads or computes the focal length from transforms dict.
    Args:
        meta: metadata from transforms.json file.
    Returns:
        Focal lengths in the x and y directions. Error is raised if these cannot be calculated.
    """
    fl_x, fl_y = 0, 0

    def fov_to_focal_length(rad, res):
        return 0.5 * res / np.tan(0.5 * rad)

    if "fl_x" in meta:
        fl_x = meta["fl_x"]
    elif "x_fov" in meta:
        fl_x = fov_to_focal_length(np.deg2rad(meta["x_fov"]), meta["w"])
    elif "camera_angle_x" in meta:
        fl_x = fov_to_focal_length(meta["camera_angle_x"], meta["w"])

    if "fl_y" in meta:
        fl_y = meta["fl_y"]
    elif "y_fov" in meta:
        fl_y = fov_to_focal_length(np.deg2rad(meta["y_fov"]), meta["h"])
    elif "camera_angle_y" in meta:
        fl_y = fov_to_focal_length(meta["camera_angle_y"], meta["h"])

    if fl_x == 0 or fl_y == 0:
        raise AttributeError("Focal length cannot be calculated from transforms.json (missing fields).")

    return (fl_x, fl_y)


def preprocessCamerasFromRealImageWithGtPose(config):
    rgb_all = sorted(glob(f"{config['rgb_path']}/*.jpg"))
    pose_all = sorted(glob(f"{config['pose_path']}/*.pkl"))
    mask_all = sorted(glob(f"{config['mask_path']}/*.png"))
    out_dir = config['out_dir']
    images_dir = os.path.join(out_dir, "images")
    rgbas_dir = os.path.join(out_dir, "rgbas")
    masks_dir = os.path.join(out_dir, "masks")
    poses_dir = os.path.join(out_dir, "poses")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)
    os.makedirs(rgbas_dir, exist_ok=True)

    if config.end_frame == -1:
        end_fname = rgb_all[-1]
        end = int(op.basename(end_fname).split(".")[0].split("_rgba")[0])
        config.end_frame = end
    assert config.end_frame > config.start_frame    

    frame_range = list(range(config.start_frame, config.end_frame, config.frame_interval))
    valid_frames = []

    for i in frame_range:
        rgb_f = os.path.join(config['rgb_path'], f"{i:04d}.jpg")
        if (rgb_f in rgb_all):
            pose_f = os.path.join(config['pose_path'], f"{i:04d}.pkl")
            if pose_f in pose_all:
                mask_f = os.path.join(config['mask_path'], f"{i:05d}.png")
                if mask_f in mask_all:
                    if i not in config.exclude_frames:
                        meta = pickle.load(open(pose_f,'rb'))
                        if meta['objTrans'] is None or meta['objRot'] is None or meta['camMat'] is None:
                            print(f"Warning: Pose not found for Meta file {pose_f}, and skip frame {rgb_f}")
                            continue

                        valid_frames.append([rgb_f, pose_f, mask_f])
                else:
                    print(f"Warning: {mask_f} not found")
            else:
                print(f"Warning: {pose_f} not found")
        else:
            print(f"Warning: {rgb_f} not found")
    assert len(valid_frames) > 0

    for ci, [rgb_f, pose_f, mask_f] in enumerate(valid_frames):
        rgb_f_out = os.path.join(images_dir, f"{ci:04d}.png")
        mask_f_out = os.path.join(masks_dir, f"{ci:04d}.png")
        pose_f_out = os.path.join(poses_dir, f"{ci:04d}.pkl")
        cv2.imwrite(rgb_f_out, cv2.imread(rgb_f))
        shutil.copy(mask_f, mask_f_out)
        shutil.copy(pose_f, pose_f_out)

        rgb_imgae = cv2.imread(rgb_f)
        mask_image = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask[:, :, np.newaxis]
        masked_image = rgb_imgae * (binary_mask // 255) + 255 * (1 - binary_mask// 255)
        alpha_channel = binary_mask
        rgba_image = cv2.merge((masked_image[:, :, 0], masked_image[:, :, 1], masked_image[:, :, 2], alpha_channel))
        rgba_f = os.path.join(rgbas_dir, f"{ci:04d}.png")
        cv2.imwrite(rgba_f, rgba_image)

    
    json_file = os.path.join(out_dir, "map.json")
    print(f"Writing to {json_file}")
    with open(json_file, 'w') as f:
        for i, entry in enumerate(valid_frames):
            rgb_f, pose_f, mask_f = entry
            f.write(f"{i:04d} {rgb_f} {mask_f} {pose_f}\n")        
# after preprocessCamerasFromRealImageWithGtPose is called, the following code can be used to read the camera information
def readCamerasFromRealImageWithGTPose_1(config):
    cam_type = config.cam_type.lower() # "cvc2cvw" or "cvc2blw" or "blc2blw"
    cam_infos = []
    rgb_all = sorted(glob(f"{config['rgb_path']}/*.png"))
    pose_all = sorted(glob(f"{config['pose_path']}/*.pkl"))
    mask_all = sorted(glob(f"{config['mask_path']}/*.png"))
    rgba_all = sorted(glob(f"{config['rgba_path']}/*.png"))

    K = None
    cam_infos = [] 
    for ci, [rgb_f, rgba_f, mask_f, pose_f] in enumerate(zip(rgb_all, rgba_all, mask_all, pose_all)):
        meta = pickle.load(open(pose_f,'rb'))
        if meta['objTrans'] is None or meta['objRot'] is None or meta['camMat'] is None:
            print(f"Warning: Pose not found for Meta file {pose_f}, and skip frame {rgba_f}")
            assert False
        if ci == 0:
            K = meta['camMat']
            fl_x, fl_y = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            rgba_imgae = cv2.imread(rgba_f)
            height, width = rgba_imgae.shape[:2]
            fov_x = math.atan(width / (2 * fl_x)) * 2
            fov_y = math.atan(height / (2 * fl_y)) * 2
            cvw2blw = np.array([[0, 0, -1, 0],[1, 0 , 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            blc2glc = np.array([[1, 0, 0, 0],[0, 1 , 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            glc2cvc = np.array([[1, 0, 0, 0],[0, -1 , 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            blc2cvc = np.array([[1, 0, 0, 0],[0, -1 , 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        cvw2glc = np.eye(4)
        cvw2glc[:3,3] = meta['objTrans']
        cvw2glc[:3,:3] = cv2.Rodrigues(meta['objRot'].reshape(3))[0]
        cvw2cvc = glc2cvc @ cvw2glc
        cvc2cvw = np.linalg.inv(cvw2cvc)
        if cam_type == "blc2blw":
            blc2blw = cvw2blw @ cvc2cvw @ blc2cvc
            c2w_final = blc2blw
        if cam_type == "cvc2cvw":
            c2w_final = cvc2cvw
        cam_infos.append(CameraInfo(uid=ci, c2w4x4=c2w_final, fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy, 
                                    rgb_name=rgb_f, rgba_name=rgba_f, mask_name=mask_f, pts3d=None,
                                    height=height, width=width, fov_x=fov_x, fov_y=fov_y))
        
    return cam_infos

def RunPreprocessCamerasFromRealImageWithGtPose():
    scene = "AP10"
    config = {
        "rgb_path": "/home/simba/Documents/project/BundleSDF/dataset/HO3D_v3/evaluation/" + scene + "/rgb/",
        "pose_path": "/home/simba/Documents/project/BundleSDF/dataset/HO3D_v3/evaluation/" + scene + "/meta/",
        "mask_path": "/home/simba/Documents/project/BundleSDF/dataset/HO3D_v3/masks_XMem/" + scene,
        "out_dir": "/home/simba/Documents/project/diff_object_mary/threestudio/dataset/HO3D_v3_gt_poose/" + scene,
        "start_frame": 0,
        "end_frame": -1,
        "frame_interval": 5,
        "exclude_frames": [],
    }
    from attrdict import AttrDict
    config = AttrDict(config)
    preprocessCamerasFromRealImageWithGtPose(config)   

def RunPreprocessCamerasFromBlenderJson():
    scene = "cracker_box"
    config = {
        "json_path": "/home/simba/Documents/project/hold-private/code/data/blender/cracker_box",
        "json_file": "transforms_train.json",
    }
    from attrdict import AttrDict
    config = AttrDict(config)
    preprocessCamerasFromBlenderJson(config)   

if __name__ == "__main__":
    # RunPreprocessCamerasFromRealImageWithGtPose()
    RunPreprocessCamerasFromBlenderJson()
 