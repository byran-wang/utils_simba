from PIL import Image
import numpy as np
import json
import os
from pathlib import Path
from utils_simba.graphic import focal2fov, fov2focal
from typing import NamedTuple
from rich.console import Console
import math
from glob import glob
import pickle
import cv2
import os.path as op
from threestudio.data import colmap_read_model as read_model
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
    image_name: str
    mask_name: str
    pts3d: np.array

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
            fname = path / Path(frame["file_path"])
            if not fname:
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
                cam_infos.append(CameraInfo(uid=idx, c2w4x4=c2w_final, fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy, image_name=fname, 
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

def readCamerasFromRealImageWithGTPose(config):
    cam_type = config.cam_type.lower() # "cvc2cvw" or "cvc2blw" or "blc2blw"
    cam_infos = []
    rgba_all = sorted(glob(f"{config['rgba_path']}/*.png"))
    pose_all = sorted(glob(f"{config['pose_path']}/*.pkl"))
    if config.end_frame == -1:
        end_fname = rgba_all[-1]
        end = int(op.basename(end_fname).split(".")[0].split("_rgba")[0])
        config.end_frame = end
    assert config.end_frame > config.start_frame

    frame_range = list(range(config.start_frame, config.end_frame, config.frame_interval))
    valid_frames = []
    
    for i in frame_range:
        rgba = os.path.join(config['rgba_path'], f"{i:04d}_rgba.png")
        if (rgba in rgba_all):
            pose = os.path.join(config['pose_path'], f"{i:04d}.pkl")
            if pose in pose_all:
                if i not in config.exclude_frames:
                    valid_frames.append([rgba, pose])
            else:
                print(f"Warning: {pose} not found")
        else:
            print(f"Warning: {rgba} not found")
    assert len(valid_frames) > 0
    for ci, [rgba_f, pose_f] in enumerate(valid_frames):
        meta = pickle.load(open(pose_f,'rb'))
        if meta['objTrans'] is None or meta['objRot'] is None or meta['camMat'] is None:
            print(f"Warning: Pose not found for Meta file {pose_f}, and skip frame {rgba_f}")
            continue
        if cam_infos == []:
            K = meta['camMat']
            fl_x, fl_y = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            rgba_imgae = cv2.imread(rgba_f, cv2.IMREAD_UNCHANGED)
            height, width = rgba_imgae.shape[:2]
            fov_x = math.atan(width / (2 * fl_x)) * 2
            fov_y = math.atan(height / (2 * fl_y)) * 2
            cvw2blw = np.array([[0, 0, -1, 0],[1, 0 , 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            blc2glc = np.array([[1, 0, 0, 0],[0, 1 , 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            glc2cvc = np.array([[1, 0, 0, 0],[0, -1 , 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        cvw2glc = np.eye(4)
        cvw2glc[:3,3] = meta['objTrans']
        cvw2glc[:3,:3] = cv2.Rodrigues(meta['objRot'].reshape(3))[0]
        glc2cvw = np.linalg.inv(cvw2glc)
        if cam_type == "blc2blw":
            blc2blw = cvw2blw @ glc2cvw @ blc2glc
            c2w_final = blc2blw
        if cam_type == "cvc2cvw":
            cvw2cvc = glc2cvc @ cvw2glc
            cvc2cvw = np.linalg.inv(cvw2cvc)
            c2w_final = cvc2cvw
        cam_infos.append(CameraInfo(uid=ci, c2w4x4=c2w_final, fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy, image_name=rgba_f, 
                            height=height, width=width, fov_x=fov_x, fov_y=fov_y))
        
    return cam_infos

def readCamerasFromRealImageWithColmapPose(config):
    
    cam_type = config.cam_type.lower() # "cvc2cvw" or "cvc2blw" or "blc2blw"
    
    camerasfile = os.path.join(config.data_path, 'cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)

    imagesfile = os.path.join(config.data_path, 'images.bin')
    imdata = read_model.read_images_binary(imagesfile)    

    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])

    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    ptsdata = read_model.read_points3d_binary(os.path.join(config.data_path, "points3D.bin"))
    ptskeys = np.array(sorted(ptsdata.keys()))
    pts3d = np.array([ptsdata[k].xyz for k in ptskeys]) # [M, 3]

    images_list = sorted(glob(os.path.join(config.data_path, "images/*.jpg")))
    #depth_list = sorted(glob(os.path.join(self.cfg.root_dir,"depths/*.png")))
    mask_list = sorted(glob(os.path.join(config.data_path,"masks/*.jpg")))    

    cam_infos = CameraInfo(uid=None, c2w4x4=c2w_mats, fl_x=f, fl_y=f, cx=cx, cy=cy, image_name=images_list, mask_name=mask_list, pts3d=pts3d,
                        height=h, width=w, fov_x=f, fov_y=f)
        
    return cam_infos