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
sys.path.append("/home/simba/Documents/project/diff_object_mary/threestudio")
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

