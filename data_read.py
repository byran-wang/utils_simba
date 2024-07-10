from PIL import Image
import numpy as np
import json
import os
from pathlib import Path
from utils_simba.graphic import focal2fov, fov2focal
from typing import NamedTuple
from rich.console import Console
CONSOLE = Console(width=120)

class CameraInfo(NamedTuple):
    uid: int
    w2c_cv: np.array
    fl_x: np.array
    fl_y: np.array
    image_name: str

def readCamerasFromBlenderJson(path, transformsfile):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        meta = json.load(json_file)
        num_skipped_image_filenames = 0
        fl_x, fl_y = get_focal_lengths(meta)
        gl_to_cv_t = np.array([[1, 0, 0, 0],[0, -1 , 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        for idx, frame in enumerate(meta["frames"]):
            fname = path / Path(frame["file_path"])
            if not fname:
                num_skipped_image_filenames += 1
            else:
                c2w_gl = np.array(frame["transform_matrix"])
                w2c_gl = np.linalg.inv(c2w_gl)
                w2c_cv = gl_to_cv_t @ w2c_gl
                cam_infos.append(CameraInfo(uid=idx, w2c_cv=w2c_cv, fl_x=fl_x, fl_y=fl_y, image_name=fname))
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