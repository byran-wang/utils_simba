from PIL import Image
import numpy as np
import json
import os
from pathlib import Path
from utils_simba.graphic import focal2fov, fov2focal
from typing import NamedTuple
from rich.console import Console
import math
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
    image_name: str

def readCamerasFromBlenderJson(path, transformsfile, cam_type="cvc2cvw"):
    # Read the camera information from the transforms.json file
    cam_type = cam_type.lower() # "cvc2cvw" or "cvc2blw" or "blc2blw"
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        meta = json.load(json_file)
        num_skipped_image_filenames = 0
        fl_x, fl_y = get_focal_lengths(meta)
        width = meta["w"]
        height = meta["h"]
        fov_x = math.atan(width / (2 * fl_y)) * 2
        fov_y = math.atan(height / (2 * fl_y)) * 2
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
                cam_infos.append(CameraInfo(uid=idx, c2w4x4=c2w_final, fl_x=fl_x, fl_y=fl_y, image_name=fname, 
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