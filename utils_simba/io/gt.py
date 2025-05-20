import os.path as op
from glob import glob

import common.viewer as viewer_utils
import numpy as np
import torch
import trimesh
from common.body_models import build_mano_aa
from common.transforms import project2d_batch, rigid_tf_torch_batch
from common.viewer import ViewerData
from PIL import Image

# from src_data.preprocessing_utils import tf.cv2gl_mano
import common.transforms as tf

# from src_data.smplx import MANO
from common.xdict import xdict
from src.utils.eval_modules import compute_bounding_box_centers
from src.utils.const import SEGM_IDS
import os
import pickle
import cv2
import trimesh
import re
from aitviewer.renderables.meshes import Meshes
from aitviewer.utils.so3 import aa2rot_numpy

glcam_in_cvcam = np.array([[1,0,0,0],
                          [0,-1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]])

def get_gt_mesh(config):
    video2name = {
      'AP': '019_pitcher_base',
      'MPM': '010_potted_meat_can',
      'SB': '021_bleach_cleanser',
      'SM': '006_mustard_bottle',
      "ABF": "021_bleach_cleanser",
      "BB": "011_banana",
      "GPMF": "010_potted_meat_can",
      "GSF": "037_scissors",
      "MC": "003_cracker_box",
      "MDF": "035_power_drill",
      "ND": "035_power_drill",
      "SMu": "025_mug",
      "SS": "004_sugar_box",
      "ShSu": "004_sugar_box",
      "SiBF": "011_banana",
      "SiS": "004_sugar_box",
      "SPS": "100_pantene_shampoo",           
    }
    video_name = config.cmd_args.seq_name
    data_dir = config.cmd_args.data_dir
    match_str = re.match(r"([a-zA-Z]+)(\d+)", video_name).group(1)
    for k in video2name:
      if match_str == k:
        ob_name = video2name[k]
        break
    mesh_path = os.path.join(data_dir, "../models", "", ob_name, "textured_simple.obj")
    texture_path = os.path.join(data_dir, "../models", "", ob_name, "texture_map.png")
    assert os.path.exists(mesh_path), f"Mesh file {mesh_path} not found"
    obj = trimesh.load(mesh_path)
    mesh = {}
    mesh["obj"] = obj
    mesh["texture_path"] = texture_path
    return mesh

def load_data(data_dir, seq_name, config):
    # load in opencv format
    fnames_all = sorted(glob(f"{data_dir}/{seq_name}/rgb/*.jpg"))
    if config.dataset.end_frame == -1:
        end_fname = fnames_all[-1]
        end = int(op.basename(end_fname).split(".")[0])
        config.dataset.end_frame = end
    assert config.dataset.end_frame > config.dataset.start_frame
    valid_frame = list(range(config.dataset.start_frame, config.dataset.end_frame, config.dataset.frame_interval))
    fnames = []
    
    for i in valid_frame:
        fname = os.path.join(data_dir, seq_name, "rgb", f"{i:04d}.jpg")
        if (fname in fnames_all):
            if i not in config.dataset.exclude_frames:
                fnames.append(fname)
        else:
            print(f"Warning: {fname} not found")
    assert len(fnames) > 0
    Ks = []
    ob_in_cams = []
    color_fs = []
    mask_fs = []
    for fname in fnames:

        meta_file = fname.replace('.jpg','.pkl').replace('rgb','meta')
        assert os.path.exists(meta_file), f"Meta file {meta_file} not found"
        meta = pickle.load(open(meta_file,'rb'))


        if meta['objTrans'] is None or meta['objRot'] is None or meta['camMat'] is None:
            print(f"Warning: Pose not found for Meta file {meta_file}, and skip frame {fname}")
            continue
        color_fs.append(fname)
        index = int(os.path.basename(fname).split(".")[0])
        mask_f = os.path.join(f"{data_dir}", "../masks_XMem", seq_name, f"{index:05d}.png")
        mask_fs.append(mask_f)
        K = meta['camMat']
        ob_in_cam = np.eye(4)
        assert not (meta['objTrans'] is None), f"Not pose for Meta file {meta_file}"
        ob_in_cam[:3,3] = meta['objTrans']
        ob_in_cam[:3,:3] = cv2.Rodrigues(meta['objRot'].reshape(3))[0]
        ob_in_cam = glcam_in_cvcam @ ob_in_cam
        Ks.append(K)
        ob_in_cams.append(ob_in_cam)
    
    assert len(color_fs) > 0 and len(color_fs) == len(ob_in_cams) and len(color_fs) == len(Ks)
    mesh = get_gt_mesh(config)

    out = {}
    out["color_fs"] = color_fs
    out["mask_fs"] = mask_fs
    out["ob_in_cams"] = np.asarray(ob_in_cams)
    out["K"] = Ks[0]
    out["mesh"] = mesh

    return out


def load_viewer_data(config):
    data = load_data(config)
    Rt = data["ob_in_cams"][:, :3, :]
    K = data["K"].reshape(3, 3)
    fnames = data["color_fs"]


    texture_image = data['mesh']['texture_path']
    obj = data['mesh']['obj']

    rotation_flip = aa2rot_numpy(np.array([1, 0, 0]) * np.pi)
    # breakpoint()
    t = Rt[:, :3, 3:]
    R = Rt[:, :3, :3]
    t_gl = rotation_flip @ t
    R_gl = rotation_flip @ R
    Rt[:, :3, 3:] = t_gl
    Rt[:, :3, :3] = R_gl

    meshes = Meshes(
        obj.vertices,
        obj.faces,
        obj.vertex_normals,
        uv_coords=obj.visual.uv,
        path_to_texture=texture_image,
        rotation=rotation_flip,
        # scale=50.0,
        # color=(1, 1, 1, 0.5),
    )

    im = Image.open(fnames[0])
    cols, rows = im.size

    images = [Image.open(im_p) for im_p in fnames]
    data = ViewerData(Rt, K, cols, rows, images)
    return {"object": meshes}, data
