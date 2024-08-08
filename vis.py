"""
2D visualization primitives based on Matplotlib.

1) Plot images with `plot_images`.
2) Call `plot_keypoints` or `plot_matches` any number of times.
3) Optionally: save a .png or .pdf plot (nice in papers!) with `save_plot`.
"""

import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch


def cm_RdGn(x):
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0]]) + (2 - x) * np.array([[1.0, 0, 0]])
    return np.clip(c, 0, 1)


def plot_images(
    imgs, titles=None, cmaps="gray", dpi=100, pad=0.5, adaptive=True, figsize=4.5
):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4 / 3] * n
    figsize = [sum(ratios) * figsize, figsize]
    fig, axs = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios}
    )
    if n == 1:
        axs = [axs]
    for i, (img, ax) in enumerate(zip(imgs, axs)):
        ax.imshow(img, cmap=plt.get_cmap(cmaps[i]))
        ax.set_axis_off()
        if titles:
            ax.set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_keypoints(kpts, colors="lime", ps=4):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    axes = plt.gcf().axes
    for a, k, c in zip(axes, kpts, colors):
        a.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0)


def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, indices=(0, 1), a=1.0):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        # transform the points into the figure coordinate system
        for i in range(len(kpts0)):
            fig.add_artist(
                matplotlib.patches.ConnectionPatch(
                    xyA=(kpts0[i, 0], kpts0[i, 1]),
                    coordsA=ax0.transData,
                    xyB=(kpts1[i, 0], kpts1[i, 1]),
                    coordsB=ax1.transData,
                    zorder=1,
                    color=color[i],
                    linewidth=lw,
                    alpha=a,
                )
            )

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def add_text(
    idx,
    text,
    pos=(0.01, 0.99),
    fs=15,
    color="w",
    lcolor="k",
    lwidth=2,
    ha="left",
    va="top",
):
    ax = plt.gcf().axes[idx]
    t = ax.text(
        *pos, text, fontsize=fs, ha=ha, va=va, color=color, transform=ax.transAxes
    )
    if lcolor is not None:
        t.set_path_effects(
            [
                path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
                path_effects.Normal(),
            ]
        )


def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)

def plot_a_sphere(location, ax, radius=0.01):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + location[0]  # Offset by location[0]
    y = radius * np.sin(u) * np.sin(v) + location[1]  # Offset by location[1]
    z = radius * np.cos(v) + location[2]              # Offset by location[2]
    ax.plot_surface(x, y, z, color='cyan', alpha=0.6)  # alpha for transparency

def show_all_views(data, scene_data, asset_3D_paths = None, intrinsic_sel = 0, cam_axis_scale=0.3, axis_limit=1, z_scale=30, uvs=[[302, 290]]):
    # uvs = [[312.4200134277344, 241.4199981689453]] # principal point of MC1 view_98
    cx = scene_data['cx'][intrinsic_sel]
    cy = scene_data['cy'][intrinsic_sel]
    focal_length = scene_data['focal_length'][intrinsic_sel][0]
    
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')
    # plot object origin
    t = np.array([0, 0, 0])
    ax.quiver(*t, *[1, 0, 0], color='r', length=0.3, normalize=False)
    ax.quiver(*t, *[0, 1, 0], color='g', length=0.3, normalize=False)
    ax.quiver(*t, *[0, 0 ,1], color='b', length=0.3, normalize=False)
    ax.text(*t, "o", fontsize=12, color='black')
    for i, glc2blw4x4 in enumerate(data['c2ws']):
        cvc2glc = np.array([[1, 0, 0, 0],[0, -1 , 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        glc2blw4x4_np = np.array(glc2blw4x4)
        cvc2blw4x4 = glc2blw4x4_np @ cvc2glc
        t = cvc2blw4x4[:3, 3]
        x = cvc2blw4x4[:3,0] * cam_axis_scale
        y = cvc2blw4x4[:3,1] * cam_axis_scale
        z = cvc2blw4x4[:3,2] * cam_axis_scale * z_scale
        ax.quiver(*t, *x[:3], color='r', length=0.3, normalize=False)
        ax.quiver(*t, *y[:3], color='g', length=0.3, normalize=False)
        ax.quiver(*t, *z[:3], color='b', length=0.3, normalize=False)

        for uv in uvs:
            X = (uv[0] - cx) / focal_length
            Y = (uv[1] - cy) / focal_length
            Z = torch.tensor(1)
            ray = np.array([X, Y, Z])[None].T
            ray_cv = ray / np.linalg.norm(ray)
            ray_gl = cvc2glc[:3, :3] @ ray_cv + cvc2glc[:3, 3][:, None]
            ray_blw = glc2blw4x4_np[:3, :3] @ ray_gl

            ray_blw_s = ray_blw * cam_axis_scale * z_scale
            ax.quiver(*t, *ray_blw_s, color='magenta', length=0.3, normalize=False)
            num_points = 1000
            points = np.linspace(t, ray_blw_s.squeeze(-1), num_points)
            from utils_simba.geometry import save_point_cloud_to_ply
            save_point_cloud_to_ply(points, f"ray_{i}.ply")
        try:
            label = data['labels'][i]
        except:
            label = f"{i}"
        ax.text(*t, label, fontsize=12, color='magenta')
        if label.startswith('gen'):
            plot_a_sphere(t, ax)

    if asset_3D_paths is not None:
        interval = 10
        for asset_3D_path in asset_3D_paths:
            mesh_info = get_mesh_info(asset_3D_path)
            mesh_info["vertex_positions"] = mesh_info["vertex_positions"][::interval]
            mesh_info["vertex_colors"] = mesh_info["vertex_colors"][::interval]
            x = mesh_info["vertex_positions"][:, 0]
            y = mesh_info["vertex_positions"][:, 1]
            z = mesh_info["vertex_positions"][:, 2]
            colors = mesh_info["vertex_colors"] / 255
            scatter = ax.scatter(x, y, z, c=colors, cmap='viridis')
            color_bar = fig.colorbar(scatter)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Camera Positions and Orientations')
    # Set plot limits
    limit = axis_limit
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.grid(False)
    plt.tight_layout()
    plt.show()
    

from pytorch3d.renderer import PerspectiveCameras, RayBundle
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.vis import (  # noqa: F401
    plot_scene,
)
from pytorch3d.vis.plotly_vis import AxisArgs
from pytorch3d.renderer import ray_bundle_to_ray_points
from pytorch3d.structures import Pointclouds    
def show_scene(c2ws4x4, rays_o, rays_d, cfg):
    mesh_path = cfg.mesh_path
    min_depth = cfg.min_depth
    max_depth = cfg.max_depth
    n_pts_per_ray = cfg.n_pts_per_ray
    camera_trace = {}
    if c2ws4x4.shape[-2:] == (3, 4):
        new_row = torch.tensor([0, 0, 0, 1])[None,None]
        c2ws4x4 = torch.cat((c2ws4x4, new_row.repeat(c2ws4x4.shape[0], 1, 1)), dim=1)
    for ci, c2w in enumerate(c2ws4x4):
        w2c = c2w.inverse()
        R = (w2c[:3, :3].T)[None] # transpose due to row-major
        T = w2c[:3, 3][None]
        cam = PerspectiveCameras(R=R, T=T)
        camera_trace[f"camera_{ci:03d}"] = cam
    if mesh_path != 'None':
        meshes = load_objs_as_meshes(mesh_path, create_texture_atlas=cfg.show_mesh_texture, texture_atlas_size=1)
    else:
        meshes = None
    
    rays = []
    for origin, dir in zip(rays_o, rays_d):
        origin = origin.view(-1, 3)
        dir = dir.view(-1, 3)
        n_rays = origin.shape[0]
        depth = torch.linspace(min_depth, max_depth, n_pts_per_ray)[None].repeat(n_rays, 1)
        ray = RayBundle(origins=origin[None],
                        xys=None,
                        directions=dir[None],
                        lengths=depth)
        rays.append(ray)

    ray_pts_trace = {}
    for ci, ray in enumerate(rays):
        pc = Pointclouds(ray_bundle_to_ray_points(ray).detach().cpu().view(1, -1, 3))
        ray_pts_trace[f"ray_pts_{ci:03d}"] = pc
    
    mesh_trace = {}
    if meshes is not None:
        for mi, mesh in enumerate(meshes):
            mesh_trace[f"mesh_{mi:03d}"] = mesh
        scene = {"scene": {**camera_trace, **ray_pts_trace, **mesh_trace}}
    else:
        scene = {"scene": {**camera_trace, **ray_pts_trace}}
    fig = plot_scene(scene,
                    camera_scale = 0.1,
                    axis_args = AxisArgs(showline=True, showgrid=True, zeroline=True, showticklabels=True, backgroundcolor="rgb(220,255,228)"),
                )
    fig.show()             

import rerun as rr  # pip install rerun-sdk
import rerun.blueprint as rrb
import re
import os
import cv2

def rotation_matrix_to_quaternion(R):
        """
        Convert a rotation matrix to a quaternion.

        Parameters:
        - R: A 3x3 rotation matrix.

        Returns:
        - A quaternion in the format [x, y, z, w].
        """
        # Make sure the matrix is a numpy array
        R = np.asarray(R)
        # Allocate space for the quaternion
        q = np.empty((4,), dtype=np.float32)
        # Compute the quaternion components
        q[3] = np.sqrt(np.maximum(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        q[0] = np.sqrt(np.maximum(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
        q[1] = np.sqrt(np.maximum(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
        q[2] = np.sqrt(np.maximum(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
        q[0] *= np.sign(q[0] * (R[2, 1] - R[1, 2]))
        q[1] *= np.sign(q[1] * (R[0, 2] - R[2, 0]))
        q[2] *= np.sign(q[2] * (R[1, 0] - R[0, 1]))
        return q

import trimesh
def get_mesh_info(mesh_file):
    mesh = trimesh.load(mesh_file)
    if mesh.is_empty:
        raise ValueError("The mesh could not be loaded. Please check the file path and format.")
    # mesh.show()
    vertex_positions = mesh.vertices
    vertex_normals = mesh.vertex_normals
    if mesh.visual.kind == 'texture':
        vertex_colors = mesh.visual.to_color().vertex_colors
    else:
        vertex_colors = None

    triangle_indices = mesh.faces
    return {"vertex_positions": vertex_positions,
            "vertex_normals": vertex_normals,
            "vertex_colors": vertex_colors,
            "triangle_indices": triangle_indices,
            }

def show_scene_in_rerun(scene_data):
    blueprint = rrb.Vertical(
        rrb.Spatial3DView(name="3D", origin="/"),
        rrb.Horizontal(
            rrb.Spatial2DView(name="Camera", origin="/camera/image"),
            rrb.TimeSeriesView(origin="/plot"),
        ),
        row_shares=[3, 2],
    )
    rr.init("rerun_diff_object", default_enabled=True, strict=True)
    rec: RecordingStream = rr.get_global_data_recording()  # type: ignore[assignment]
    rec.spawn(default_blueprint=blueprint)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True) 
    
    for image_file, glc2blw4x4 in zip(scene_data.image_name, scene_data.c2w):
        
        idx_match = re.search(r"\d+", os.path.basename(image_file))
        assert idx_match is not None
        frame_idx = int(idx_match.group(0))
        
        rr.set_time_sequence("frame", frame_idx)
        ### viz the mesh
        mesh_info = get_mesh_info("./outputs/MC1_rr_3d_only/Phase1/save/it1000-export/model.obj")
        rr.log(
            "/asset", 
            rr.Mesh3D(
                vertex_positions=mesh_info["vertex_positions"],
                vertex_normals=mesh_info["vertex_normals"],
                vertex_colors=mesh_info["vertex_colors"],
                triangle_indices=mesh_info["triangle_indices"],
            ),           
        )


        ### viz the camera
        cvc2glc = np.array([[1, 0, 0, 0],[0, -1 , 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        glc2blw4x4_np = np.array(glc2blw4x4)
        cvc2blw4x4 = glc2blw4x4_np @ cvc2glc
        translation = cvc2blw4x4[:3, 3]
        quaternion = rotation_matrix_to_quaternion(cvc2blw4x4[:3,:3])
        # scene_data.cx = 142 *2
        # scene_data.cy = 126 * 2
        rr.log(
            "camera/image",
            rr.Pinhole(
                resolution=[scene_data.width, scene_data.height],
                focal_length=np.array([scene_data.focal_length, scene_data.focal_length]).reshape(-1),
                principal_point=np.array([scene_data.cx, scene_data.cy]).reshape(-1),
            ),
        )
        bgr = cv2.imread(image_file)
        bgr = cv2.resize(bgr, (scene_data.width, scene_data.height), interpolation=cv2.INTER_AREA)
        bgr[int(scene_data.cy), int(scene_data.cx), :] = 0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rr.log("camera/image", rr.Image(rgb))

        tf = rr.Transform3D(translation=translation, rotation=quaternion, from_parent=False)       
        rr.log("camera", tf)
        rr.log("camera", rr.ViewCoordinates.RDF, static=True)  # X=Right, Y=Down, Z=Forward    

def set_blueprint(condition_data, observed_data):
    cond_views = [
        rrb.Spatial2DView(
            name=f"cond_{cond_index}",
            origin=f"world/cond_image/cond_{cond_index}",
        )
        for cond_index in condition_data["image_index"]
    ]
    observed_views = [
        rrb.Spatial2DView(
            name=f"observed_{observed_index}",
            origin=f"world/observed_image/observed_{observed_index}",
        )
        for observed_index in observed_data["image_index"]
    ]
    blueprint = rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="3D",
                origin="world",
                # Default for `ImagePlaneDistance` so that the pinhole frustum visualizations don't take up too much space.
                defaults=[rr.components.ImagePlaneDistance(0.5)],
                # Transform arrows for the vehicle shouldn't be too long.
                overrides={"world/object": [rr.components.AxisLength(5.0)]},
            ),
            # rrb.TextDocumentView(origin="description", name="Description"),
            column_shares=[3, 1],
        ),
        rrb.Grid(*(cond_views+observed_views)),
        row_shares=[4, 2],
    )
    return blueprint

def start_rr(rerun_name, blueprint):
    rr.init(rerun_name, default_enabled=True, strict=True)
    rec: RecordingStream = rr.get_global_data_recording()  # type: ignore[assignment]
    rec.spawn(default_blueprint=blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)   

def log_asset_3D(asset_3D_paths):
    for asset_3D_path in asset_3D_paths:
        mesh_info = get_mesh_info(asset_3D_path)
        mesh_name = asset_3D_path.split("/")[2]
        rr.log(
            f"world/{mesh_name}", 
            rr.Mesh3D(
                vertex_positions=mesh_info["vertex_positions"],
                vertex_normals=mesh_info["vertex_normals"],
                vertex_colors=mesh_info["vertex_colors"],
                triangle_indices=mesh_info["triangle_indices"],
            ),           
        )  

def log_asset_axis():
    origins = np.zeros((3, 3))
    ends = np.eye(3)
    colors = np.eye(3,4)
    colors[:,-1] = 1
    rr.log("world/axis", rr.Arrows3D(origins=origins, vectors=ends, colors=colors))

def show_cameras_images(scene_data, pre_fix, intrinsic_sel=0):
   ### viz the camera
    width = scene_data['width'][intrinsic_sel]
    height = scene_data['height'][intrinsic_sel]
    focal_length = scene_data['focal_length'][intrinsic_sel]
    cx = scene_data['cx'][intrinsic_sel]
    cy = scene_data['cy'][intrinsic_sel]

    
    
    for image_file, image_index, glc2blw4x4 in zip(scene_data['image_name'], scene_data['image_index'], scene_data['c2w4x4']):
        cvc2glc = np.array([[1, 0, 0, 0],[0, -1 , 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        glc2blw4x4_np = np.array(glc2blw4x4)
        cvc2blw4x4 = glc2blw4x4_np @ cvc2glc
        translation = cvc2blw4x4[:3, 3]
        quaternion = rotation_matrix_to_quaternion(cvc2blw4x4[:3,:3])

        tf = rr.Transform3D(translation=translation, rotation=quaternion, from_parent=False)
        rr.log(f"world/{pre_fix}image/{pre_fix}{image_index}", tf)

        rr.log(
            f"world/{pre_fix}image/{pre_fix}{image_index}",
            rr.Pinhole(
                resolution=[width, height],
                focal_length=np.array([focal_length, focal_length]).reshape(-1),
                principal_point=np.array([cx, cy]).reshape(-1),
            ),
        )
        rr.log(f"world/{pre_fix}image/{pre_fix}{image_index}", rr.ViewCoordinates.RDF, static=True)  # X=Right, Y=Down, Z=Forward    
        bgr = cv2.imread(image_file)
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        bgr[int(cy), int(cx), :] = 0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rr.log(f"world/{pre_fix}image/{pre_fix}{image_index}", rr.Image(rgb))

def rr_show_scene(condition_data, observed_data, rerun_name, asset_3D_path_list):
    blueprint = set_blueprint(condition_data, observed_data)
    start_rr(rerun_name, blueprint)
    rr.set_time_sequence("frame", 0)
    log_asset_3D(asset_3D_path_list)
    log_asset_axis()
    show_cameras_images(condition_data, "cond_", intrinsic_sel = 1)
    show_cameras_images(observed_data, "observed_")             