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

def show_all_views(data, cam_axis_scale=0.3, axis_limit=1.0, z_scale=10):
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')
    # plot object origin
    t = np.array([0, 0, 0])
    ax.quiver(*t, *[1, 0, 0], color='r', length=0.3, normalize=False)
    ax.quiver(*t, *[0, 1, 0], color='g', length=0.3, normalize=False)
    ax.quiver(*t, *[0, 0 ,1], color='b', length=0.3, normalize=False)
    ax.text(*t, "o", fontsize=12, color='black')
    for i, c2w in enumerate(data['c2ws']):
        t = c2w[:3, 3]
        x = c2w[:3,0] * cam_axis_scale
        y = c2w[:3,1] * cam_axis_scale
        z = c2w[:3,2] * cam_axis_scale * z_scale
        ax.quiver(*t, *x[:3], color='r', length=0.3, normalize=False)
        ax.quiver(*t, *y[:3], color='g', length=0.3, normalize=False)
        ax.quiver(*t, *z[:3], color='b', length=0.3, normalize=False)
        try:
            label = data['labels'][i]
        except:
            label = f"{i}"
        ax.text(*t, label, fontsize=12, color='magenta')
        if label.startswith('gen'):
            plot_a_sphere(t, ax)

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
    if mesh_path is not None:
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
