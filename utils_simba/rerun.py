import rerun as rr
import numpy as np
import os
import cv2
import trimesh
import rerun.blueprint as rrb
from .vis import rotation_matrix_to_quaternion

def add_material(color: list) -> rr.Material:
    """
    Creates a ReRun material with the specified color.

    Parameters:
        color (list): RGBA color list.

    Returns:
        rr.Material: ReRun material instance.
    """
    return rr.Material(albedo_factor=color)

def compute_vertex_normals(vertices, faces):
    # Initialize normals to zero
    normals = np.zeros(vertices.shape, dtype=np.float32)
    
    # Compute normals for each face
    for face in faces:
        idx0, idx1, idx2 = face
        v0 = vertices[idx0]
        v1 = vertices[idx1]
        v2 = vertices[idx2]
        
        # Compute the normal of the face
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        
        # Add the face normal to each vertex normal
        normals[idx0] += face_normal
        normals[idx1] += face_normal
        normals[idx2] += face_normal
    
    # Normalize the normals
    norm = np.linalg.norm(normals, axis=1)
    norm[norm == 0] = 1  # Avoid division by zero
    normals /= norm[:, np.newaxis]
    
    return normals

class Visualizer:
    def __init__(
        self,
        viewer_name: str = "trellis",
        jpeg_quality: int = 75,
        log_axis: bool = True,
        world_coordinate: str = "object",
    ) -> None:
        # To be parametrized later
        self._jpeg_quality = jpeg_quality
        #
        # Prepare the rerun rerun log configuration
        #
        blueprint = rrb.Vertical(
            rrb.Spatial3DView(name="object", 
                            defaults=[rr.components.ImagePlaneDistance(1.0)],
                            origin="/"),                         
            rrb.Horizontal(
                rrb.Spatial2DView(name="camera", origin="/camera"),
            ),
            row_shares=[5, 2],
        )
        rr.init(viewer_name, spawn=True)
        rr.send_blueprint(blueprint)     

        # rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        # rr.log("plot/normal_mean", rr.SeriesLine(color=[240, 45, 58]), static=True)
        # rr.log("plot/normal_variance", rr.SeriesLine(color=[188, 77, 165]), static=True)

        if log_axis:
            self.log_axis()
        self.world_coordinate = world_coordinate
        self.world_transform = np.eye(4)  

    def log_mesh(self, label: str, 
                 mesh_file: str,
                 material: rr.Material = None, 
                 colors: np.ndarray = None, 
                 normals: np.ndarray = None, 
                 static=False, 
                 faces_downsample_ratio: float = 1,
                 ) -> None:
        assert os.path.exists(mesh_file), f"Mesh file {mesh_file} does not exist"
        mesh = trimesh.load(mesh_file)
        vertices = mesh.vertices
        faces = mesh.faces
        if faces_downsample_ratio < 1:
            # random sample faces
            indices = np.random.choice(faces.shape[0], int(faces.shape[0] * faces_downsample_ratio), replace=False)
            faces = faces[indices]
        vertices = (self.world_transform[:3,:3] @ vertices.T + self.world_transform[:3,3:4]).T
        if colors is None:    
            if normals is None:
                normals = compute_vertex_normals(vertices, faces)
            rr.log(label, rr.Mesh3D(
                vertex_positions = vertices,
                triangle_indices = faces,
                vertex_normals = normals,
                mesh_material = material,
            ), static=static)
        else:
            rr.log(label, rr.Mesh3D(
                vertex_positions = vertices,
                triangle_indices = faces,
                vertex_normals = normals,
                vertex_colors = colors,
            ), static=static)        

    def log_image(self,label: str, 
                  image_file: str, 
                  jpeg_quality: int = 75, 
                  static=False,
                  ) -> None:
        assert os.path.exists(image_file), f"Image file {image_file} does not exist"
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rr.log(label, rr.Image(image).compress(jpeg_quality=jpeg_quality), static=static)

    def log_calibration(
        self,
        label: str,
        resolution: list[int], # [width, height]
        intrins: np.ndarray,
        image_plane_distance: float = 0.1,
        static=False,
    ) -> None:
        rr.log(
            label,
            rr.Pinhole(
                resolution=resolution,
                focal_length=[intrins[0,0], intrins[1,1]],
                principal_point=[intrins[0,2], intrins[1,2]],
                image_plane_distance=image_plane_distance,
            ),
            static=static,
        )

    def log_cam_pose(self, label: str, 
                     c2w: np.ndarray, 
                     static=False,
                     ) -> None:
        c2w = self.world_transform @ c2w
        tvec = c2w[:3, 3]
        quat_xyzw = rotation_matrix_to_quaternion(c2w[:3, :3])
        rr.log(label, rr.Transform3D(translation=tvec, rotation=rr.Quaternion(xyzw=quat_xyzw)), static=static)

    def set_time_sequence(self, frame_index: int) -> None:
        rr.set_time_sequence("frame_index", frame_index)

    def log_points(self, label: str, 
                   points: np.ndarray, 
                   colors: np.ndarray = None, 
                   sizes: np.ndarray = None,
                   radii: float = 0.001,
                   static=False,
                   ) -> None:
        points = (self.world_transform[:3,:3] @ points.T + self.world_transform[:3,3:4]).T
        if colors is None:
            rr.log(label, rr.Points3D(positions=points, radii=radii), static=static)
        else:
            if sizes is None:
                rr.log(label, rr.Points3D(positions=points, colors=colors, radii=radii), static=static)
            else:
                rr.log(label, rr.Points3D(positions=points, colors=colors, radii=sizes), static=static)

    def log_axis(self,
                       label: str = "world/", 
                       scale: float = 1.0):
        origins = np.zeros((3, 3))
        ends = np.eye(3) * scale
        colors = np.eye(3,4)
        colors[:,-1] = 1
        rr.log(f"{label}axis", rr.Arrows3D(origins=origins, vectors=ends, colors=colors), timeless=True)             