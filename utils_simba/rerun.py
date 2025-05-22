import rerun as rr
import numpy as np

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