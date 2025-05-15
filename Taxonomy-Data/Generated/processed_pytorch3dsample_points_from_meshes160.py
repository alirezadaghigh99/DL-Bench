import torch
from pytorch3d.structures import Meshes

def sample_points_from_meshes(meshes, num_samples, return_normals=False, return_textures=False):
    """
    Convert a batch of meshes to a batch of point clouds by uniformly sampling points on the surface
    of the mesh with probability proportional to the face area.
    
    Args:
        meshes: A Meshes object with a batch of N meshes.
        num_samples: An integer specifying the number of point samples per mesh.
        return_normals: A boolean indicating whether to return normals for the sampled points.
        return_textures: A boolean indicating whether to return textures for the sampled points.
    
    Returns:
        A tuple containing:
        - samples: FloatTensor of shape (N, num_samples, 3) giving the coordinates of sampled points.
        - normals: FloatTensor of shape (N, num_samples, 3) giving a normal vector to each sampled point.
        - textures: FloatTensor of shape (N, num_samples, C) giving a C-dimensional texture vector to each sampled point.
    """
    device = meshes.device
    N = len(meshes)
    
    # Initialize outputs
    samples = torch.zeros((N, num_samples, 3), device=device)
    normals = torch.zeros((N, num_samples, 3), device=device) if return_normals else None
    textures = None
    
    if return_textures:
        # Assuming textures are stored in a C-dimensional vector per face
        C = meshes.textures.verts_features_packed().shape[-1]
        textures = torch.zeros((N, num_samples, C), device=device)
    
    for i, mesh in enumerate(meshes):
        if mesh.isempty():
            continue
        
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()
        
        # Calculate face areas
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        face_areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0), dim=1)
        
        # Sample faces based on area
        face_probs = face_areas / face_areas.sum()
        sampled_face_indices = torch.multinomial(face_probs, num_samples, replacement=True)
        
        # Sample points on the faces
        selected_faces = faces[sampled_face_indices]
        u = torch.sqrt(torch.rand(num_samples, device=device)).unsqueeze(1)
        v = torch.rand(num_samples, device=device).unsqueeze(1)
        points = (1 - u) * v0[selected_faces[:, 0]] + u * (1 - v) * v1[selected_faces[:, 1]] + u * v * v2[selected_faces[:, 2]]
        samples[i] = points
        
        if return_normals:
            # Calculate normals for the sampled points
            face_normals = torch.cross(v1 - v0, v2 - v0)
            face_normals = face_normals / face_normals.norm(dim=1, keepdim=True)
            normals[i] = face_normals[sampled_face_indices]
        
        if return_textures:
            # Calculate textures for the sampled points
            face_textures = mesh.textures.faces_verts_textures_packed()
            textures[i] = face_textures[sampled_face_indices]
    
    result = (samples,)
    if return_normals:
        result += (normals,)
    if return_textures:
        result += (textures,)
    
    return result