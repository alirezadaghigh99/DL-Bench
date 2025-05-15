import torch
from torch_sparse import coalesce

def cot_laplacian(verts: torch.Tensor, faces: torch.Tensor, eps: float = 1e-12):
    # Number of vertices
    n_verts = verts.size(0)
    
    # Extract vertices of each face
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    
    # Compute edge vectors
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2
    
    # Compute cotangent weights
    cot0 = torch.sum(e1 * e2, dim=1) / (torch.norm(torch.cross(e1, e2), dim=1) + eps)
    cot1 = torch.sum(e2 * e0, dim=1) / (torch.norm(torch.cross(e2, e0), dim=1) + eps)
    cot2 = torch.sum(e0 * e1, dim=1) / (torch.norm(torch.cross(e0, e1), dim=1) + eps)
    
    # Compute face areas
    face_areas = 0.5 * torch.norm(torch.cross(e0, e1), dim=1)
    
    # Clamp face areas to avoid division by zero
    face_areas = torch.clamp(face_areas, min=eps)
    
    # Inverse face areas
    inv_face_areas = 1.0 / face_areas
    
    # Prepare indices for sparse matrix
    I = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 1], faces[:, 2]])
    J = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 2], faces[:, 0], faces[:, 1]])
    W = torch.cat([cot0, cot1, cot2, cot0, cot1, cot2])
    
    # Create sparse Laplacian matrix
    L = torch.sparse_coo_tensor(torch.stack([I, J]), W, (n_verts, n_verts))
    
    # Add diagonal elements
    diag_indices = torch.arange(n_verts)
    diag_values = torch.zeros(n_verts, device=verts.device)
    for i in range(3):
        diag_values.index_add_(0, faces[:, i], W[i::3])
    
    L = L + torch.sparse_coo_tensor(torch.stack([diag_indices, diag_indices]), -diag_values, (n_verts, n_verts))
    
    # Coalesce to sum duplicate entries
    L = coalesce(L.indices(), L.values(), m=n_verts, n=n_verts)
    
    return L, inv_face_areas

