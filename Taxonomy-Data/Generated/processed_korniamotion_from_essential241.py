import torch

def motion_from_essential(E_mat: torch.Tensor):
    # Check the shape of the input tensor
    assert E_mat.shape[-2:] == (3, 3), "E_mat must have shape (*, 3, 3)"
    
    # Define the W and Z matrices used in the decomposition
    W = torch.tensor([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]], dtype=E_mat.dtype, device=E_mat.device)
    
    Z = torch.tensor([[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 0]], dtype=E_mat.dtype, device=E_mat.device)
    
    # Perform SVD on the essential matrix
    U, _, Vt = torch.linalg.svd(E_mat)
    
    # Ensure a proper rotation matrix by adjusting the determinant
    if torch.det(U @ Vt) < 0:
        Vt = -Vt
    
    # Compute the two possible rotation matrices
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    # Compute the translation vector (up to scale)
    t_skew = U @ Z @ U.transpose(-2, -1)
    t = torch.stack([t_skew[..., 2, 1], t_skew[..., 0, 2], t_skew[..., 1, 0]], dim=-1).unsqueeze(-1)
    
    # Normalize the translation vector
    t = t / torch.norm(t, dim=-2, keepdim=True)
    
    # Prepare the output tensors
    Rs = torch.stack([R1, R1, R2, R2], dim=-3)
    Ts = torch.stack([t, -t, t, -t], dim=-3)
    
    return Rs, Ts

