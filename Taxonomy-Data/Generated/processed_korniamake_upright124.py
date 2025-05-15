import torch

def KORNIA_CHECK_LAF(laf):
    if laf.ndim != 4 or laf.shape[2:] != (2, 3):
        raise ValueError("Input LAF should have shape (B, N, 2, 3)")

def make_upright(laf, eps=1e-6):
    """
    Rectifies an affine matrix to make it upright.

    Args:
        laf (torch.Tensor): Input affine matrix of shape (B, N, 2, 3).
        eps (float, optional): Small value for safe division. Default is 1e-6.

    Returns:
        torch.Tensor: Rectified affine matrix of the same shape (B, N, 2, 3).
    """
    KORNIA_CHECK_LAF(laf)
    
    # Extract the 2x2 affine part
    A = laf[..., :2, :2]  # Shape (B, N, 2, 2)
    
    # Compute the determinant
    det = torch.det(A)  # Shape (B, N)
    
    # Perform SVD on the 2x2 matrices
    U, S, Vt = torch.linalg.svd(A)
    
    # Create an identity matrix for the rotation part
    identity = torch.eye(2, device=laf.device, dtype=laf.dtype).expand_as(A)
    
    # Scale the identity matrix by the original determinant
    scale = det.unsqueeze(-1).unsqueeze(-1) / (S.prod(dim=-1, keepdim=True) + eps)
    upright_A = identity * scale
    
    # Replace the original affine part with the upright version
    rectified_laf = laf.clone()
    rectified_laf[..., :2, :2] = upright_A
    
    return rectified_laf

