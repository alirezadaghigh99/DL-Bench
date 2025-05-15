import torch

def fundamental_from_projections(P1, P2):
    # Ensure the input tensors are of the correct shape
    assert P1.ndim >= 3 and P1.shape[-2:] == (3, 4), "P1 must have shape (*, 3, 4)"
    assert P2.ndim >= 3 and P2.shape[-2:] == (3, 4), "P2 must have shape (*, 3, 4)"
    assert P1.shape[:-2] == P2.shape[:-2], "P1 and P2 must have matching batch dimensions"
    
    # Convert inputs to float32 if they are not float32 or float64
    if P1.dtype not in [torch.float32, torch.float64]:
        P1 = P1.to(torch.float32)
    if P2.dtype not in [torch.float32, torch.float64]:
        P2 = P2.to(torch.float32)
    
    # Extract the last column of each projection matrix
    e1 = P1[..., :, 3]
    e2 = P2[..., :, 3]
    
    # Compute the cross product matrix for e2
    def cross_product_matrix(e):
        zero = torch.zeros_like(e[..., 0])
        return torch.stack([
            zero, -e[..., 2], e[..., 1],
            e[..., 2], zero, -e[..., 0],
            -e[..., 1], e[..., 0], zero
        ], dim=-1).reshape(*e.shape[:-1], 3, 3)
    
    e2_cross = cross_product_matrix(e2)
    
    # Compute the fundamental matrix
    F = e2_cross @ P2[..., :, :3] @ torch.linalg.pinv(P1[..., :, :3])
    
    return F

