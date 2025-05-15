import torch

def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    # Ensure the last dimension is 3
    if not x.shape[-1] == 3:
        raise AssertionError(f"Expected last dimension to be 3, but got {x.shape[-1]}")
    
    # Extract components
    a, b, c = x[..., 0], x[..., 1], x[..., 2]
    
    # Create the skew-symmetric matrix
    zero = torch.zeros_like(a)
    cross_prod_mat = torch.stack([
        torch.stack([zero, -c, b], dim=-1),
        torch.stack([c, zero, -a], dim=-1),
        torch.stack([-b, a, zero], dim=-1)
    ], dim=-2)
    
    return cross_prod_mat

