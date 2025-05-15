import torch

def run_5point(points1, points2, weights=None):
    """
    Computes the essential matrix using Nister's 5-point algorithm.
    
    Args:
        points1 (torch.Tensor): A tensor of shape (B, N, 2) representing the calibrated points from the first image.
        points2 (torch.Tensor): A tensor of shape (B, N, 2) representing the calibrated points from the second image.
        weights (torch.Tensor, optional): A tensor of shape (B, N) representing the weights for each point pair.
        
    Returns:
        torch.Tensor: A tensor of shape (B, 3, 3) representing the essential matrix for each batch.
    """
    # Validate input shapes
    assert points1.shape == points2.shape, "points1 and points2 must have the same shape"
    assert points1.shape[-1] == 2, "points1 and points2 must have shape (B, N, 2)"
    B, N, _ = points1.shape
    assert N >= 5, "At least 5 point correspondences are required"

    # Construct the linear system
    x1, y1 = points1[..., 0], points1[..., 1]
    x2, y2 = points2[..., 0], points2[..., 1]

    # Create the design matrix A
    A = torch.stack([
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1, y1, torch.ones_like(x1)
    ], dim=-1)  # Shape: (B, N, 9)

    if weights is not None:
        assert weights.shape == (B, N), "weights must have shape (B, N)"
        A *= weights.unsqueeze(-1)

    # Solve for the null space of A using SVD
    _, _, V = torch.svd(A)
    E_candidates = V[..., -1].reshape(B, 3, 3)  # Last column of V gives the solution

    # Enforce the rank-2 constraint on the essential matrix
    U, S, Vt = torch.svd(E_candidates)
    S = torch.diag_embed(torch.tensor([1.0, 1.0, 0.0], device=S.device).expand(B, -1))
    E = U @ S @ Vt

    return E

