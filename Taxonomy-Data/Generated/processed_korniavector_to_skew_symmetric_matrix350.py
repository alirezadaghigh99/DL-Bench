import torch

def vector_to_skew_symmetric_matrix(v):
    """
    Converts a 3D vector or a batch of 3D vectors into their corresponding skew-symmetric matrices.

    Parameters:
    v (torch.Tensor): A tensor of shape (B, 3) or (3,), representing a batch of 3D vectors or a single 3D vector.

    Returns:
    torch.Tensor: A tensor of shape (B, 3, 3) or (3, 3) containing the skew-symmetric matrices.

    Raises:
    ValueError: If the input tensor does not have the correct shape.
    """
    if v.ndim == 1:
        if v.shape[0] != 3:
            raise ValueError("Input vector must have shape (3,) for a single 3D vector.")
        v = v.unsqueeze(0)  # Add batch dimension for consistent processing

    elif v.ndim == 2:
        if v.shape[1] != 3:
            raise ValueError("Input tensor must have shape (B, 3) for a batch of 3D vectors.")
    else:
        raise ValueError("Input tensor must have shape (B, 3) or (3,).")

    # Extract components
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    # Create skew-symmetric matrices
    zero = torch.zeros_like(x)
    skew_matrices = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1)
    ], dim=-2)

    if skew_matrices.shape[0] == 1:
        return skew_matrices.squeeze(0)  # Remove batch dimension if input was a single vector
    return skew_matrices

