import torch

def projections_from_fundamental(F_mat):
    """
    Get the projection matrices from the Fundamental Matrix.

    Args:
        F_mat: the fundamental matrix with the shape (B, 3, 3). -> Tensor

    Returns:
        The projection matrices with shape (B, 3, 4, 2). -> Tensor
    """
    if len(F_mat.shape) != 3:
        raise AssertionError(F_mat.shape)
    if F_mat.shape[-2:] != (3, 3):
        raise AssertionError(F_mat.shape)

    batch_size = F_mat.shape[0]

    # Initialize the first projection matrix P1
    P1 = torch.zeros((batch_size, 3, 4), dtype=F_mat.dtype, device=F_mat.device)
    P1[:, :3, :3] = torch.eye(3, dtype=F_mat.dtype, device=F_mat.device).unsqueeze(0).repeat(batch_size, 1, 1)

    # Compute the epipole in the second image
    U, S, Vt = torch.svd(F_mat)
    e2 = Vt[:, -1, :]  # The epipole is the last column of V (or row of Vt)

    # Skew-symmetric matrix for e2
    e2_skew = torch.zeros((batch_size, 3, 3), dtype=F_mat.dtype, device=F_mat.device)
    e2_skew[:, 0, 1] = -e2[:, 2]
    e2_skew[:, 0, 2] = e2[:, 1]
    e2_skew[:, 1, 0] = e2[:, 2]
    e2_skew[:, 1, 2] = -e2[:, 0]
    e2_skew[:, 2, 0] = -e2[:, 1]
    e2_skew[:, 2, 1] = e2[:, 0]

    # Compute the second projection matrix P2
    P2 = torch.zeros((batch_size, 3, 4), dtype=F_mat.dtype, device=F_mat.device)
    P2[:, :, :3] = e2_skew.bmm(F_mat)
    P2[:, :, 3] = e2

    # Stack the projection matrices
    P = torch.stack((P1, P2), dim=-1)

    return P