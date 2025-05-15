def vector_to_skew_symmetric_matrix(vec: Tensor) -> Tensor:
    r"""Converts a vector to a skew symmetric matrix.

    A vector :math:`(v1, v2, v3)` has a corresponding skew-symmetric matrix, which is of the form:

    .. math::
        \begin{bmatrix} 0 & -v3 & v2 \\
        v3 & 0 & -v1 \\
        -v2 & v1 & 0\end{bmatrix}

    Args:
        x: tensor of shape :math:`(B, 3)`.

    Returns:
        tensor of shape :math:`(B, 3, 3)`.

    Example:
        >>> vec = torch.tensor([1.0, 2.0, 3.0])
        >>> vector_to_skew_symmetric_matrix(vec)
        tensor([[ 0., -3.,  2.],
                [ 3.,  0., -1.],
                [-2.,  1.,  0.]])
    """
    # KORNIA_CHECK_SHAPE(vec, ["B", "3"])
    if vec.shape[-1] != 3 or len(vec.shape) > 2:
        raise ValueError(f"Input vector must be of shape (B, 3) or (3,). Got {vec.shape}")
    v1, v2, v3 = vec[..., 0], vec[..., 1], vec[..., 2]
    zeros = zeros_like(v1)
    skew_symmetric_matrix = stack(
        [stack([zeros, -v3, v2], dim=-1), stack([v3, zeros, -v1], dim=-1), stack([-v2, v1, zeros], dim=-1)], dim=-2
    )
    return skew_symmetric_matrix