def projections_from_fundamental(F_mat: Tensor) -> Tensor:
    r"""Get the projection matrices from the Fundamental Matrix.

    Args:
       F_mat: the fundamental matrix with the shape :math:`(B, 3, 3)`.

    Returns:
        The projection matrices with shape :math:`(B, 3, 4, 2)`.
    """
    KORNIA_CHECK_SHAPE(F_mat, ["*", "3", "3"])

    R1 = eye_like(3, F_mat)  # Bx3x3
    t1 = vec_like(3, F_mat)  # Bx3

    Ft_mat = F_mat.transpose(-2, -1)

    _, e2 = _nullspace(Ft_mat)

    R2 = cross_product_matrix(e2) @ F_mat  # Bx3x3
    t2 = e2[..., :, None]  # Bx3x1

    P1 = torch.cat([R1, t1], dim=-1)  # Bx3x4
    P2 = torch.cat([R2, t2], dim=-1)  # Bx3x4

    return stack([P1, P2], dim=-1)