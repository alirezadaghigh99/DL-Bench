def make_upright(laf: Tensor, eps: float = 1e-9) -> Tensor:
    """Rectify the affine matrix, so that it becomes upright.

    Args:
        laf: :math:`(B, N, 2, 3)`
        eps: for safe division.

    Returns:
        laf: :math:`(B, N, 2, 3)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = make_upright(input)  #  BxNx2x3
    """
    KORNIA_CHECK_LAF(laf)
    det = get_laf_scale(laf)
    scale = det
    # The function is equivalent to doing 2x2 SVD and resetting rotation
    # matrix to an identity: U, S, V = svd(LAF); LAF_upright = U * S.
    b2a2 = torch.sqrt(laf[..., 0:1, 1:2] ** 2 + laf[..., 0:1, 0:1] ** 2) + eps
    laf1_ell = concatenate([(b2a2 / det).contiguous(), torch.zeros_like(det)], dim=3)
    laf2_ell = concatenate(
        [
            ((laf[..., 1:2, 1:2] * laf[..., 0:1, 1:2] + laf[..., 1:2, 0:1] * laf[..., 0:1, 0:1]) / (b2a2 * det)),
            (det / b2a2).contiguous(),
        ],
        dim=3,
    )
    laf_unit_scale = concatenate([concatenate([laf1_ell, laf2_ell], dim=2), laf[..., :, 2:3]], dim=3)
    return scale_laf(laf_unit_scale, scale)