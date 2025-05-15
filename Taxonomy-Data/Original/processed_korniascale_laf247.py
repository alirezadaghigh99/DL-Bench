def scale_laf(laf: Tensor, scale_coef: Union[float, Tensor]) -> Tensor:
    """Multiplies region part of LAF ([:, :, :2, :2]) by a scale_coefficient.

    So the center, shape and orientation of the local feature stays the same, but the region area changes.

    Args:
        LAF :math:`(B, N, 2, 3)`
        scale_coef: broadcastable tensor or float.

    Returns:
        LAF :math:`(B, N, 2, 3)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> scale = 0.5
        >>> output = scale_laf(input, scale)  # BxNx2x3
    """
    if not isinstance(scale_coef, (float, Tensor)):
        raise TypeError(f"scale_coef should be float or Tensor. Got {type(scale_coef)}")
    KORNIA_CHECK_LAF(laf)
    centerless_laf = laf[:, :, :2, :2]
    return concatenate([scale_coef * centerless_laf, laf[:, :, :, 2:]], dim=3)