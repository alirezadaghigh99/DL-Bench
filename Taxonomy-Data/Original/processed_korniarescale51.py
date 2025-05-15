def rescale(
    input: Tensor,
    factor: Union[float, Tuple[float, float]],
    interpolation: str = "bilinear",
    align_corners: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:
    r"""Rescale the input Tensor with the given factor.

    .. image:: _static/img/rescale.png

    Args:
        input: The image tensor to be scale with shape of :math:`(B, C, H, W)`.
        factor: Desired scaling factor in each direction. If scalar, the value is used
            for both the x- and y-direction.
        interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'linear'`` | ``'bilinear'`` |
            ``'bicubic'`` | ``'trilinear'`` | ``'area'``.
        align_corners: interpolation flag.
        side: Corresponding side if ``size`` is an integer. Can be one of ``'short'``, ``'long'``, ``'vert'``,
            or ``'horz'``.
        antialias: if True, then image will be filtered with Gaussian before downscaling.
            No effect for upscaling.

    Returns:
        The rescaled tensor with the shape as the specified size.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> out = rescale(img, (2, 3))
        >>> print(out.shape)
        torch.Size([1, 3, 8, 12])
    """
    if isinstance(factor, float):
        factor_vert = factor_horz = factor
    else:
        factor_vert, factor_horz = factor

    height, width = input.size()[-2:]
    size = (int(height * factor_vert), int(width * factor_horz))
    return resize(input, size, interpolation=interpolation, align_corners=align_corners, antialias=antialias)