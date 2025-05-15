def remap(
    image: Tensor,
    map_x: Tensor,
    map_y: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
    normalized_coordinates: bool = False,
) -> Tensor:
    r"""Apply a generic geometrical transformation to an image tensor.

    .. image:: _static/img/remap.png

    The function remap transforms the source tensor using the specified map:

    .. math::
        \text{dst}(x, y) = \text{src}(map_x(x, y), map_y(x, y))

    Args:
        image: the tensor to remap with shape (B, C, H, W).
          Where C is the number of channels.
        map_x: the flow in the x-direction in pixel coordinates.
          The tensor must be in the shape of (B, H, W).
        map_y: the flow in the y-direction in pixel coordinates.
          The tensor must be in the shape of (B, H, W).
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: mode for grid_generation.
        normalized_coordinates: whether the input coordinates are
           normalized in the range of [-1, 1].

    Returns:
        the warped tensor with same shape as the input grid maps.

    Example:
        >>> import torch
        >>> from kornia.utils import create_meshgrid
        >>> grid = create_meshgrid(2, 2, False)  # 1x2x2x2
        >>> grid += 1  # apply offset in both directions
        >>> input = torch.ones(1, 1, 2, 2)
        >>> remap(input, grid[..., 0], grid[..., 1], align_corners=True)   # 1x1x2x2
        tensor([[[[1., 0.],
                  [0., 0.]]]])

    .. note::
        This function is often used in conjunction with :func:`kornia.utils.create_meshgrid`.
    """
    KORNIA_CHECK_SHAPE(image, ["B", "C", "H", "W"])
    KORNIA_CHECK_SHAPE(map_x, ["B", "H", "W"])
    KORNIA_CHECK_SHAPE(map_y, ["B", "H", "W"])

    batch_size, _, height, width = image.shape

    # grid_sample need the grid between -1/1
    map_xy: Tensor = stack([map_x, map_y], -1)

    # normalize coordinates if not already normalized
    if not normalized_coordinates:
        map_xy = normalize_pixel_coordinates(map_xy, height, width)

    # simulate broadcasting since grid_sample does not support it
    map_xy = map_xy.expand(batch_size, -1, -1, -1)

    # warp the image tensor and return
    return F.grid_sample(image, map_xy, mode=mode, padding_mode=padding_mode, align_corners=align_corners)