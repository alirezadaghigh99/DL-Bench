def center_crop3d(
    tensor: torch.Tensor, size: Tuple[int, int, int], interpolation: str = "bilinear", align_corners: bool = True
) -> torch.Tensor:
    r"""Crop the 3D volumes (5D tensor) at the center.

    Args:
        tensor: the 3D volume tensor with shape (B, C, D, H, W).
        size: a tuple with the expected depth, height and width
            of the output patch.
        interpolation: Interpolation flag.
        align_corners : mode for grid_generation.

    Returns:
        the output tensor with patches.

    Examples:
        >>> input = torch.arange(64, dtype=torch.float32).view(1, 1, 4, 4, 4)
        >>> input
        tensor([[[[[ 0.,  1.,  2.,  3.],
                   [ 4.,  5.,  6.,  7.],
                   [ 8.,  9., 10., 11.],
                   [12., 13., 14., 15.]],
        <BLANKLINE>
                  [[16., 17., 18., 19.],
                   [20., 21., 22., 23.],
                   [24., 25., 26., 27.],
                   [28., 29., 30., 31.]],
        <BLANKLINE>
                  [[32., 33., 34., 35.],
                   [36., 37., 38., 39.],
                   [40., 41., 42., 43.],
                   [44., 45., 46., 47.]],
        <BLANKLINE>
                  [[48., 49., 50., 51.],
                   [52., 53., 54., 55.],
                   [56., 57., 58., 59.],
                   [60., 61., 62., 63.]]]]])
        >>> center_crop3d(input, (2, 2, 2), align_corners=True)
        tensor([[[[[21.0000, 22.0000],
                   [25.0000, 26.0000]],
        <BLANKLINE>
                  [[37.0000, 38.0000],
                   [41.0000, 42.0000]]]]])
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input tensor type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 5:
        raise AssertionError(f"Only tensor with shape (B, C, D, H, W) supported. Got {tensor.shape}.")

    if not isinstance(size, (tuple, list)) and len(size) == 3:
        raise ValueError(f"Input size must be a tuple/list of length 3. Got {size}")

    # unpack input sizes
    dst_d, dst_h, dst_w = size
    src_d, src_h, src_w = tensor.shape[-3:]

    # compute start/end offsets
    dst_d_half = dst_d / 2
    dst_h_half = dst_h / 2
    dst_w_half = dst_w / 2
    src_d_half = src_d / 2
    src_h_half = src_h / 2
    src_w_half = src_w / 2

    start_x = src_w_half - dst_w_half
    start_y = src_h_half - dst_h_half
    start_z = src_d_half - dst_d_half

    end_x = start_x + dst_w - 1
    end_y = start_y + dst_h - 1
    end_z = start_z + dst_d - 1
    # [x, y, z] origin
    # top-left-front, top-right-front, bottom-right-front, bottom-left-front
    # top-left-back, top-right-back, bottom-right-back, bottom-left-back
    points_src: torch.Tensor = torch.tensor(
        [
            [
                [start_x, start_y, start_z],
                [end_x, start_y, start_z],
                [end_x, end_y, start_z],
                [start_x, end_y, start_z],
                [start_x, start_y, end_z],
                [end_x, start_y, end_z],
                [end_x, end_y, end_z],
                [start_x, end_y, end_z],
            ]
        ],
        device=tensor.device,
    )

    # [x, y, z] destination
    # top-left-front, top-right-front, bottom-right-front, bottom-left-front
    # top-left-back, top-right-back, bottom-right-back, bottom-left-back
    points_dst: torch.Tensor = torch.tensor(
        [
            [
                [0, 0, 0],
                [dst_w - 1, 0, 0],
                [dst_w - 1, dst_h - 1, 0],
                [0, dst_h - 1, 0],
                [0, 0, dst_d - 1],
                [dst_w - 1, 0, dst_d - 1],
                [dst_w - 1, dst_h - 1, dst_d - 1],
                [0, dst_h - 1, dst_d - 1],
            ]
        ],
        device=tensor.device,
    ).expand(points_src.shape[0], -1, -1)

    return crop_by_boxes3d(
        tensor, points_src.to(tensor.dtype), points_dst.to(tensor.dtype), interpolation, align_corners
    )