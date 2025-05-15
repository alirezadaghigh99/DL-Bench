def laf_from_center_scale_ori(xy: Tensor, scale: Optional[Tensor] = None, ori: Optional[Tensor] = None) -> Tensor:
    """Creates a LAF from keypoint center, scale and orientation.

    Useful to create kornia LAFs from OpenCV keypoints.

    Args:
        xy: :math:`(B, N, 2)`.
        scale: :math:`(B, N, 1, 1)`. If not provided, scale = 1.0 is assumed
        angle in degrees: :math:`(B, N, 1)`. If not provided orientation = 0 is assumed

    Returns:
        LAF :math:`(B, N, 2, 3)`
    """
    KORNIA_CHECK_SHAPE(xy, ["B", "N", "2"])
    device = xy.device
    dtype = xy.dtype
    B, N = xy.shape[:2]
    if scale is None:
        scale = torch.ones(B, N, 1, 1, device=device, dtype=dtype)
    if ori is None:
        ori = zeros(B, N, 1, device=device, dtype=dtype)
    KORNIA_CHECK_SHAPE(scale, ["B", "N", "1", "1"])
    KORNIA_CHECK_SHAPE(ori, ["B", "N", "1"])
    unscaled_laf = concatenate([angle_to_rotation_matrix(ori.squeeze(-1)), xy.unsqueeze(-1)], dim=-1)
    laf = scale_laf(unscaled_laf, scale)
    return laf