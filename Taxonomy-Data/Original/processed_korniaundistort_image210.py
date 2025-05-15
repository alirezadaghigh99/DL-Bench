def undistort_image(image: torch.Tensor, K: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    r"""Compensate an image for lens distortion.

    Radial :math:`(k_1, k_2, k_3, k_4, k_4, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\tau_x, \tau_y)`
    distortion models are considered in this function.

    Args:
        image: Input image with shape :math:`(*, C, H, W)`.
        K: Intrinsic camera matrix with shape :math:`(*, 3, 3)`.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape :math:`(*, n)`.

    Returns:
        Undistorted image with shape :math:`(*, C, H, W)`.

    Example:
        >>> img = torch.rand(1, 3, 5, 5)
        >>> K = torch.eye(3)[None]
        >>> dist_coeff = torch.rand(1, 4)
        >>> out = undistort_image(img, K, dist_coeff)
        >>> out.shape
        torch.Size([1, 3, 5, 5])
    """
    if len(image.shape) < 3:
        raise ValueError(f"Image shape is invalid. Got: {image.shape}.")

    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K matrix shape is invalid. Got {K.shape}.")

    if dist.shape[-1] not in [4, 5, 8, 12, 14]:
        raise ValueError(f"Invalid number of distortion coefficients. Got {dist.shape[-1]}.")

    if not image.is_floating_point():
        raise ValueError(f"Invalid input image data type. Input should be float. Got {image.dtype}.")

    if image.shape[:-3] != K.shape[:-2] or image.shape[:-3] != dist.shape[:-1]:
        # Input with image shape (1, C, H, W), K shape (3, 3), dist shape (4)
        # allowed to avoid a breaking change.
        if not all((image.shape[:-3] == (1,), K.shape[:-2] == (), dist.shape[:-1] == ())):
            raise ValueError(
                "Input shape is invalid. Input batch dimensions should match. "
                f"Got {image.shape[:-3]}, {K.shape[:-2]}, {dist.shape[:-1]}."
            )

    channels, rows, cols = image.shape[-3:]
    B = image.numel() // (channels * rows * cols)

    # Create point coordinates for each pixel of the image
    xy_grid: torch.Tensor = create_meshgrid(rows, cols, False, image.device, image.dtype)
    pts = xy_grid.reshape(-1, 2)  # (rows*cols)x2 matrix of pixel coordinates

    # Distort points and define maps
    ptsd: torch.Tensor = distort_points(pts, K, dist)  # Bx(rows*cols)x2
    mapx: torch.Tensor = ptsd[..., 0].reshape(B, rows, cols)  # B x rows x cols, float
    mapy: torch.Tensor = ptsd[..., 1].reshape(B, rows, cols)  # B x rows x cols, float

    # Remap image to undistort
    out = remap(image.reshape(B, channels, rows, cols), mapx, mapy, align_corners=True)

    return out.view_as(image)