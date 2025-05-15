def rgb_to_yuv420(image: Tensor) -> tuple[Tensor, Tensor]:
    r"""Convert an RGB image to YUV 420 (subsampled).

    Input need to be padded to be evenly divisible by 2 horizontal and vertical.

    The image data is assumed to be in the range of :math:`(0, 1)`. The range of the output is of :math:`(0, 1)` to
    luma and the ranges of U and V are :math:`(-0.436, 0.436)` and :math:`(-0.615, 0.615)`, respectively.

    The YUV model adopted here follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        A Tensor containing the Y plane with shape :math:`(*, 1, H, W)`
        A Tensor containing the UV planes with shape :math:`(*, 2, H/2, W/2)`

    Example:
        >>> input = torch.rand(2, 3, 4, 6)
        >>> output = rgb_to_yuv420(input)  # (2x1x4x6, 2x2x2x3)
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if len(image.shape) < 2 or image.shape[-2] % 2 == 1 or image.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {image.shape}")

    yuvimage = rgb_to_yuv(image)

    return (
        yuvimage[..., :1, :, :],
        yuvimage[..., 1:3, :, :].unfold(-2, 2, 2).unfold(-2, 2, 2).mean((-1, -2)),
    )