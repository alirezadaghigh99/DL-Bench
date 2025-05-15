def yuv420_to_rgb(imagey: Tensor, imageuv: Tensor) -> Tensor:
    r"""Convert an YUV420 image to RGB.

    Input need to be padded to be evenly divisible by 2 horizontal and vertical.

    The image data is assumed to be in the range of :math:`(0, 1)` for luma (Y). The ranges of U and V are
    :math:`(-0.436, 0.436)` and :math:`(-0.615, 0.615)`, respectively.

    YUV formula follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Args:
        imagey: Y (luma) Image plane to be converted to RGB with shape :math:`(*, 1, H, W)`.
        imageuv: UV (chroma) Image planes to be converted to RGB with shape :math:`(*, 2, H/2, W/2)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 2, 3)
        >>> output = yuv420_to_rgb(inputy, inputuv)  # 2x3x4x6
    """
    if not isinstance(imagey, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(imagey)}")

    if not isinstance(imageuv, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(imageuv)}")

    if len(imagey.shape) < 3 or imagey.shape[-3] != 1:
        raise ValueError(f"Input imagey size must have a shape of (*, 1, H, W). Got {imagey.shape}")

    if len(imageuv.shape) < 3 or imageuv.shape[-3] != 2:
        raise ValueError(f"Input imageuv size must have a shape of (*, 2, H/2, W/2). Got {imageuv.shape}")

    if len(imagey.shape) < 2 or imagey.shape[-2] % 2 == 1 or imagey.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {imagey.shape}")

    if (
        len(imageuv.shape) < 2
        or len(imagey.shape) < 2
        or imagey.shape[-2] / imageuv.shape[-2] != 2
        or imagey.shape[-1] / imageuv.shape[-1] != 2
    ):
        raise ValueError(
            f"Input imageuv H&W must be half the size of the luma plane. Got {imagey.shape} and {imageuv.shape}"
        )

    # first upsample
    yuv444image = torch.cat(
        [imagey, imageuv.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)],
        dim=-3,
    )
    # then convert the yuv444 tensor

    return yuv_to_rgb(yuv444image)