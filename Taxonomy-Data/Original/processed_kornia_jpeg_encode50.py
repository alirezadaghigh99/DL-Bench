def _jpeg_encode(
    image_rgb: Tensor,
    jpeg_quality: Tensor,
    quantization_table_y: Tensor,
    quantization_table_c: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Performs JPEG encoding.

    Args:
        image_rgb (Tensor): RGB input images of the shape :math:`(B, 3, H, W)`.
        jpeg_quality (Tensor): Compression strength of the shape :math:`(B)`.
        quantization_table_y (Tensor): Quantization table for Y channel.
        quantization_table_c (Tensor): Quantization table for C channels.

    Returns:
        y_encoded (Tensor): Encoded Y component of the shape :math:`(B, N, 8, 8)`.
        cb_encoded (Tensor): Encoded Cb component of the shape :math:`(B, N, 8, 8)`.
        cr_encoded (Tensor): Encoded Cr component of the shape :math:`(B, N, 8, 8)`.
    """
    # Convert RGB image to YCbCr.
    image_ycbcr: Tensor = rgb_to_ycbcr(image_rgb)
    # Scale pixel-range to [0, 255]
    image_ycbcr = 255.0 * image_ycbcr
    # Perform chroma subsampling
    input_y, input_cb, input_cr = _chroma_subsampling(image_ycbcr)
    # Patchify, DCT, and rounding
    input_y, input_cb, input_cr = (
        _patchify_8x8(input_y),
        _patchify_8x8(input_cb),
        _patchify_8x8(input_cr),
    )
    dct_y = _dct_8x8(input_y)
    dct_cb_cr = _dct_8x8(torch.cat((input_cb, input_cr), dim=1))
    y_encoded: Tensor = _quantize(
        dct_y,
        jpeg_quality,
        quantization_table_y,
    )
    cb_encoded, cr_encoded = _quantize(
        dct_cb_cr,
        jpeg_quality,
        quantization_table_c,
    ).chunk(2, dim=1)
    return y_encoded, cb_encoded, cr_encoded