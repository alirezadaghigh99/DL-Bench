def _jpeg_decode(
    input_y: Tensor,
    input_cb: Tensor,
    input_cr: Tensor,
    jpeg_quality: Tensor,
    H: int,
    W: int,
    quantization_table_y: Tensor,
    quantization_table_c: Tensor,
) -> Tensor:
    """Performs JPEG decoding.

    Args:
        input_y (Tensor): Compressed Y component of the shape :math:`(B, N, 8, 8)`.
        input_cb (Tensor): Compressed Cb component of the shape :math:`(B, N, 8, 8)`.
        input_cr (Tensor): Compressed Cr component of the shape :math:`(B, N, 8, 8)`.
        jpeg_quality (Tensor): Compression strength of the shape :math:`(B)`.
        H (int): Original image height.
        W (int): Original image width.
        quantization_table_y (Tensor): Quantization table for Y channel.
        quantization_table_c (Tensor): Quantization table for C channels.

    Returns:
        rgb_decoded (Tensor): Decompressed RGB image of the shape :math:`(B, 3, H, W)`.
    """
    # Dequantize inputs
    input_y = _dequantize(
        input_y,
        jpeg_quality,
        quantization_table_y,
    )
    input_cb_cr = _dequantize(
        torch.cat((input_cb, input_cr), dim=1),
        jpeg_quality,
        quantization_table_c,
    )
    # Perform inverse DCT
    idct_y: Tensor = _idct_8x8(input_y)
    idct_cb, idct_cr = _idct_8x8(input_cb_cr).chunk(2, dim=1)
    # Reverse patching
    image_y: Tensor = _unpatchify_8x8(idct_y, H, W)
    image_cb: Tensor = _unpatchify_8x8(idct_cb, H // 2, W // 2)
    image_cr: Tensor = _unpatchify_8x8(idct_cr, H // 2, W // 2)
    # Perform chroma upsampling
    image_cb = _chroma_upsampling(image_cb)
    image_cr = _chroma_upsampling(image_cr)
    # Back to [0, 1] pixel-range
    image_ycbcr: Tensor = torch.stack((image_y, image_cb, image_cr), dim=1) / 255.0
    # Convert back to RGB space.
    rgb_decoded: Tensor = ycbcr_to_rgb(image_ycbcr)
    return rgb_decoded