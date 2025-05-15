def _perform_padding(image: Tensor) -> tuple[Tensor, int, int]:
    """Pads a given image to be dividable by 16.

    Args:
        image: Image of the shape :math:`(*, 3, H, W)`.

    Returns:
        image_padded: Padded image of the shape :math:`(*, 3, H_{new}, W_{new})`.
        h_pad: Padded pixels along the horizontal axis.
        w_pad: Padded pixels along the vertical axis.
    """
    # Get spatial dimensions of the image
    H, W = image.shape[-2:]
    # Compute horizontal and vertical padding
    h_pad: int = math.ceil(H / 16) * 16 - H
    w_pad: int = math.ceil(W / 16) * 16 - W
    # Perform padding (we follow JPEG and pad only the bottom and right side of the image)
    image_padded: Tensor = F.pad(image, (0, w_pad, 0, h_pad), "replicate")
    return image_padded, h_pad, w_pad