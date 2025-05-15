def draw_point2d(image: Tensor, points: Tensor, color: Tensor) -> Tensor:
    r"""Sets one or more coordinates in a Tensor to a color.

    Args:
        image: the input image on which to draw the points with shape :math`(C,H,W)` or :math`(H,W)`.
        points: the [x, y] points to be drawn on the image.
        color: the color of the pixel with :math`(C)` where :math`C` is the number of channels of the image.

    Return:
        The image with points set to the color.
    """
    KORNIA_CHECK(
        (len(image.shape) == 2 and len(color.shape) == 1) or (image.shape[0] == color.shape[0]),
        "Color dim must match the channel dims of the provided image",
    )
    points = points.to(dtype=torch.int64, device=image.device)
    x, y = zip(*points)
    if len(color.shape) == 1:
        color = torch.unsqueeze(color, dim=1)
    color = color.to(dtype=image.dtype, device=image.device)
    if len(image.shape) == 2:
        image[y, x] = color
    else:
        image[:, y, x] = color
    return image