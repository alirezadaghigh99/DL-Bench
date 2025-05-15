def affine(
    img: Image.Image,
    matrix: List[float],
    interpolation: int = Image.NEAREST,
    fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    output_size = img.size
    opts = _parse_fill(fill, img)
    return img.transform(output_size, Image.AFFINE, matrix, interpolation, **opts)