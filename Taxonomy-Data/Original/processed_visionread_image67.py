def read_image(
    path: str,
    mode: ImageReadMode = ImageReadMode.UNCHANGED,
    apply_exif_orientation: bool = False,
) -> torch.Tensor:
    """[OBSOLETE] Use :func:`~torchvision.io.decode_image` instead."""
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(read_image)
    data = read_file(path)
    return decode_image(data, mode, apply_exif_orientation=apply_exif_orientation)