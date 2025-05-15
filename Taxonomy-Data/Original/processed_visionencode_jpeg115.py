def encode_jpeg(
    input: Union[torch.Tensor, List[torch.Tensor]], quality: int = 75
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Encode RGB tensor(s) into raw encoded jpeg bytes, on CPU or CUDA.

    .. note::
        Passing a list of CUDA tensors is more efficient than repeated individual calls to ``encode_jpeg``.
        For CPU tensors the performance is equivalent.

    Args:
        input (Tensor[channels, image_height, image_width] or List[Tensor[channels, image_height, image_width]]):
            (list of) uint8 image tensor(s) of ``c`` channels, where ``c`` must be 1 or 3
        quality (int): Quality of the resulting JPEG file(s). Must be a number between
            1 and 100. Default: 75

    Returns:
        output (Tensor[1] or list[Tensor[1]]): A (list of) one dimensional uint8 tensor(s) that contain the raw bytes of the JPEG file.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(encode_jpeg)
    if quality < 1 or quality > 100:
        raise ValueError("Image quality should be a positive number between 1 and 100")
    if isinstance(input, list):
        if not input:
            raise ValueError("encode_jpeg requires at least one input tensor when a list is passed")
        if input[0].device.type == "cuda":
            return torch.ops.image.encode_jpegs_cuda(input, quality)
        else:
            return [torch.ops.image.encode_jpeg(image, quality) for image in input]
    else:  # single input tensor
        if input.device.type == "cuda":
            return torch.ops.image.encode_jpegs_cuda([input], quality)[0]
        else:
            return torch.ops.image.encode_jpeg(input, quality)