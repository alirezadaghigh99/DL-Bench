def deg2rad(tensor: Tensor) -> Tensor:
    r"""Function that converts angles from degrees to radians.

    Args:
        tensor: Tensor of arbitrary shape.

    Returns:
        tensor with same shape as input.

    Examples:
        >>> input = tensor(180.)
        >>> deg2rad(input)
        tensor(3.1416)
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(tensor)}")

    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.0