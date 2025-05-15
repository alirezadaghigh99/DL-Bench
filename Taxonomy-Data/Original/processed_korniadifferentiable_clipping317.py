def differentiable_clipping(
    input: Tensor,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    scale: float = 0.02,
) -> Tensor:
    """This function implements a differentiable and soft approximation of the clipping operation.

    Args:
        input (Tensor): Input tensor of any shape.
        min_val (Optional[float]): Minimum value.
        max_val (Optional[float]): Maximum value.
        scale (float): Scale value. Default 0.02.

    Returns:
        output (Tensor): Clipped output tensor of the same shape as the input tensor.
    """
    # Make a copy of the input tensor
    output: Tensor = input.clone()
    # Perform differentiable soft clipping
    if max_val is not None:
        output[output > max_val] = -scale * (torch.exp(-output[output > max_val] + max_val) - 1.0) + max_val
    if min_val is not None:
        output[output < min_val] = scale * (torch.exp(output[output < min_val] - min_val) - 1.0) + min_val
    return output