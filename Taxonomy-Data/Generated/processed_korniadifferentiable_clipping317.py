import torch
import torch.nn.functional as F

def differentiable_clipping(input, min_val=None, max_val=None, scale=0.02):
    """
    Implements a differentiable and soft approximation of the clipping operation.

    Args:
        input (Tensor): Input tensor of any shape.
        min_val (Optional[float]): Minimum value.
        max_val (Optional[float]): Maximum value.
        scale (float): Scale value. Default 0.02.

    Returns:
        output (Tensor): Clipped output tensor of the same shape as the input tensor.
    """
    if min_val is not None:
        # Apply soft clipping for the lower bound
        input = min_val + F.softplus((input - min_val) / scale) * scale

    if max_val is not None:
        # Apply soft clipping for the upper bound
        input = max_val - F.softplus((max_val - input) / scale) * scale

    return input

