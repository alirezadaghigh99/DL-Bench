import torch
import torch.nn.functional as F

def resize(input_tensor, output_size, interpolation='bilinear', align_corners=False, side='both', antialias=False):
    """
    Resizes an input torch.Tensor to a specified size using various parameters.

    Parameters:
    - input_tensor (torch.Tensor): The input tensor to resize.
    - output_size (tuple): The desired output size (height, width).
    - interpolation (str): The interpolation method ('bilinear', 'nearest', etc.).
    - align_corners (bool): If True, aligns the corners of the input and output tensors.
    - side (str): The side to resize ('both', 'height', 'width').
    - antialias (bool): If True, applies antialiasing.

    Returns:
    - torch.Tensor: The resized tensor.
    """
    # Validate input_tensor
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("input_tensor must be a torch.Tensor")

    # Validate output_size
    if not (isinstance(output_size, tuple) and len(output_size) == 2):
        raise ValueError("output_size must be a tuple of two integers (height, width)")

    # Validate interpolation method
    valid_interpolations = ['bilinear', 'nearest', 'bicubic', 'trilinear', 'area']
    if interpolation not in valid_interpolations:
        raise ValueError(f"interpolation must be one of {valid_interpolations}")

    # Determine the size to resize based on the side parameter
    if side == 'both':
        size = output_size
    elif side == 'height':
        size = (output_size[0], input_tensor.shape[-1])
    elif side == 'width':
        size = (input_tensor.shape[-2], output_size[1])
    else:
        raise ValueError("side must be 'both', 'height', or 'width'")

    # Perform the resizing
    resized_tensor = F.interpolate(
        input_tensor,
        size=size,
        mode=interpolation,
        align_corners=align_corners,
        recompute_scale_factor=antialias
    )

    return resized_tensor

