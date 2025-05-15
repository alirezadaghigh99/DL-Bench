import torch

def normal_transform_pixel(height, width, eps=1e-14, device=None, dtype=None):
    """
    Computes the normalization matrix to transform image coordinates from pixel space to [-1, 1].

    Parameters:
    - height (int): The height of the image in pixels.
    - width (int): The width of the image in pixels.
    - eps (float): A small epsilon value to prevent division by zero.
    - device (torch.device, optional): The device on which to create the tensor.
    - dtype (torch.dtype, optional): The data type of the tensor.

    Returns:
    - torch.Tensor: A tensor of shape (1, 3, 3) representing the normalization matrix.
    """
    # Calculate scale factors
    scale_x = 2.0 / (width - 1 + eps)
    scale_y = 2.0 / (height - 1 + eps)

    # Create the normalization matrix
    transform_matrix = torch.tensor([
        [scale_x, 0, -1],
        [0, scale_y, -1],
        [0, 0, 1]
    ], device=device, dtype=dtype)

    # Add an additional dimension at the beginning
    transform_matrix = transform_matrix.unsqueeze(0)  # Shape: (1, 3, 3)

    return transform_matrix

