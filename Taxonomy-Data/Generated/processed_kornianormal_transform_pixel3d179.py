import torch

def normal_transform_pixel3d(depth, height, width, eps=1e-6, device='cpu', dtype=torch.float32):
    """
    Computes the normalization matrix from image size in pixels to the range [-1, 1].

    Parameters:
    - depth (int): The depth of the image.
    - height (int): The height of the image.
    - width (int): The width of the image.
    - eps (float): A small epsilon value to prevent divide-by-zero errors.
    - device (str): The device on which to create the tensor ('cpu' or 'cuda').
    - dtype (torch.dtype): The data type of the tensor.

    Returns:
    - torch.Tensor: A normalized transform matrix with shape (1, 4, 4).
    """
    # Create a 4x4 identity matrix
    transform_matrix = torch.eye(4, device=device, dtype=dtype)

    # Calculate scale factors, adding epsilon to prevent division by zero
    scale_x = 2.0 / (width - 1 + eps)
    scale_y = 2.0 / (height - 1 + eps)
    scale_z = 2.0 / (depth - 1 + eps)

    # Set the scale factors in the matrix
    transform_matrix[0, 0] = scale_x
    transform_matrix[1, 1] = scale_y
    transform_matrix[2, 2] = scale_z

    # Set the translation factors to shift the range from [0, width/height/depth] to [-1, 1]
    transform_matrix[0, 3] = -1.0
    transform_matrix[1, 3] = -1.0
    transform_matrix[2, 3] = -1.0

    # Reshape to (1, 4, 4) to match the expected output shape
    transform_matrix = transform_matrix.unsqueeze(0)

    return transform_matrix

