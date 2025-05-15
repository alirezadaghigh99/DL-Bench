import torch
from torchvision.utils import make_grid as tv_make_grid

def make_grid(tensor, nrow=8, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0):
    """
    Create a grid of images from a 4D mini-batch of images.

    Parameters:
    - tensor (torch.Tensor or list of torch.Tensors): A 4D mini-batch of images.
    - nrow (int): Number of images per row in the grid.
    - padding (int): Amount of padding between images.
    - normalize (bool): Whether to shift the image to the range (0, 1).
    - value_range (tuple): Tuple (min, max) for normalization.
    - scale_each (bool): Whether to scale each image in the batch separately.
    - pad_value (float): Value for padded pixels.

    Returns:
    - grid (torch.Tensor): A tensor containing the grid of images.
    """
    if isinstance(tensor, list):
        # If a list of tensors is provided, concatenate them along the batch dimension
        tensor = torch.cat(tensor, dim=0)

    # Ensure the input is a 4D tensor
    if tensor.dim() != 4:
        raise ValueError("Input tensor should be a 4D mini-batch of images")

    # Create the grid using torchvision's make_grid function
    grid = tv_make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize,
                        range=value_range, scale_each=scale_each, pad_value=pad_value)
    
    return grid