import torch

def _create_identity_grid(size):
    """
    Generates a grid of normalized coordinates for a given image size.

    Parameters:
    size (list of int): A list containing the dimensions of the grid (height, width).

    Returns:
    torch.Tensor: A tensor containing the grid coordinates, normalized to [-1, 1].
    """
    height, width = size
    # Create a grid of coordinates
    y_coords = torch.linspace(-1, 1, steps=height)
    x_coords = torch.linspace(-1, 1, steps=width)
    
    # Create a meshgrid from the coordinates
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Stack the grid to create a (height, width, 2) tensor
    grid = torch.stack((grid_x, grid_y), dim=-1)
    
    return grid

