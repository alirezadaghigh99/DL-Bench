import torch

def create_meshgrid(height, width, normalized_coordinates=True, device='cpu', dtype=torch.float32):
    # Create a grid of coordinates
    y = torch.linspace(0, height - 1, steps=height, device=device, dtype=dtype)
    x = torch.linspace(0, width - 1, steps=width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Stack to create a grid of shape (H, W, 2)
    grid = torch.stack((xx, yy), dim=-1)
    
    if normalized_coordinates:
        # Normalize the coordinates to the range [-1, 1]
        grid[..., 0] = 2.0 * grid[..., 0] / (width - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (height - 1) - 1.0
    
    # Add a batch dimension to the grid
    grid = grid.unsqueeze(0)
    
    return grid

