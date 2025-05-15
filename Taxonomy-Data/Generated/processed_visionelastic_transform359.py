import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

def elastic_transform(img, displacement, interpolation=InterpolationMode.BILINEAR, fill=None):
    """
    Apply elastic transformation to a tensor image.

    Parameters:
    - img: PIL Image or Tensor
    - displacement: Tensor of shape (2, H, W) containing displacement vectors
    - interpolation: InterpolationMode (default: InterpolationMode.BILINEAR)
    - fill: Optional list of floats for filling the border areas

    Returns:
    - Transformed tensor image
    """
    if isinstance(img, Image.Image):
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    if not isinstance(img, torch.Tensor):
        raise TypeError("img should be a PIL Image or a Tensor")

    if img.dim() == 3:
        img = img.unsqueeze(0)  # Add batch dimension

    _, _, height, width = img.shape

    # Create meshgrid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, height, dtype=torch.float32),
                                    torch.arange(0, width, dtype=torch.float32))
    grid = torch.stack((grid_x, grid_y), 0)  # Shape: (2, H, W)

    # Add displacement to the grid
    displacement = displacement.to(img.device)
    grid = grid + displacement

    # Normalize grid to [-1, 1] for grid_sample
    grid = grid.permute(1, 2, 0)  # Shape: (H, W, 2)
    grid[..., 0] = 2.0 * grid[..., 0] / (width - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (height - 1) - 1.0

    # Apply grid sampling
    mode = 'bilinear' if interpolation == InterpolationMode.BILINEAR else 'nearest'
    padding_mode = 'border' if fill is None else 'zeros'
    transformed_img = F.grid_sample(img, grid.unsqueeze(0), mode=mode, padding_mode=padding_mode, align_corners=True)

    if fill is not None:
        fill_tensor = torch.tensor(fill, dtype=img.dtype, device=img.device).view(1, -1, 1, 1)
        mask = (grid[..., 0] < -1) | (grid[..., 0] > 1) | (grid[..., 1] < -1) | (grid[..., 1] > 1)
        transformed_img = torch.where(mask.unsqueeze(0).unsqueeze(0), fill_tensor, transformed_img)

    return transformed_img.squeeze(0)

