from typing import Optional
import torch
from kornia.core import Tensor, stack
from kornia.utils._compat import torch_meshgrid

def create_meshgrid(height: int, width: int, normalized_coordinates: bool=True, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None) -> Tensor:
    """Generate a Python function called create_meshgrid that generates a coordinate grid for an image. The function takes in parameters such as height, width, normalized_coordinates (defaulted to True), device, and dtype. It returns a grid tensor with shape (1, H, W, 2) where H is the image height and W is the image width. The function normalizes coordinates to be in the range [-1,1] if normalized_coordinates is set to True. The output is a tensor representing the coordinate grid for the image. Return:
        grid tensor with shape :math:`(1, H, W, 2)`.

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])

        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])"""
    xs: Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    base_grid = stack(torch_meshgrid([xs, ys], indexing='ij'), dim=-1)
    return base_grid.permute(1, 0, 2).unsqueeze(0)

def create_meshgrid3d(depth: int, height: int, width: int, normalized_coordinates: bool=True, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None) -> Tensor:
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        depth: the image depth (channels).
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, D, H, W, 3)`.
    """
    xs: Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    zs: Tensor = torch.linspace(0, depth - 1, depth, device=device, dtype=dtype)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
        zs = (zs / (depth - 1) - 0.5) * 2
    base_grid = stack(torch_meshgrid([zs, xs, ys], indexing='ij'), dim=-1)
    return base_grid.permute(0, 2, 1, 3).unsqueeze(0)
