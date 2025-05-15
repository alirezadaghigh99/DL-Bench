import torch
from torch import Tensor

def deg2rad(tensor: Tensor) -> Tensor:
    """
    Converts angles from degrees to radians.

    Args:
        tensor: Tensor of arbitrary shape.

    Returns:
        Tensor with same shape as input, with angles converted to radians.

    Raises:
        TypeError: If the input is not a Tensor.
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(tensor)}")
    
    # Conversion factor from degrees to radians
    conversion_factor = torch.tensor(3.141592653589793 / 180.0)
    
    # Convert degrees to radians
    return tensor * conversion_factor

