import torch
from typing import Tuple

def acos_linear_extrapolation(x: torch.Tensor, bounds: Tuple[float, float]) -> torch.Tensor:
    # Validate the bounds
    lower_bound, upper_bound = bounds
    if not (-1 < lower_bound < 1) or not (-1 < upper_bound < 1):
        raise ValueError("Bounds must be within the range (-1, 1).")
    if lower_bound > upper_bound:
        raise ValueError("Lower bound must be less than or equal to the upper bound.")
    
    # Initialize the result tensor
    result = torch.empty_like(x)
    
    # Compute arccos for values within the bounds
    within_bounds = (x >= lower_bound) & (x <= upper_bound)
    result[within_bounds] = torch.acos(x[within_bounds])
    
    # Linear extrapolation for values outside the bounds
    # For x < lower_bound
    below_lower_bound = x < lower_bound
    if below_lower_bound.any():
        # Taylor approximation at lower_bound: acos(x) ≈ acos(lower_bound) - sqrt(1 - lower_bound^2) * (x - lower_bound)
        acos_lower = torch.acos(torch.tensor(lower_bound))
        slope_lower = -torch.sqrt(1 - lower_bound**2)
        result[below_lower_bound] = acos_lower + slope_lower * (x[below_lower_bound] - lower_bound)
    
    # For x > upper_bound
    above_upper_bound = x > upper_bound
    if above_upper_bound.any():
        # Taylor approximation at upper_bound: acos(x) ≈ acos(upper_bound) - sqrt(1 - upper_bound^2) * (x - upper_bound)
        acos_upper = torch.acos(torch.tensor(upper_bound))
        slope_upper = -torch.sqrt(1 - upper_bound**2)
        result[above_upper_bound] = acos_upper + slope_upper * (x[above_upper_bound] - upper_bound)
    
    return result

