import math
from typing import Tuple
import torch
DEFAULT_ACOS_BOUND: float = 1.0 - 0.0001

def acos_linear_extrapolation(x: torch.Tensor, bounds: Tuple[float, float]=(-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND)) -> torch.Tensor:
    """Create a Python function named acos_linear_extrapolation that computes the arccosine of a tensor, with linear extrapolation applied outside the domain 
(
−
1
,
1
)
(−1,1) for stable backpropagation. The function should take a tensor x and a tuple bounds representing the lower and upper bounds for the extrapolation region. If the value of x is within the bounds, the function should return the standard arccos(x) value. If x is outside the bounds, it should apply a first-order Taylor approximation for extrapolation. The function should validate that the bounds are within the valid range 
(
−
1
,
1
)
(−1,1) and that the lower bound is less than or equal to the upper bound. The function returns a tensor containing the computed values.

Inputs:

x (torch.Tensor): The input tensor for which to compute the arccosine.
bounds (Tuple[float, float]): A tuple containing the lower and upper bounds for the linear extrapolation.
Outputs:

Returns a tensor containing the extrapolated arccos(x) values.
Error Handling:

Raise a ValueError if the bounds are outside the range 
(
−
1
,
1
)
(−1,1) or if the lower bound is greater than the upper bound."""
    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError('lower bound has to be smaller or equal to upper bound')

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError('Both lower bound and upper bound have to be within (-1, 1)')

    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    acos_extrap[x_mid] = torch.acos(x[x_mid])
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)

    return acos_extrap

def _acos_linear_approximation(x: torch.Tensor, x0: float) -> torch.Tensor:
    """
    Calculates the 1st order Taylor expansion of `arccos(x)` around `x0`.
    """
    return (x - x0) * _dacos_dx(x0) + math.acos(x0)

def _dacos_dx(x: float) -> float:
    """
    Calculates the derivative of `arccos(x)` w.r.t. `x`.
    """
    return -1.0 / math.sqrt(1.0 - x * x)
