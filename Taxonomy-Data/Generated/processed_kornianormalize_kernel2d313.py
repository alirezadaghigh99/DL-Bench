import torch
from torch import Tensor

def KORNIA_CHECK_SHAPE(tensor: Tensor, shape: list):
    # This is a placeholder for the actual shape checking function.
    # You should replace this with the actual implementation or import it if available.
    assert len(tensor.shape) == len(shape), "Input tensor does not match the required shape."
    for dim, s in zip(tensor.shape, shape):
        if s != "*" and dim != s:
            raise ValueError(f"Expected dimension {s} but got {dim}.")

def normalize_kernel2d(input: Tensor) -> Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    KORNIA_CHECK_SHAPE(input, ["*", "H", "W"])
    
    # Calculate the sum of the kernel elements
    kernel_sum = input.sum()
    
    # If the sum is zero, we cannot normalize by dividing by zero
    if kernel_sum == 0:
        raise ValueError("The sum of the kernel elements is zero, cannot normalize.")
    
    # Normalize the kernel by dividing each element by the sum
    normalized_kernel = input / kernel_sum
    
    return normalized_kernel

