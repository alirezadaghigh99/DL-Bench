from typing import Optional
import torch
from kornia.core import Tensor, eye, zeros

def eye_like(n: int, input: Tensor, shared_memory: bool=False) -> Tensor:
    """Return a 2-D tensor with ones on the diagonal and zeros elsewhere with the same batch size as the input.

    Args:
        n: the number of rows :math:`(N)`.
        input: image tensor that will determine the batch size of the output matrix.
          The expected shape is :math:`(B, *)`.
        shared_memory: when set, all samples in the batch will share the same memory.

    Returns:
       The identity matrix with the same batch size as the input :math:`(B, N, N)`.

    Notes:
        When the dimension to expand is of size 1, using torch.expand(...) yields the same tensor as torch.repeat(...)
        without using extra memory. Thus, when the tensor obtained by this method will be later assigned -
        use this method with shared_memory=False, otherwise, prefer using it with shared_memory=True.
    """
    if n <= 0:
        raise AssertionError(type(n), n)
    if len(input.shape) < 1:
        raise AssertionError(input.shape)
    identity = eye(n, device=input.device).type(input.dtype)
    return identity[None].expand(input.shape[0], n, n) if shared_memory else identity[None].repeat(input.shape[0], 1, 1)

def vec_like(n: int, tensor: Tensor, shared_memory: bool=False) -> Tensor:
    """Return a 2-D tensor with a vector containing zeros with the same batch size as the input.

    Args:
        n: the number of rows :math:`(N)`.
        tensor: image tensor that will determine the batch size of the output matrix.
          The expected shape is :math:`(B, *)`.
        shared_memory: when set, all samples in the batch will share the same memory.

    Returns:
        The vector with the same batch size as the input :math:`(B, N, 1)`.

    Notes:
        When the dimension to expand is of size 1, using torch.expand(...) yields the same tensor as torch.repeat(...)
        without using extra memory. Thus, when the tensor obtained by this method will be later assigned -
        use this method with shared_memory=False, otherwise, prefer using it with shared_memory=True.
    """
    if n <= 0:
        raise AssertionError(type(n), n)
    if len(tensor.shape) < 1:
        raise AssertionError(tensor.shape)
    vec = zeros(n, 1, device=tensor.device, dtype=tensor.dtype)
    return vec[None].expand(tensor.shape[0], n, 1) if shared_memory else vec[None].repeat(tensor.shape[0], 1, 1)

def differentiable_polynomial_rounding(input: Tensor) -> Tensor:
    """This function implements differentiable rounding.

    Args:
        input (Tensor): Input tensor of any shape to be rounded.

    Returns:
        output (Tensor): Pseudo rounded tensor of the same shape as input tensor.
    """
    input_round = input.round()
    output: Tensor = input_round + (input - input_round) ** 3
    return output

def differentiable_polynomial_floor(input: Tensor) -> Tensor:
    """This function implements differentiable floor.

    Args:
        input (Tensor): Input tensor of any shape to be floored.

    Returns:
        output (Tensor): Pseudo rounded tensor of the same shape as input tensor.
    """
    input_floor = input.floor()
    output: Tensor = input_floor + (input - 0.5 - input_floor) ** 3
    return output

def differentiable_clipping(input: Tensor, min_val: Optional[float]=None, max_val: Optional[float]=None, scale: float=0.02) -> Tensor:
    """Write a python function differentiable_clipping implements a differentiable and soft approximation of the clipping operation.

    Args:
        input (Tensor): Input tensor of any shape.
        min_val (Optional[float]): Minimum value.
        max_val (Optional[float]): Maximum value.
        scale (float): Scale value. Default 0.02.

    Returns:
        output (Tensor): Clipped output tensor of the same shape as the input tensor."""
    output = input
    if max_val is not None:
        max_val_tensor = torch.tensor(max_val, device=input.device, dtype=input.dtype)
        output = torch.minimum(max_val_tensor, max_val_tensor + (output - max_val_tensor) * scale)
    if min_val is not None:
        min_val_tensor = torch.tensor(min_val, device=input.device, dtype=input.dtype)
        output = torch.maximum(min_val_tensor, min_val_tensor + (output - min_val_tensor) * scale)
    return output
