from typing import Any, Callable, Optional, Tuple
import torch
from torch import Tensor
from torch.autograd import Function

class STEFunction(Function):
    """Straight-Through Estimation (STE) function."""

    @staticmethod
    def forward(ctx: Any, input: Tensor, output: Tensor, grad_fn: Optional[Callable[..., Any]] = None) -> Tensor:
        ctx.in_shape = input.shape
        ctx.out_shape = output.shape
        ctx.grad_fn = grad_fn
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, Tensor, None]:
        # Retrieve the gradient function if provided
        grad_fn = ctx.grad_fn

        # If a gradient function is provided, apply it to the grad_output
        if grad_fn is not None:
            grad_input = grad_fn(grad_output)
        else:
            # Otherwise, pass the gradient through as is
            grad_input = grad_output

        # Return the gradient for each input of the forward function
        # grad_input for `input`, None for `output` (since it doesn't require gradient),
        # and None for `grad_fn` (since it's not a tensor)
        return grad_input, None, None