import torch
from typing import Union, Tuple
from torch import Tensor

class RandomGaussianBlur(IntensityAugmentationBase2D):
    def __init__(
        self,
        kernel_size: Union[Tuple[int, int], int],
        sigma: Union[Tuple[float, float], Tensor],
        border_type: str = "reflect",
        separable: bool = True,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)

        self.flags = {
            "kernel_size": kernel_size,
            "separable": separable,
            "border_type": BorderType.get(border_type),
        }
        self._param_generator = rg.RandomGaussianBlurGenerator(sigma)

        self._gaussian_blur2d_fn = gaussian_blur2d

    def apply_transform(self, input: Tensor, params: dict) -> Tensor:
        # Ensure input is 4D
        if input.dim() == 3:
            input = input.unsqueeze(0)  # Add batch dimension

        # Extract parameters
        kernel_size = self.flags["kernel_size"]
        separable = self.flags["separable"]
        border_type = self.flags["border_type"]
        sigma = params.get("sigma", None)

        if sigma is None:
            raise ValueError("Sigma parameter is required for Gaussian blur.")

        # Apply Gaussian blur
        output = self._gaussian_blur2d_fn(
            input,
            kernel_size=kernel_size,
            sigma=sigma,
            border_type=border_type,
            separable=separable
        )

        return output

# Assuming gaussian_blur2d and other dependencies are defined elsewhere