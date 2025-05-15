import torch
from torch import Tensor
from typing import Dict, Any, Optional, Union, Tuple

class RandomErasing(IntensityAugmentationBase2D):
    def __init__(
        self,
        scale: Union[Tensor, Tuple[float, float]] = (0.02, 0.33),
        ratio: Union[Tensor, Tuple[float, float]] = (0.3, 3.3),
        value: float = 0.0,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self._param_generator = rg.RectangleEraseGenerator(scale, ratio, value)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # Extract dimensions of the input tensor
        batch_size, channels, height, width = input.shape

        # Generate a tensor of values to fill the erased area
        fill_value = torch.full((batch_size, channels, 1, 1), self.value, device=input.device, dtype=input.dtype)

        # Generate bounding boxes using the specified parameters
        xs = params["xs"]
        ys = params["ys"]
        widths = params["widths"]
        heights = params["heights"]

        # Create a mask from the bounding boxes
        mask = torch.ones_like(input, device=input.device, dtype=input.dtype)

        for i in range(batch_size):
            x1 = xs[i]
            y1 = ys[i]
            x2 = x1 + widths[i]
            y2 = y1 + heights[i]

            # Ensure the coordinates are within the image bounds
            x2 = min(x2, width)
            y2 = min(y2, height)

            # Apply the mask
            mask[i, :, y1:y2, x1:x2] = 0

        # Apply the mask to the input tensor, replacing the masked area with the generated values
        output = input * mask + fill_value * (1 - mask)

        return output