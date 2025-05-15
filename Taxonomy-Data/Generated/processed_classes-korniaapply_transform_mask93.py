import torch
from torch import Tensor
from typing import Dict, Any, Optional, Union, Tuple

class IntensityAugmentationBase2D:
    def __init__(self, p: float, same_on_batch: bool, keepdim: bool) -> None:
        self.p = p
        self.same_on_batch = same_on_batch
        self.keepdim = keepdim

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
        # Assuming rg.RectangleEraseGenerator is defined elsewhere
        self._param_generator = rg.RectangleEraseGenerator(scale, ratio, value)

    def apply_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # Extract parameters
        xs = params["xs"]
        ys = params["ys"]
        widths = params["widths"]
        heights = params["heights"]

        # Iterate over each image in the batch
        for i in range(input.size(0)):
            x = xs[i].item()
            y = ys[i].item()
            width = widths[i].item()
            height = heights[i].item()

            # Erase the specified rectangle by setting it to the specified value
            input[i, :, y:y+height, x:x+width] = self.value

        return input

