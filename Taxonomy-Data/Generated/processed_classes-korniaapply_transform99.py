import torch
from typing import Dict, Optional

class RandomContrast(IntensityAugmentationBase2D):
    def __init__(
        self,
        contrast: Tuple[float, float] = (1.0, 1.0),
        clip_output: bool = True,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.contrast: Tensor = _range_bound(contrast, "contrast", center=1.0)
        self._param_generator = rg.PlainUniformGenerator((self.contrast, "contrast_factor", None, None))
        self.clip_output = clip_output

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, torch.Tensor],
        transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Extract the contrast factor from the params
        contrast_factor = params.get("contrast_factor", torch.tensor(1.0))

        # Calculate the mean of the input tensor
        mean = input.mean(dim=(-3, -2, -1), keepdim=True)

        # Apply the contrast adjustment
        output = (input - mean) * contrast_factor + mean

        # Clip the output if required
        if self.clip_output:
            output = torch.clamp(output, 0.0, 1.0)

        return output