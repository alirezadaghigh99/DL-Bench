import torch
import torchvision.transforms.functional as F

class RandomSaturation(IntensityAugmentationBase2D):
    def __init__(
        self,
        saturation: Tuple[float, float] = (1.0, 1.0),
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.saturation: Tensor = _range_bound(saturation, "saturation", center=1.0)
        self._param_generator = rg.PlainUniformGenerator((self.saturation, "saturation_factor", None, None))

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Extract the saturation factor from params
        saturation_factor = params.get("saturation_factor", torch.tensor(1.0))

        # Convert the input tensor from RGB to HSV
        hsv_image = F.rgb_to_hsv(input)

        # Adjust the saturation channel
        h, s, v = hsv_image.unbind(dim=-3)
        s = s * saturation_factor
        s = torch.clamp(s, 0, 1)  # Ensure the saturation is within valid range

        # Recombine the channels and convert back to RGB
        hsv_adjusted = torch.stack((h, s, v), dim=-3)
        output = F.hsv_to_rgb(hsv_adjusted)

        return output