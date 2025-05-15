import torch

class RandomGamma(IntensityAugmentationBase2D):
    def __init__(
        self,
        gamma: Tuple[float, float] = (1.0, 1.0),
        gain: Tuple[float, float] = (1.0, 1.0),
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator(
            (gamma, "gamma_factor", None, None), (gain, "gain_factor", None, None)
        )

    def apply_transform(self, input: torch.Tensor, params: dict, flags: dict, transform: torch.Tensor = None) -> torch.Tensor:
        # Extract gamma and gain from params
        gamma_factor = params.get("gamma_factor", 1.0)
        gain_factor = params.get("gain_factor", 1.0)
        
        # Apply the gamma transformation
        # Ensure the input tensor is in the correct range (0, 1) for gamma correction
        input_clamped = torch.clamp(input, min=0.0, max=1.0)
        transformed = gain_factor * torch.pow(input_clamped, gamma_factor)
        
        return transformed