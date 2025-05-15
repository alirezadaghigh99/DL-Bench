def gaussian_noise_image(image: torch.Tensor, mean: float = 0.0, sigma: float = 0.1, clip: bool = True) -> torch.Tensor:
    if not image.is_floating_point():
        raise ValueError(f"Input tensor is expected to be in float dtype, got dtype={image.dtype}")
    if sigma < 0:
        raise ValueError(f"sigma shouldn't be negative. Got {sigma}")

    noise = mean + torch.randn_like(image) * sigma
    out = image + noise
    if clip:
        out = torch.clamp(out, 0, 1)
    return out