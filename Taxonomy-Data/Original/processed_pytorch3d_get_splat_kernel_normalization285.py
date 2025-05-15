def _get_splat_kernel_normalization(
    offsets: torch.Tensor,
    sigma: float = 0.5,
):
    if sigma <= 0.0:
        raise ValueError("Only positive standard deviations make sense.")

    epsilon = 0.05
    normalization_constant = torch.exp(
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        -(offsets**2).sum(dim=1)
        / (2 * sigma**2)
    ).sum()

    # We add an epsilon to the normalization constant to ensure the gradient will travel
    # through non-boundary pixels' normalization factor, see Sec. 3.3.1 in "Differentia-
    # ble Surface Rendering via Non-Differentiable Sampling", Cole et al.
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    return (1 + epsilon) / normalization_constant