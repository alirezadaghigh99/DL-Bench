def normalize_kernel2d(input: Tensor) -> Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    KORNIA_CHECK_SHAPE(input, ["*", "H", "W"])

    norm = input.abs().sum(dim=-1).sum(dim=-1)

    return input / (norm[..., None, None])