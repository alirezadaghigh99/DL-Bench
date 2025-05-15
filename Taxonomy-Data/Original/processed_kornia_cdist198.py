def _cdist(d1: Tensor, d2: Tensor) -> Tensor:
    r"""Manual `torch.cdist` for M1."""
    if (not is_mps_tensor_safe(d1)) and (not is_mps_tensor_safe(d2)):
        return torch.cdist(d1, d2)
    d1_sq = (d1**2).sum(dim=1, keepdim=True)
    d2_sq = (d2**2).sum(dim=1, keepdim=True)
    dm = d1_sq.repeat(1, d2.size(0)) + d2_sq.repeat(1, d1.size(0)).t() - 2.0 * d1 @ d2.t()
    dm = dm.clamp(min=0.0).sqrt()
    return dm