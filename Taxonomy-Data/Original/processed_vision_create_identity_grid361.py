def _create_identity_grid(size: List[int]) -> Tensor:
    hw_space = [torch.linspace((-s + 1) / s, (s - 1) / s, s) for s in size]
    grid_y, grid_x = torch.meshgrid(hw_space, indexing="ij")
    return torch.stack([grid_x, grid_y], -1).unsqueeze(0)  # 1 x H x W x 2