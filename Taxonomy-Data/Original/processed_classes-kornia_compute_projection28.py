    def _compute_projection(self, x: float, y: float, invd: float) -> Tensor:
        if self._dst_proj_src is None or self._pinhole_src is None:
            raise ValueError("Please, call compute_projection_matrix.")

        point = tensor([[[x], [y], [invd], [1.0]]], device=self._dst_proj_src.device, dtype=self._dst_proj_src.dtype)
        flow = torch.matmul(self._dst_proj_src, point)
        z = 1.0 / flow[:, 2]
        _x = flow[:, 0] * z
        _y = flow[:, 1] * z
        return kornia_ops.concatenate([_x, _y], 1)