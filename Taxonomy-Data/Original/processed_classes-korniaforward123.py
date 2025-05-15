    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"Input type is not a Tensor. Got {type(x)}")
        if not len(x.shape) == 4:
            raise ValueError(f"Invalid input shape, we expect Bx1xHxW. Got: {x.shape}")
        # Modify 'diff' gradient. Before we had lambda function, but it is not jittable
        grads_xy = -self.grad(x)
        gx = grads_xy[:, :, 0, :, :]
        gy = grads_xy[:, :, 1, :, :]
        y = torch.cat(cart2pol(gx, gy, self.eps), dim=1)
        return y