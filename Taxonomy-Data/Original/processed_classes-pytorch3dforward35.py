    def forward(
        self, x: torch.Tensor, diag_cov: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [..., dim]
            diag_cov: An optional tensor of shape `(..., dim)`
                representing the diagonal covariance matrices of our Gaussians, joined with x
                as means of the Gaussians.

        Returns:
            embedding: a harmonic embedding of `x` of shape
            [..., (n_harmonic_functions * 2 + int(append_input)) * num_points_per_ray]
        """
        # [..., dim, n_harmonic_functions]
        embed = x[..., None] * self._frequencies
        # [..., 1, dim, n_harmonic_functions] + [2, 1, 1] => [..., 2, dim, n_harmonic_functions]
        embed = embed[..., None, :, :] + self._zero_half_pi[..., None, None]
        # Use the trig identity cos(x) = sin(x + pi/2)
        # and do one vectorized call to sin([x, x+pi/2]) instead of (sin(x), cos(x)).
        embed = embed.sin()
        if diag_cov is not None:
            x_var = diag_cov[..., None] * torch.pow(self._frequencies, 2)
            exp_var = torch.exp(-0.5 * x_var)
            # [..., 2, dim, n_harmonic_functions]
            embed = embed * exp_var[..., None, :, :]

        embed = embed.reshape(*x.shape[:-1], -1)

        if self.append_input:
            return torch.cat([embed, x], dim=-1)
        return embed