    def forward(self, x, update=True):
        """Normalize mean and variance of values based on emprical values.

        Args:
            x (ndarray or Variable): Input values
            update (bool): Flag to learn the input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        if update:
            self.experience(x)

        normalized = (x - self._mean) * self._std_inverse
        if self.clip_threshold is not None:
            normalized = torch.clamp(
                normalized, -self.clip_threshold, self.clip_threshold
            )
        return normalized