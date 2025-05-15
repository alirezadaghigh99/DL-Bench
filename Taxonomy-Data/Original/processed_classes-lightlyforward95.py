    def forward(self, online: torch.Tensor, momentum: torch.Tensor) -> torch.Tensor:
        """Computes the MMCR loss for the online and momentum network outputs.

        Args:
            online:
                Output of the online network for the current batch. Expected to be
                of shape (batch_size, k, embedding_size), where k represents the
                number of randomly augmented views for each sample.
            momentum:
                Output of the momentum network for the current batch. Expected to be
                of shape (batch_size, k, embedding_size), where k represents the
                number of randomly augmented views for each sample.

        Returns:
            The computed loss value.
        """
        assert (
            online.shape == momentum.shape
        ), "online and momentum need to have the same shape"

        B = online.shape[0]

        # Concatenate and calculate centroid
        z = torch.cat([online, momentum], dim=1)
        c = torch.mean(z, dim=1)  # B x D

        # Calculate singular values
        _, S_z, _ = svd(z)
        _, S_c, _ = svd(c)

        # Calculate loss
        loss = -1.0 * torch.sum(S_c) + self.lmda * torch.sum(S_z) / B

        return loss