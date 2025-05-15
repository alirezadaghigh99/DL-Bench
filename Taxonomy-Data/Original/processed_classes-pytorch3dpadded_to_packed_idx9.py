    def padded_to_packed_idx(self):
        """
        Return a 1D tensor x with length equal to the total number of points
        such that points_packed()[i] is element x[i] of the flattened padded
        representation.
        The packed representation can be calculated as follows.

        .. code-block:: python

            p = points_padded().reshape(-1, 3)
            points_packed = p[x]

        Returns:
            1D tensor of indices.
        """
        if self._padded_to_packed_idx is not None:
            return self._padded_to_packed_idx
        if self._N == 0:
            self._padded_to_packed_idx = []
        else:
            self._padded_to_packed_idx = torch.cat(
                [
                    torch.arange(v, dtype=torch.int64, device=self.device) + i * self._P
                    for (i, v) in enumerate(self.num_points_per_cloud())
                ],
                dim=0,
            )
        return self._padded_to_packed_idx