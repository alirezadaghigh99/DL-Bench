    def stack(self, *others: "Transform3d") -> "Transform3d":
        """
        Return a new batched Transform3d representing the batch elements from
        self and all the given other transforms all batched together.

        Args:
            *others: Any number of Transform3d objects

        Returns:
            A new Transform3d.
        """
        transforms = [self] + list(others)
        matrix = torch.cat([t.get_matrix() for t in transforms], dim=0)
        out = Transform3d(dtype=self.dtype, device=self.device)
        out._matrix = matrix
        return out