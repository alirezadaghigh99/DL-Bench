    def _get_matrix_inverse(self) -> torch.Tensor:
        """
        Return the inverse of self._matrix.
        """
        return torch.inverse(self._matrix)