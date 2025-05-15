    def fit(self, x: Tensor) -> "ZCAWhitening":
        r"""Fit ZCA whitening matrices to the data.

        Args:

            x: Input data.

        returns:
            Returns a fitted ZCAWhiten object instance.
        """
        T, mean, T_inv = zca_mean(x, self.dim, self.unbiased, self.eps, self.compute_inv)

        self.mean_vector = mean
        self.transform_matrix = T
        if T_inv is None:
            self.transform_inv = torch.empty([0])
        else:
            self.transform_inv = T_inv

        if self.detach_transforms:
            self.mean_vector = self.mean_vector.detach()
            self.transform_matrix = self.transform_matrix.detach()
            self.transform_inv = self.transform_inv.detach()

        self.fitted = True

        return self