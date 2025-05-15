    def K_diag(self, X: TensorType) -> tf.Tensor:
        X_product = self._diag_weighted_product(X)
        const = (1.0 / np.pi) * self._J(to_default_float(0.0))
        return self.variance * const * X_product ** self.order