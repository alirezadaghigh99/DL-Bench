def inverse_transform(self, x: Tensor) -> Tensor:
    r"""Apply the inverse transform to the whitened data.

    Args:
        x: Whitened data.

    Returns:
        Original data.
    """
    if not self.fitted:
        raise RuntimeError("Needs to be fitted first before running. Please call fit or set include_fit to True.")

    if not self.compute_inv:
        raise RuntimeError("Did not compute inverse ZCA. Please set compute_inv to True")

    if self.transform_inv is None:
        raise TypeError("The transform inverse should be a Tensor. Gotcha None.")

    # Assuming self.mean is the mean used during the whitening process
    # and self.transform_inv is the inverse transformation matrix
    if hasattr(self, 'mean'):
        return x @ self.transform_inv + self.mean
    else:
        return x @ self.transform_inv