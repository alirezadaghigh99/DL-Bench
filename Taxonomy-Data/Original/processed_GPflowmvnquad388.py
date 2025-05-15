def mvnquad(
    func: Callable[[tf.Tensor], tf.Tensor],
    means: TensorType,
    covs: TensorType,
    H: int,
    Din: Optional[int] = None,
    Dout: Optional[Tuple[int, ...]] = None,
) -> tf.Tensor:
    """
    Computes N Gaussian expectation integrals of a single function 'f'
    using Gauss-Hermite quadrature.

    :param f: integrand function. Takes one input of shape ?xD.
    :param H: Number of Gauss-Hermite evaluation points.
    :param Din: Number of input dimensions. Needs to be known at call-time.
    :param Dout: Number of output dimensions. Defaults to (). Dout is assumed
        to leave out the item index, i.e. f actually maps (?xD)->(?x*Dout).
    :return: quadratures
    """
    # Figure out input shape information
    if Din is None:
        Din = means.shape[1]

    if Din is None:
        raise ValueError(
            "If `Din` is passed as `None`, `means` must have a known shape. "
            "Running mvnquad in `autoflow` without specifying `Din` and `Dout` "
            "is problematic. Consider using your own session."
        )  # pragma: no cover

    xn, wn = mvhermgauss(H, Din)
    N = means.shape[0]

    # transform points based on Gaussian parameters
    cholXcov = tf.linalg.cholesky(covs)  # NxDxD
    Xt = tf.linalg.matmul(
        cholXcov, tf.tile(xn[None, :, :], (N, 1, 1)), transpose_b=True
    )  # NxDxH**D
    X = 2.0 ** 0.5 * Xt + tf.expand_dims(means, 2)  # NxDxH**D
    Xr = tf.reshape(tf.transpose(X, [2, 0, 1]), (-1, Din))  # (H**D*N)xD

    # perform quadrature
    fevals = func(Xr)
    if Dout is None:
        Dout = tuple((d if type(d) is int else d.value) for d in fevals.shape[1:])

    if any([d is None for d in Dout]):
        raise ValueError(
            "If `Dout` is passed as `None`, the output of `func` must have known "
            "shape. Running mvnquad in `autoflow` without specifying `Din` and `Dout` "
            "is problematic. Consider using your own session."
        )  # pragma: no cover
    fX = tf.reshape(
        fevals,
        (
            H ** Din,
            N,
        )
        + Dout,
    )
    wr = np.reshape(wn * np.pi ** (-Din * 0.5), (-1,) + (1,) * (1 + len(Dout)))
    return tf.reduce_sum(fX * wr, 0)