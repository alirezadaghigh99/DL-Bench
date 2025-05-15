def multivariate_normal(x: TensorType, mu: TensorType, L: TensorType) -> tf.Tensor:
    """
    Computes the log-density of a multivariate normal.

    :param x: sample(s) for which we want the density
    :param mu: mean(s) of the normal distribution
    :param L: Cholesky decomposition of the covariance matrix
    :return: log densities
    """

    d = x - mu
    alpha = tf.linalg.triangular_solve(L, d, lower=True)
    num_dims = tf.cast(tf.shape(d)[0], L.dtype)
    p = -0.5 * tf.reduce_sum(tf.square(alpha), 0)
    p -= 0.5 * num_dims * np.log(2 * np.pi)
    p -= tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))

    return p