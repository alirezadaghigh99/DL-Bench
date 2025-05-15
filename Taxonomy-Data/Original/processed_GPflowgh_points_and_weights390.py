def gh_points_and_weights(n_gh: int) -> Tuple[tf.Tensor, tf.Tensor]:
    r"""
    Given the number of Gauss-Hermite points n_gh,
    returns the points z and the weights dz to perform the following
    uni-dimensional gaussian quadrature:

    X ~ N(mean, stddev²)
    E[f(X)] = ∫ f(x) p(x) dx = \sum_{i=1}^{n_gh} f(mean + stddev*z_i) dz_i

    :param n_gh: Number of Gauss-Hermite points
    :returns: Points z and weights dz to compute uni-dimensional gaussian expectation
    """
    z, dz = np.polynomial.hermite.hermgauss(n_gh)
    z = z * np.sqrt(2)
    dz = dz / np.sqrt(np.pi)
    z, dz = z.astype(default_float()), dz.astype(default_float())
    return tf.convert_to_tensor(z), tf.convert_to_tensor(dz)