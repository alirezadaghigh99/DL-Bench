    if Dout is None:
        Dout = ()

    # Gauss-Hermite quadrature points and weights
    gh_x, gh_w = np.polynomial.hermite.hermgauss(H)
    gh_x = tf.constant(gh_x, dtype=means.dtype)
    gh_w = tf.constant(gh_w, dtype=means.dtype)

    # Reshape and tile the quadrature points
    gh_x = tf.reshape(gh_x, [1, 1, H, 1])
    gh_w = tf.reshape(gh_w, [1, 1, H])

    # Compute the Cholesky decomposition of the covariance matrices
    chol_covs = tf.linalg.cholesky(covs)

    # Prepare the shape for broadcasting
    means = tf.expand_dims(means, axis=2)  # [N, Din, 1]
    chol_covs = tf.expand_dims(chol_covs, axis=2)  # [N, Din, Din, 1]

    # Transform the quadrature points
    transformed_x = means + tf.matmul(chol_covs, gh_x)  # [N, Din, H, 1]

    # Evaluate the function at the transformed points
    func_values = func(tf.squeeze(transformed_x, axis=-1))  # [N, H, Dout...]

    # Compute the weighted sum of the function values
    weighted_sum = tf.reduce_sum(func_values * gh_w, axis=1)  # [N, Dout...]

    # Scale by the normalization factor
    normalization_factor = np.pi ** (Din / 2)
    result = weighted_sum / normalization_factor

    return result