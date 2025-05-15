def noise_matrix_is_valid(noise_matrix, py, *, verbose=False) -> bool:
    """Given a prior `py` representing ``p(true_label=k)``, checks if the given `noise_matrix` is a
    learnable matrix. Learnability means that it is possible to achieve
    better than random performance, on average, for the amount of noise in
    `noise_matrix`.

    Parameters
    ----------
    noise_matrix : np.ndarray
      An array of shape ``(K, K)`` representing the conditional probability
      matrix ``P(label=k_s|true_label=k_y)`` containing the fraction of
      examples in every class, labeled as every other class. Assumes columns of
      `noise_matrix` sum to 1.

    py : np.ndarray
      An array of shape ``(K,)`` representing the fraction (prior probability)
      of each true class label, ``P(true_label = k)``.

    Returns
    -------
    is_valid : bool
      Whether the noise matrix is a learnable matrix.
    """

    # Number of classes
    K = len(py)

    # let's assume some number of training examples for code readability,
    # but it doesn't matter what we choose as it's not actually used.
    N = float(10000)

    ps = np.dot(noise_matrix, py)  # P(true_label=k)

    # P(label=k, true_label=k')
    joint_noise = np.multiply(noise_matrix, py)  # / float(N)

    # Check that joint_probs is valid probability matrix
    if not (abs(joint_noise.sum() - 1.0) < FLOATING_POINT_COMPARISON):
        return False

    # Check that noise_matrix is a valid matrix
    # i.e. check p(label=k)*p(true_label=k) < p(label=k, true_label=k)
    for i in range(K):
        C = N * joint_noise[i][i]
        E1 = N * joint_noise[i].sum() - C
        E2 = N * joint_noise.T[i].sum() - C
        O = N - E1 - E2 - C
        if verbose:
            print(
                "E1E2/C",
                round(E1 * E2 / C),
                "E1",
                round(E1),
                "E2",
                round(E2),
                "C",
                round(C),
                "|",
                round(E1 * E2 / C + E1 + E2 + C),
                "|",
                round(E1 * E2 / C),
                "<",
                round(O),
            )
            print(
                round(ps[i] * py[i]),
                "<",
                round(joint_noise[i][i]),
                ":",
                ps[i] * py[i] < joint_noise[i][i],
            )

        if not (ps[i] * py[i] < joint_noise[i][i]):
            return False

    return True