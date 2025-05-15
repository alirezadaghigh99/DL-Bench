def compute_ps_py_inv_noise_matrix(
    labels, noise_matrix
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``ps := P(labels=k), py := P(true_labels=k)``, and the inverse noise matrix.

    Parameters
    ----------
    labels : np.ndarray
          A discrete vector of noisy labels, i.e. some labels may be erroneous.
          *Format requirements*: for dataset with `K` classes, labels must be in ``{0,1,...,K-1}``.

    noise_matrix : np.ndarray
        A conditional probability matrix (of shape ``(K, K)``) of the form ``P(label=k_s|true_label=k_y)`` containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1."""

    ps = value_counts(labels) / float(len(labels))  # p(labels=k)
    py, inverse_noise_matrix = compute_py_inv_noise_matrix(ps, noise_matrix)
    return ps, py, inverse_noise_matrix