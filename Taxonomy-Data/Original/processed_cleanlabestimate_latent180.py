def estimate_latent(
    confident_joint: np.ndarray,
    labels: np.ndarray,
    *,
    py_method: str = "cnt",
    converge_latent_estimates: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the latent prior ``p(y)``, the noise matrix ``P(labels|y)`` and the
    inverse noise matrix ``P(y|labels)`` from the `confident_joint` ``count(labels, y)``. The
    `confident_joint` can be estimated by `~cleanlab.count.compute_confident_joint`
    which counts confident examples.

    Parameters
    ----------
    confident_joint : np.ndarray
      An array of shape ``(K, K)`` representing the confident joint, the matrix used for identifying label issues, which
      estimates a confident subset of the joint distribution of the noisy and true labels, ``P_{noisy label, true label}``.
      Entry ``(j, k)`` in the matrix is the number of examples confidently counted into the pair of ``(noisy label=j, true label=k)`` classes.
      The `confident_joint` can be computed using `~cleanlab.count.compute_confident_joint`.
      If not provided, it is computed from the given (noisy) `labels` and `pred_probs`.

    labels : np.ndarray
      A 1D array of shape ``(N,)`` containing class labels for a standard (multi-class) classification dataset. Some given labels may be erroneous.
      Elements must be integers in the set 0, 1, ..., K-1, where K is the number of classes.

    py_method : {"cnt", "eqn", "marginal", "marginal_ps"}, default="cnt"
      `py` is shorthand for the "class proportions (a.k.a prior) of the true labels".
      This method defines how to compute the latent prior ``p(true_label=k)``. Default is ``"cnt"``,
      which works well even when the noise matrices are estimated poorly by using
      the matrix diagonals instead of all the probabilities.

    converge_latent_estimates : bool, optional
      If ``True``, forces numerical consistency of estimates. Each is estimated
      independently, but they are related mathematically with closed form
      equivalences. This will iteratively make them mathematically consistent.

    Returns
    ------
    tuple
      A tuple containing (py, noise_matrix, inv_noise_matrix).

    Note
    ----
    Multi-label classification is not supported in this method.
    """

    num_classes = len(confident_joint)
    label_counts = value_counts_fill_missing_classes(labels, num_classes)
    # 'ps' is p(labels=k)
    ps = label_counts / float(len(labels))
    # Number of training examples confidently counted from each noisy class
    labels_class_counts = confident_joint.sum(axis=1).astype(float)
    # Number of training examples confidently counted into each true class
    true_labels_class_counts = confident_joint.sum(axis=0).astype(float)
    # p(label=k_s|true_label=k_y) ~ |label=k_s and true_label=k_y| / |true_label=k_y|
    noise_matrix = confident_joint / np.clip(true_labels_class_counts, a_min=TINY_VALUE, a_max=None)
    # p(true_label=k_y|label=k_s) ~ |true_label=k_y and label=k_s| / |label=k_s|
    inv_noise_matrix = confident_joint.T / np.clip(
        labels_class_counts, a_min=TINY_VALUE, a_max=None
    )
    # Compute the prior p(y), the latent (uncorrupted) class distribution.
    py = compute_py(
        ps,
        noise_matrix,
        inv_noise_matrix,
        py_method=py_method,
        true_labels_class_counts=true_labels_class_counts,
    )
    # Clip noise rates to be valid probabilities.
    noise_matrix = clip_noise_rates(noise_matrix)
    inv_noise_matrix = clip_noise_rates(inv_noise_matrix)
    # Make latent estimates mathematically agree in their algebraic relations.
    if converge_latent_estimates:
        py, noise_matrix, inv_noise_matrix = _converge_estimates(
            ps, py, noise_matrix, inv_noise_matrix
        )
        # Again clip py and noise rates into proper range [0,1)
        py = clip_values(py, low=1e-5, high=1.0, new_sum=1.0)
        noise_matrix = clip_noise_rates(noise_matrix)
        inv_noise_matrix = clip_noise_rates(inv_noise_matrix)

    return py, noise_matrix, inv_noise_matrix