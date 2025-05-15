def compute_pyx(pred_probs, noise_matrix, inverse_noise_matrix):
    """Compute ``pyx := P(true_label=k|x)`` from ``pred_probs := P(label=k|x)``, `noise_matrix` and
    `inverse_noise_matrix`.

    This method is ROBUST - meaning it works well even when the
    noise matrices are estimated poorly by only using the diagonals of the
    matrices which tend to be easy to estimate correctly.

    Parameters
    ----------
    pred_probs : np.ndarray
        ``P(label=k|x)`` is a ``(N x K)`` matrix with K model-predicted probabilities.
        Each row of this matrix corresponds to an example `x` and contains the model-predicted
        probabilities that `x` belongs to each possible class.
        The columns must be ordered such that these probabilities correspond to class 0,1,2,...
        `pred_probs` should have been computed using 3 (or higher) fold cross-validation.

    noise_matrix : np.ndarray
        A conditional probability matrix (of shape ``(K, K)``) of the form ``P(label=k_s|true_label=k_y)`` containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of `noise_matrix` sum to 1.

    inverse_noise_matrix : np.ndarray
        A conditional probability matrix (of shape ``(K, K)``)  of the form ``P(true_label=k_y|label=k_s)`` representing
        the estimated fraction observed examples in each class `k_s`, that are
        mislabeled examples from every other class `k_y`. If None, the
        inverse_noise_matrix will be computed from `pred_probs` and `labels`.
        Assumes columns of `inverse_noise_matrix` sum to 1.

    Returns
    -------
    pyx : np.ndarray
        ``P(true_label=k|x)`` is a  ``(N, K)`` matrix of model-predicted probabilities.
        Each row of this matrix corresponds to an example `x` and contains the model-predicted
        probabilities that `x` belongs to each possible class.
        The columns must be ordered such that these probabilities correspond to class 0,1,2,...
        `pred_probs` should have been computed using 3 (or higher) fold cross-validation."""

    if len(np.shape(pred_probs)) != 2:
        raise ValueError(
            "Input parameter np.ndarray 'pred_probs' has shape "
            + str(np.shape(pred_probs))
            + ", but shape should be (N, K)"
        )

    pyx = (
        pred_probs
        * inverse_noise_matrix.diagonal()
        / np.clip(noise_matrix.diagonal(), a_min=TINY_VALUE, a_max=None)
    )
    # Make sure valid probabilities that sum to 1.0
    return np.apply_along_axis(
        func1d=clip_values, axis=1, arr=pyx, **{"low": 0.0, "high": 1.0, "new_sum": 1.0}
    )