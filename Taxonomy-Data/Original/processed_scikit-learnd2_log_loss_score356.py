def d2_log_loss_score(y_true, y_pred, *, sample_weight=None, labels=None):
    """
    :math:`D^2` score function, fraction of log loss explained.

    Best possible score is 1.0 and it can be negative (because the model can be
    arbitrarily worse). A model that always predicts the per-class proportions
    of `y_true`, disregarding the input features, gets a D^2 score of 0.0.

    Read more in the :ref:`User Guide <d2_score_classification>`.

    .. versionadded:: 1.5

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        The actuals labels for the n_samples samples.

    y_pred : array-like of shape (n_samples, n_classes) or (n_samples,)
        Predicted probabilities, as returned by a classifier's
        predict_proba method. If ``y_pred.shape = (n_samples,)``
        the probabilities provided are assumed to be that of the
        positive class. The labels in ``y_pred`` are assumed to be
        ordered alphabetically, as done by
        :class:`~sklearn.preprocessing.LabelBinarizer`.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    labels : array-like, default=None
        If not provided, labels will be inferred from y_true. If ``labels``
        is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
        assumed to be binary and are inferred from ``y_true``.

    Returns
    -------
    d2 : float or ndarray of floats
        The D^2 score.

    Notes
    -----
    This is not a symmetric function.

    Like R^2, D^2 score may be negative (it need not actually be the square of
    a quantity D).

    This metric is not well-defined for a single sample and will return a NaN
    value if n_samples is less than two.
    """
    y_pred = check_array(y_pred, ensure_2d=False, dtype="numeric")
    check_consistent_length(y_pred, y_true, sample_weight)
    if _num_samples(y_pred) < 2:
        msg = "D^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float("nan")

    # log loss of the fitted model
    numerator = log_loss(
        y_true=y_true,
        y_pred=y_pred,
        normalize=False,
        sample_weight=sample_weight,
        labels=labels,
    )

    # Proportion of labels in the dataset
    weights = _check_sample_weight(sample_weight, y_true)

    _, y_value_indices = np.unique(y_true, return_inverse=True)
    counts = np.bincount(y_value_indices, weights=weights)
    y_prob = counts / weights.sum()
    y_pred_null = np.tile(y_prob, (len(y_true), 1))

    # log loss of the null model
    denominator = log_loss(
        y_true=y_true,
        y_pred=y_pred_null,
        normalize=False,
        sample_weight=sample_weight,
        labels=labels,
    )

    return 1 - (numerator / denominator)