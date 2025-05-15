def _subtract_confident_thresholds(
    labels: Optional[np.ndarray],
    pred_probs: np.ndarray,
    multi_label: bool = False,
    confident_thresholds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return adjusted predicted probabilities by subtracting the class confident thresholds and renormalizing.

    The confident class threshold for a class j is the expected (average) "self-confidence" for class j.
    The purpose of this adjustment is to handle class imbalance.

    Parameters
    ----------
    labels : np.ndarray
      Labels in the same format expected by the `cleanlab.count.get_confident_thresholds()` method.
      If labels is None, confident_thresholds needs to be passed in as it will not be calculated.
    pred_probs : np.ndarray (shape (N, K))
      Predicted-probabilities in the same format expected by the `cleanlab.count.get_confident_thresholds()` method.
    confident_thresholds : np.ndarray (shape (K,))
      Pre-calculated confident thresholds. If passed in, function will subtract these thresholds instead of calculating
      confident_thresholds from the given labels and pred_probs.
    multi_label : bool, optional
      If ``True``, labels should be an iterable (e.g. list) of iterables, containing a
      list of labels for each example, instead of just a single label.
      The multi-label setting supports classification tasks where an example has 1 or more labels.
      Example of a multi-labeled `labels` input: ``[[0,1], [1], [0,2], [0,1,2], [0], [1], ...]``.
      The major difference in how this is calibrated versus single-label is that
      the total number of errors considered is based on the number of labels,
      not the number of examples. So, the calibrated `confident_joint` will sum
      to the number of total labels.

    Returns
    -------
    pred_probs_adj : np.ndarray (float)
      Adjusted pred_probs.
    """
    # Get expected (average) self-confidence for each class
    # TODO: Test this for multi-label
    if confident_thresholds is None:
        if labels is None:
            raise ValueError(
                "Cannot calculate confident_thresholds without labels. Pass in either labels or already calculated "
                "confident_thresholds parameter. "
            )
        confident_thresholds = get_confident_thresholds(labels, pred_probs, multi_label=multi_label)

    # Subtract the class confident thresholds
    pred_probs_adj = pred_probs - confident_thresholds

    # Re-normalize by shifting data to take care of negative values from the subtraction
    pred_probs_adj += confident_thresholds.max()
    pred_probs_adj /= pred_probs_adj.sum(axis=1, keepdims=True)

    return pred_probs_adj