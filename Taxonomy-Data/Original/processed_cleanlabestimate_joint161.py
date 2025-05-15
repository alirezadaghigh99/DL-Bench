def estimate_joint(
    labels: LabelLike,
    pred_probs: np.ndarray,
    *,
    confident_joint: Optional[np.ndarray] = None,
    multi_label: bool = False,
) -> np.ndarray:
    """
    Estimates the joint distribution of label noise ``P(label=i, true_label=j)`` guaranteed to:

    * Sum to 1
    * Satisfy ``np.sum(joint_estimate, axis = 1) == p(labels)``

    Parameters
    ----------
    labels : np.ndarray or list
      Given class labels for each example in the dataset, some of which may be erroneous,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    pred_probs : np.ndarray
      Model-predicted class probabilities for each example in the dataset,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    confident_joint : np.ndarray, optional
      Array of estimated class label error statisics used for identifying label issues,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.
      The `confident_joint` can be computed using `~cleanlab.count.compute_confident_joint`.
      If not provided, it is internally computed from the given (noisy) `labels` and `pred_probs`.

    multi_label : bool, optional
      If ``False``, dataset is for regular (multi-class) classification, where each example belongs to exactly one class.
      If ``True``, dataset is for multi-label classification, where each example can belong to multiple classes.
      See documentation of `~cleanlab.count.compute_confident_joint` for details.

    Returns
    -------
    confident_joint_distribution : np.ndarray
      An array of shape ``(K, K)`` representing an
      estimate of the true joint distribution of noisy and true labels (if `multi_label` is False).
      If `multi_label` is True, an array of shape ``(K, 2, 2)`` representing an
      estimate of the true joint distribution of noisy and true labels for each class in a one-vs-rest fashion.
      Entry ``(c, i, j)`` in this array is the number of examples confidently counted into a ``(class c, noisy label=i, true label=j)`` bin,
      where `i, j` are either 0 or 1 to denote whether this example belongs to class `c` or not
      (recall examples can belong to multiple classes in multi-label classification).
    """

    if confident_joint is None:
        calibrated_cj = compute_confident_joint(
            labels,
            pred_probs,
            calibrate=True,
            multi_label=multi_label,
        )
    else:
        if labels is not None:
            calibrated_cj = calibrate_confident_joint(
                confident_joint, labels, multi_label=multi_label
            )
        else:
            calibrated_cj = confident_joint

    assert isinstance(calibrated_cj, np.ndarray)
    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")
        else:
            return _estimate_joint_multilabel(
                labels=labels, pred_probs=pred_probs, confident_joint=confident_joint
            )
    else:
        return calibrated_cj / np.clip(float(np.sum(calibrated_cj)), a_min=TINY_VALUE, a_max=None)