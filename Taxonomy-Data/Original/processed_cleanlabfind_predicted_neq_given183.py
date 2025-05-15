def find_predicted_neq_given(
    labels: LabelLike, pred_probs: np.ndarray, *, multi_label: bool = False
) -> np.ndarray:
    """A simple baseline approach that considers ``argmax(pred_probs) != labels`` as the examples with label issues.

    Parameters
    ----------
    labels : np.ndarray or list
      Labels in the same format expected by the `~cleanlab.filter.find_label_issues` function.

    pred_probs : np.ndarray
      Predicted-probabilities in the same format expected by the `~cleanlab.filter.find_label_issues` function.

    multi_label : bool, optional
      Whether each example may have multiple labels or not (see documentation for the `~cleanlab.filter.find_label_issues` function).

    Returns
    -------
    label_issues_mask : np.ndarray
      A boolean mask for the entire dataset where ``True`` represents a
      label issue and ``False`` represents an example that is accurately
      labeled with high confidence.
    """

    assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs, multi_label=multi_label)
    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")
        else:
            return _find_predicted_neq_given_multilabel(labels=labels, pred_probs=pred_probs)
    else:
        return np.argmax(pred_probs, axis=1) != np.asarray(labels)