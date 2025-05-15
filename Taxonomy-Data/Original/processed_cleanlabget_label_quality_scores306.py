def get_label_quality_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
) -> np.ndarray:
    """Returns a label quality score for each datapoint.

    This is a function to compute label quality scores for standard (multi-class) classification datasets,
    where lower scores indicate labels less likely to be correct.

    Score is between 0 and 1.

    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels : np.ndarray
      A discrete vector of noisy labels, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, labels must be in 0, 1, ..., K-1.
      Note: multi-label classification is not supported by this method, each example must belong to a single class, e.g. format: ``labels = np.ndarray([1,0,2,1,1,0...])``.

    pred_probs : np.ndarray, optional
      An array of shape ``(N, K)`` of model-predicted probabilities,
      ``P(label=k|x)``. Each row of this matrix corresponds
      to an example `x` and contains the model-predicted probabilities that
      `x` belongs to each possible class, for each of the K classes. The
      columns must be ordered such that these probabilities correspond to
      class 0, 1, ..., K-1.

      **Note**: Returned label issues are most accurate when they are computed based on out-of-sample `pred_probs` from your model.
      To obtain out-of-sample predicted probabilities for every datapoint in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`.
      This is encouraged to get better results.

    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
      Label quality scoring method.

      Letting ``k = labels[i]`` and ``P = pred_probs[i]`` denote the given label and predicted class-probabilities
      for datapoint *i*, its score can either be:

      - ``'normalized_margin'``: ``P[k] - max_{k' != k}[ P[k'] ]``
      - ``'self_confidence'``: ``P[k]``
      - ``'confidence_weighted_entropy'``: ``entropy(P) / self_confidence``

      Note: the actual label quality scores returned by this method
      may be transformed versions of the above, in order to ensure
      their values lie between 0-1 with lower values indicating more likely mislabeled data.

      Let ``C = {0, 1, ..., K-1}`` be the set of classes specified for our classification task.

      The `normalized_margin` score works better for identifying class conditional label errors,
      i.e. examples for which another label in ``C`` is appropriate but the given label is not.

      The `self_confidence` score works better for identifying alternative label issues
      corresponding to bad examples that are: not from any of the classes in ``C``,
      well-described by 2 or more labels in ``C``,
      or generally just out-of-distribution (i.e. anomalous outliers).

    adjust_pred_probs : bool, optional
      Account for class imbalance in the label-quality scoring by adjusting predicted probabilities
      via subtraction of class confident thresholds and renormalization.
      Set this to ``True`` if you prefer to account for class-imbalance.
      See `Northcutt et al., 2021 <https://jair.org/index.php/jair/article/view/12125>`_.

    Returns
    -------
    label_quality_scores : np.ndarray
      Contains one score (between 0 and 1) per example.
      Lower scores indicate more likely mislabeled examples.

    See Also
    --------
    get_self_confidence_for_each_label
    get_normalized_margin_for_each_label
    get_confidence_weighted_entropy_for_each_label
    """

    assert_valid_inputs(
        X=None, y=labels, pred_probs=pred_probs, multi_label=False, allow_one_class=True
    )
    return _compute_label_quality_scores(
        labels=labels, pred_probs=pred_probs, method=method, adjust_pred_probs=adjust_pred_probs
    )