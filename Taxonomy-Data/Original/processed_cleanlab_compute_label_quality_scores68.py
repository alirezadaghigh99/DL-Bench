def _compute_label_quality_scores(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    *,
    method: Optional[str] = "objectlab",
    aggregation_weights: Optional[Dict[str, float]] = None,
    threshold: Optional[float] = None,
    overlapping_label_check: Optional[bool] = True,
    verbose: bool = True,
) -> np.ndarray:
    """Internal function to prune extra bounding boxes and compute label quality scores based on passed in method."""

    pred_probs_prepruned = False
    min_pred_prob = _get_min_pred_prob(predictions)
    aggregation_weights = _get_aggregation_weights(aggregation_weights)

    if threshold is not None:
        predictions = _prune_by_threshold(
            predictions=predictions, threshold=threshold, verbose=verbose
        )
        if np.abs(min_pred_prob - threshold) < 0.001 and threshold > 0:
            pred_probs_prepruned = True  # the provided threshold is the threshold used for pre_pruning the pred_probs during model prediction.
    else:
        threshold = min_pred_prob  # assume model was not pre_pruned if no threshold was provided

    if method == "objectlab":
        scores = _get_subtype_label_quality_scores(
            labels=labels,
            predictions=predictions,
            alpha=ALPHA,
            low_probability_threshold=LOW_PROBABILITY_THRESHOLD,
            high_probability_threshold=HIGH_PROBABILITY_THRESHOLD,
            temperature=TEMPERATURE,
            aggregation_weights=aggregation_weights,
            overlapping_label_check=overlapping_label_check,
        )
    else:
        raise ValueError(
            "Invalid method: '{}' is not a valid method for computing label quality scores. Please use the 'objectlab' method.".format(
                method
            )
        )
    return scores