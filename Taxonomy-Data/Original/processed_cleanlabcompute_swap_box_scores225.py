def compute_swap_box_scores(
    *,
    labels: Optional[List[Dict[str, Any]]] = None,
    predictions: Optional[List[np.ndarray]] = None,
    alpha: Optional[float] = None,
    high_probability_threshold: Optional[float] = None,
    overlapping_label_check: Optional[bool] = True,
    auxiliary_inputs: Optional[List[AuxiliaryTypesDict]] = None,
) -> List[np.ndarray]:
    """
    Returns a numeric score for each annotated bounding box in each image, estimating the likelihood that the class label for this box was not accidentally swapped with another class.
    This is a helper method mostly for advanced users.

    A swapped box error occurs when a bounding box should be labeled as a class different to what the current label is.
    Score per high-confidence predicted bounding box is between 0 and 1, with lower values indicating boxes we are more confident were overlooked in the given label.

    Each image has ``L`` annotated bounding boxes and ``M`` predicted bounding boxes.
    A score is calculated for each predicted box in each of the ``N`` images in dataset.

    Note: ``M`` and ``L`` can be a different values for each image, as the number of annotated and predicted boxes varies.

    Parameters
    ----------
    labels:
        A list of ``N`` dictionaries such that ``labels[i]`` contains the given labels for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    predictions:
        A list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model predictions for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    alpha:
        Optional weighting between IoU and Euclidean distance when calculating similarity between predicted and annotated boxes. High alpha means weighting IoU more heavily over Euclidean distance. If no alpha is provided, a good default is used.

    high_probability_threshold:
        Optional probability threshold that determines which predicted boxes are considered high-confidence when computing overlooked scores. If not provided, a good default is used.

    overlapping_label_check : bool, default = True
        If True, boxes annotated with more than one class label have their swap score penalized. Set this to False if you are not concerned when two very similar boxes exist with different class labels in the given annotations.

    auxiliary_inputs:
        Optional list of ``N`` dictionaries containing keys for sub-parts of label and prediction per image. Useful to minimize computation when computing multiple box scores for a single set of images. For the `i`-th image, `auxiliary_inputs[i]` should contain following keys:

       * pred_labels: np.ndarray
            Array of predicted classes for `i`-th image of shape ``(M,)``.
       * pred_label_probs: np.ndarray
            Array of predicted class probabilities for `i`-th image of shape ``(M,)``.
       * pred_bboxes: np.ndarray
            Array of predicted bounding boxes for `i`-th image of shape ``(M, 4)``.
       * lab_labels: np.ndarray
            Array of given label classed for `i`-th image of shape ``(L,)``.
       * lab_bboxes: np.ndarray
            Array of given label bounding boxes for `i`-th image of shape ``(L, 4)``.
       * similarity_matrix: np.ndarray
            Similarity matrix between labels and predictions `i`-th image.
       * min_possible_similarity: float
            Minimum possible similarity value greater than 0 between labels and predictions for the entire dataset.
    Returns
    ---------
    scores_swap:
        A list of ``N`` numpy arrays where scores_swap[i] is an array of size ``L`` swap scores per annotated box for the `i`-th image.
    """
    (
        alpha,
        low_probability_threshold,
        high_probability_threshold,
        temperature,
    ) = _get_valid_subtype_score_params(alpha, None, high_probability_threshold, None)

    if auxiliary_inputs is None:
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(alpha, labels, predictions)

    scores_swap = []
    for auxiliary_inputs in auxiliary_inputs:
        scores_swap_per_box = _compute_swap_box_scores_for_image(
            alpha=alpha,
            high_probability_threshold=high_probability_threshold,
            overlapping_label_check=overlapping_label_check,
            **auxiliary_inputs,
        )
        scores_swap.append(scores_swap_per_box)
    return scores_swap