def prediction_result(proba, multi_label, num_classes, return_proba=False):
    """Helper method which returns a single- or multi-label prediction result.

    Parameters
    ----------
    proba : np.ndarray[float]
        A (dense) probability matrix of shape (num_instances, num_classes).
    multi_label : bool
        If True, this method returns a result suitable for a multi-label classification,
        otherwise for a single-label classification.
    num_classes : int
        The number of classes.
    return_proba : bool, default=False
        Also returns the probability if `True`. This is intended to be used with `multi_label=True`
        where it returns a sparse matrix with only the probabilities for the predicted labels. For
        the single-label case this simply returns the given `proba` input.

    Returns
    -------
    result : np.ndarray[int] or csr_matrix
        An empty ndarray of predictions if `return_prediction` is True.
    proba : np.ndarray[float] or csr_matrix[np.float64]
        An empty ndarray of predictions if `return_prediction` is True.
    """

    if multi_label:
        predictions_binarized = np.where(proba > 0.5, 1, 0)

        def multihot_to_list(x):
            return [i for i, item in enumerate(x) if item > 0]

        predictions = [multihot_to_list(row) for row in predictions_binarized]
        predictions = list_to_csr(predictions, shape=(len(predictions), num_classes))

        if return_proba:
            data = proba[predictions_binarized.astype(bool)]
            proba = csr_matrix((data, predictions.indices, predictions.indptr),
                               shape=predictions.shape,
                               dtype=np.float64)
    else:
        predictions = np.argmax(proba, axis=1)

    if return_proba:
        return predictions, proba

    return predictions