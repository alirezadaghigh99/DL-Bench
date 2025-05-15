def empty_result(multi_label, num_classes, return_prediction=True, return_proba=True):
    """Helper method which returns an empty classification result. This ensures that all results
    have the correct dtype.

    At least one of `prediction` and `proba` must be `True`.

    Parameters
    ----------
    multi_label : bool
        Indicates a multi-label setting if `True`, otherwise a single-label setting
        when set to `False`.
    num_classes : int
        The number of classes.
    return_prediction : bool, default=True
        If `True`, returns an empty result of prediction.
    return_proba : bool, default=True
        If `True`, returns an empty result of probabilities.

    Returns
    -------
    predictions : np.ndarray[np.int64]
        An empty ndarray of predictions if `return_prediction` is True.
    proba : np.ndarray[float]
        An empty ndarray of probabilities if `return_proba` is True.
    """
    if not return_prediction and not return_proba:
        raise ValueError('Invalid usage: At least one of \'prediction\' or \'proba\' must be True')

    elif multi_label:
        if return_prediction and return_proba:
            return csr_matrix((0, num_classes), dtype=np.int64), csr_matrix((0, num_classes), dtype=float)
        elif return_prediction:
            return csr_matrix((0, num_classes), dtype=np.int64)
        else:
            return csr_matrix((0, num_classes), dtype=float)
    else:
        if return_prediction and return_proba:
            return np.empty((0, num_classes), dtype=np.int64), np.empty(shape=(0, num_classes), dtype=float)
        elif return_prediction:
            return np.empty((0, num_classes), dtype=np.int64)
        else:
            return np.empty((0, num_classes), dtype=float)