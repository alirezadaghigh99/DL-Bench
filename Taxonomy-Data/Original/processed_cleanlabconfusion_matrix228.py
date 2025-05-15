def confusion_matrix(true, pred) -> np.ndarray:
    """Implements a confusion matrix for true labels
    and predicted labels. true and pred MUST BE the same length
    and have the same distinct set of class labels represented.

    Results are identical (and similar computation time) to:
        "sklearn.metrics.confusion_matrix"

    However, this function avoids the dependency on sklearn.

    Parameters
    ----------
    true : np.ndarray 1d
      Contains labels.
      Assumes true and pred contains the same set of distinct labels.

    pred : np.ndarray 1d
      A discrete vector of noisy labels, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    Returns
    -------
    confusion_matrix : np.ndarray (2D)
      matrix of confusion counts with true on rows and pred on columns."""

    assert len(true) == len(pred)
    true_classes = np.unique(true)
    pred_classes = np.unique(pred)
    K_true = len(true_classes)  # Number of classes in true
    K_pred = len(pred_classes)  # Number of classes in pred
    map_true = dict(zip(true_classes, range(K_true)))
    map_pred = dict(zip(pred_classes, range(K_pred)))

    result = np.zeros((K_true, K_pred))
    for i in range(len(true)):
        result[map_true[true[i]]][map_pred[pred[i]]] += 1

    return result