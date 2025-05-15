import numpy as np

def log_loss(y_true, y_pred, normalize=True, sample_weight=None, labels=None):
    """
    Calculate the log loss, also known as logistic loss or cross-entropy loss.

    Parameters:
    - y_true: array-like of shape (n_samples,) - Ground truth labels.
    - y_pred: array-like of shape (n_samples, n_classes) - Predicted probabilities.
    - normalize: bool, default=True - If True, return the mean loss per sample. Otherwise, return the sum of per-sample losses.
    - sample_weight: array-like of shape (n_samples,), default=None - Optional sample weights.
    - labels: array-like, default=None - Optional labels for the classes.

    Returns:
    - float - The calculated log loss.

    Examples:
    >>> y_true = [0, 1, 1]
    >>> y_pred = [[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]]
    >>> log_loss(y_true, y_pred)
    0.1738073366910675

    Notes:
    - Log loss is a measure of the performance of a classification model where the prediction is a probability value between 0 and 1.
    - The function assumes that y_pred contains probabilities for each class and that the sum of probabilities for each sample is 1.

    References:
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    - https://en.wikipedia.org/wiki/Cross_entropy
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if labels is not None:
        labels = np.array(labels)
        y_true = np.searchsorted(labels, y_true)

    if sample_weight is None:
        sample_weight = np.ones(y_true.shape[0])

    # Clip y_pred to prevent log(0)
    eps = np.finfo(float).eps
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Calculate log loss
    n_samples = y_true.shape[0]
    n_classes = y_pred.shape[1]

    # Create a one-hot encoded matrix for y_true
    y_true_one_hot = np.zeros((n_samples, n_classes))
    y_true_one_hot[np.arange(n_samples), y_true] = 1

    # Calculate the log loss
    loss = -np.sum(y_true_one_hot * np.log(y_pred) * sample_weight[:, np.newaxis])

    if normalize:
        return loss / np.sum(sample_weight)
    else:
        return loss

