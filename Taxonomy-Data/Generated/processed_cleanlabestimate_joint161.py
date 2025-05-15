import numpy as np

def estimate_joint(labels, pred_probs, confident_joint=None, multi_label=False):
    """
    Estimate the joint distribution of label noise P(label=i, true_label=j).

    Parameters:
    - labels: array-like, shape (n_samples,)
        Class labels for each example in the dataset.
    - pred_probs: array-like, shape (n_samples, n_classes)
        Model-predicted class probabilities for each example in the dataset.
    - confident_joint: array-like, optional
        Estimated class label error statistics.
    - multi_label: bool
        Indicates whether the dataset is for multi-class or multi-label classification.

    Returns:
    - confident_joint_distribution: array
        An estimate of the true joint distribution of noisy and true labels.
    """
    labels = np.array(labels)
    pred_probs = np.array(pred_probs)
    n_samples, n_classes = pred_probs.shape

    if confident_joint is None:
        # Initialize the confident joint matrix
        confident_joint = np.zeros((n_classes, n_classes), dtype=float)

        # Compute the confident joint
        for i in range(n_samples):
            true_label = labels[i]
            predicted_label = np.argmax(pred_probs[i])
            confident_joint[predicted_label, true_label] += 1

    # Normalize the confident joint to get a distribution
    confident_joint_distribution = confident_joint / np.sum(confident_joint)

    if multi_label:
        return _estimate_joint_multilabel(labels, pred_probs, confident_joint_distribution)
    else:
        return confident_joint_distribution

def _estimate_joint_multilabel(labels, pred_probs, confident_joint_distribution):
    """
    Estimate the joint distribution for multi-label classification.

    Parameters:
    - labels: array-like, shape (n_samples, n_classes)
        Binary matrix indicating the presence of each class label for each example.
    - pred_probs: array-like, shape (n_samples, n_classes)
        Model-predicted class probabilities for each example in the dataset.
    - confident_joint_distribution: array-like
        Initial confident joint distribution.

    Returns:
    - joint_distribution: array
        An estimate of the true joint distribution of noisy and true labels for multi-label data.
    """
    n_samples, n_classes = pred_probs.shape
    joint_distribution = np.zeros((n_classes, 2, 2), dtype=float)

    for k in range(n_classes):
        for i in range(n_samples):
            true_label = labels[i, k]
            predicted_label = pred_probs[i, k] > 0.5  # Thresholding at 0.5 for binary decision
            joint_distribution[k, int(predicted_label), int(true_label)] += 1

    # Normalize the joint distribution
    for k in range(n_classes):
        joint_distribution[k] /= np.sum(joint_distribution[k])

    return joint_distribution