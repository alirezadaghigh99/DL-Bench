import numpy as np
from sklearn.preprocessing import Binarizer

def prediction_result(probability_matrix, multi_label, num_classes, return_proba=False, deprecated_arg=None):
    """
    Generate predictions or probabilities from a probability matrix.

    Parameters:
    - probability_matrix: np.ndarray, the matrix of probabilities.
    - multi_label: bool, indicates if the task is multi-label classification.
    - num_classes: int, the number of classes.
    - return_proba: bool, whether to return the probability matrix.
    - deprecated_arg: any, a deprecated argument that is ignored.

    Returns:
    - np.ndarray or list: predictions or probabilities based on the input parameters.
    """
    if deprecated_arg is not None:
        print("Warning: 'deprecated_arg' is deprecated and will be ignored.")

    if multi_label:
        # Binarize the probability matrix for multi-label classification
        binarizer = Binarizer(threshold=0.5)
        predictions = binarizer.fit_transform(probability_matrix)
        predictions = predictions.astype(int).tolist()  # Convert to list format
    else:
        # For single-label, take the argmax for each sample
        predictions = np.argmax(probability_matrix, axis=1)

    if return_proba:
        return predictions, probability_matrix
    else:
        return predictions

