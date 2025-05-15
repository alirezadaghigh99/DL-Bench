import numpy as np

def flatten_and_preprocess_masks(labels, pred_probs):
    """
    Flattens a 2D/3D labels array and reshapes a 4D pred_probs array so that
    the channels align with the flattened labels.

    Parameters:
    - labels: np.ndarray, 2D or 3D array of labels.
    - pred_probs: np.ndarray, 4D array of predicted probabilities.

    Returns:
    - flattened_labels: np.ndarray, 1D array of flattened labels.
    - transposed_flattened_pred_probs: np.ndarray, 2D array of transposed and flattened predicted probabilities.
    """
    # Flatten the labels array
    if labels.ndim == 2:
        flattened_labels = labels.flatten()
    elif labels.ndim == 3:
        # Assuming the first dimension is the number of masks or classes
        flattened_labels = labels.reshape(-1)
    else:
        raise ValueError("Labels array must be 2D or 3D.")

    # Reshape and transpose the pred_probs array
    if pred_probs.ndim != 4:
        raise ValueError("pred_probs array must be 4D.")
    
    # Transpose to bring the channels to the first dimension
    # Shape: (num_classes, batch_size * height * width)
    transposed_pred_probs = np.transpose(pred_probs, (3, 0, 1, 2))
    
    # Flatten the transposed pred_probs
    transposed_flattened_pred_probs = transposed_pred_probs.reshape(pred_probs.shape[3], -1)

    return flattened_labels, transposed_flattened_pred_probs

