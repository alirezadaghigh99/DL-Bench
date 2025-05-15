import numpy as np

def compute_pyx(pred_probs, noise_matrix, inverse_noise_matrix):
    """
    Compute P(true_label=k|x) from P(label=k|x), noise_matrix, and inverse_noise_matrix.

    Parameters
    ----------
    pred_probs : np.ndarray
        (N x K) matrix with model-predicted probabilities P(label=k|x).

    noise_matrix : np.ndarray
        (K x K) matrix P(label=k_s|true_label=k_y).

    inverse_noise_matrix : np.ndarray
        (K x K) matrix P(true_label=k_y|label=k_s).

    Returns
    -------
    pyx : np.ndarray
        (N x K) matrix of model-predicted probabilities P(true_label=k|x).
    """
    # Validate input shapes
    if len(np.shape(pred_probs)) != 2:
        raise ValueError(
            "Input parameter np.ndarray 'pred_probs' has shape "
            + str(np.shape(pred_probs))
            + ", but shape should be (N, K)"
        )
    
    N, K = pred_probs.shape

    # Validate noise matrices
    if noise_matrix.shape != (K, K):
        raise ValueError("noise_matrix must be of shape (K, K)")
    if inverse_noise_matrix.shape != (K, K):
        raise ValueError("inverse_noise_matrix must be of shape (K, K)")

    # Compute the diagonal of the noise matrix
    noise_diag = np.diag(noise_matrix)

    # Compute the diagonal of the inverse noise matrix
    inverse_noise_diag = np.diag(inverse_noise_matrix)

    # Adjust the predicted probabilities using the diagonals of the noise matrices
    adjusted_probs = pred_probs * inverse_noise_diag / noise_diag

    # Normalize the adjusted probabilities to ensure they sum to 1 for each example
    pyx = adjusted_probs / np.sum(adjusted_probs, axis=1, keepdims=True)

    return pyx