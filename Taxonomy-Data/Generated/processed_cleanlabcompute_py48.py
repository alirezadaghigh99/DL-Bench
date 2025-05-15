import numpy as np

def compute_py(ps, noise_matrix, inverse_noise_matrix, py_method, true_labels_class_counts):
    """
    Compute the prior probability of true class labels based on observed noisy labels,
    noise matrices, and inverse noise matrices.

    Parameters:
    - ps: array-like, shape (K,) or (1, K), observed noisy label distribution.
    - noise_matrix: array-like, shape (K, K), noise matrix.
    - inverse_noise_matrix: array-like, shape (K, K), inverse of the noise matrix.
    - py_method: str, method to compute prior probabilities ('cnt', 'eqn', 'marginal', 'marginal_ps').
    - true_labels_class_counts: array-like, shape (K,), counts of true class labels.

    Returns:
    - py: array, shape (K,) or (1, K), prior probability of each true class label.
    """
    # Validate input shapes
    if ps is None or noise_matrix is None or inverse_noise_matrix is None or true_labels_class_counts is None:
        raise ValueError("All input parameters must be provided.")
    
    ps = np.asarray(ps)
    noise_matrix = np.asarray(noise_matrix)
    inverse_noise_matrix = np.asarray(inverse_noise_matrix)
    true_labels_class_counts = np.asarray(true_labels_class_counts)

    K = noise_matrix.shape[0]

    if ps.shape not in [(K,), (1, K)]:
        raise ValueError(f"ps must have shape ({K},) or (1, {K}).")
    if noise_matrix.shape != (K, K):
        raise ValueError(f"noise_matrix must have shape ({K}, {K}).")
    if inverse_noise_matrix.shape != (K, K):
        raise ValueError(f"inverse_noise_matrix must have shape ({K}, {K}).")
    if true_labels_class_counts.shape != (K,):
        raise ValueError(f"true_labels_class_counts must have shape ({K},).")

    # Compute py based on the specified method
    if py_method == 'cnt':
        # Use true label counts to compute prior probabilities
        py = true_labels_class_counts / np.sum(true_labels_class_counts)
    elif py_method == 'eqn':
        # Use the equation method: py = inverse_noise_matrix * ps
        py = np.dot(inverse_noise_matrix, ps)
    elif py_method == 'marginal':
        # Use marginal method: py = noise_matrix.T * ps
        py = np.dot(noise_matrix.T, ps)
    elif py_method == 'marginal_ps':
        # Use marginal_ps method: py = ps
        py = ps
    else:
        raise ValueError(f"Invalid py_method: {py_method}. Choose from 'cnt', 'eqn', 'marginal', 'marginal_ps'.")

    # Ensure py is a 1D array
    py = np.squeeze(py)

    # Clip py to ensure values are between 0 and 1
    py = np.clip(py, 0, 1)

    return py

