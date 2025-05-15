import numpy as np

def zca_mean(inp, dim=0, unbiased=True, eps=1e-5, return_inverse=False):
    """
    Computes the ZCA whitening matrix and mean vector for a given input tensor.

    Parameters:
    - inp: np.ndarray, the input tensor.
    - dim: int, the dimension along which the samples are located.
    - unbiased: bool, whether to use the unbiased estimate of the covariance matrix.
    - eps: float, a small value for numerical stability.
    - return_inverse: bool, whether to return the inverse ZCA transform.

    Returns:
    - A tuple containing the ZCA matrix, the mean vector, and optionally the inverse ZCA matrix.
    """
    # Validate input types
    if not isinstance(inp, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    if not isinstance(dim, int):
        raise TypeError("Dimension must be an integer.")
    if not isinstance(unbiased, bool):
        raise TypeError("Unbiased must be a boolean.")
    if not isinstance(eps, (float, int)):
        raise TypeError("Eps must be a float or int.")
    if not isinstance(return_inverse, bool):
        raise TypeError("Return_inverse must be a boolean.")

    # Validate input dimensions
    if dim < 0 or dim >= inp.ndim:
        raise ValueError("Dimension is out of bounds for the input tensor.")

    # Move the samples dimension to the first axis
    inp = np.moveaxis(inp, dim, 0)
    num_samples = inp.shape[0]

    # Compute the mean vector
    mean_vector = np.mean(inp, axis=0)

    # Center the data
    centered_data = inp - mean_vector

    # Compute the covariance matrix
    if unbiased:
        cov_matrix = np.dot(centered_data.T, centered_data) / (num_samples - 1)
    else:
        cov_matrix = np.dot(centered_data.T, centered_data) / num_samples

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Add eps for numerical stability
    eigvals = np.maximum(eigvals, eps)

    # Compute the ZCA whitening matrix
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    zca_matrix = eigvecs @ D_inv_sqrt @ eigvecs.T

    # Optionally compute the inverse ZCA matrix
    if return_inverse:
        D_sqrt = np.diag(np.sqrt(eigvals))
        zca_inverse_matrix = eigvecs @ D_sqrt @ eigvecs.T
        return zca_matrix, mean_vector, zca_inverse_matrix

    return zca_matrix, mean_vector

