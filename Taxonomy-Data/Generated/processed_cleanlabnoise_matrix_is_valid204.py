import numpy as np

def noise_matrix_is_valid(noise_matrix: np.ndarray, py: np.ndarray, verbose: bool = False) -> bool:
    """
    Check if the given noise_matrix is a learnable matrix.

    Parameters:
    - noise_matrix (np.ndarray): A square matrix where element (i, j) represents the probability
                                 of class i being flipped to class j.
    - py (np.ndarray): A 1D array representing the prior probabilities of each class.
    - verbose (bool): If True, print detailed information about the checks.

    Returns:
    - bool: True if the noise matrix is learnable, False otherwise.
    """
    # Check if the noise matrix is a square matrix
    if noise_matrix.shape[0] != noise_matrix.shape[1]:
        if verbose:
            print("Noise matrix is not square.")
        return False

    # Check if the noise matrix is a valid stochastic matrix
    if not np.allclose(noise_matrix.sum(axis=1), 1):
        if verbose:
            print("Rows of the noise matrix do not sum to 1.")
        return False

    if np.any(noise_matrix < 0):
        if verbose:
            print("Noise matrix contains negative probabilities.")
        return False

    # Check if the diagonal elements are sufficiently large
    # This ensures that the true class is more likely than any other class
    diagonal_elements = np.diag(noise_matrix)
    if np.any(diagonal_elements <= 0.5):
        if verbose:
            print("One or more diagonal elements are not greater than 0.5.")
        return False

    # Check if the prior probabilities are valid
    if not np.isclose(py.sum(), 1):
        if verbose:
            print("Prior probabilities do not sum to 1.")
        return False

    if np.any(py < 0):
        if verbose:
            print("Prior probabilities contain negative values.")
        return False

    # If all checks pass, the noise matrix is considered learnable
    if verbose:
        print("The noise matrix is learnable.")
    return True