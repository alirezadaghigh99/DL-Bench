import numpy as np

def matrix_cofactor_tensor(matrix):
    """
    Computes the cofactor matrix for a given tensor of 3x3 matrices.

    Parameters:
    - matrix: A tensor of shape (*, 3, 3) representing the input matrices.

    Returns:
    - A tensor containing the cofactor matrices of the input matrices.

    Raises:
    - Exception if all input matrices are singular (i.e., their determinants are zero).
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Input tensor must have shape (*, 3, 3)")

    # Calculate the determinant of each 3x3 matrix
    dets = np.linalg.det(matrix)
    
    # Check if all matrices are singular
    if np.all(dets == 0):
        raise Exception("All input matrices are singular (determinants are zero).")

    # Prepare an array to store the cofactor matrices
    cofactor_matrices = np.empty_like(matrix)

    # Iterate over the first dimensions (batch dimensions)
    for index in np.ndindex(matrix.shape[:-2]):
        mat = matrix[index]
        cofactor = np.empty((3, 3))

        # Calculate the cofactor matrix
        for i in range(3):
            for j in range(3):
                # Create the minor matrix by removing the i-th row and j-th column
                minor = np.delete(np.delete(mat, i, axis=0), j, axis=1)
                # Calculate the cofactor value
                cofactor[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)

        cofactor_matrices[index] = cofactor

    return cofactor_matrices

