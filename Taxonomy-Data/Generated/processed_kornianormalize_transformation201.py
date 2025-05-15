import numpy as np

def normalize_transformation(M, eps=1e-8):
    # Ensure the input is a numpy array for easier manipulation
    M = np.array(M)
    
    # Check that the input matrix has at least 2 dimensions
    assert M.ndim >= 2, "Input matrix must have at least 2 dimensions"
    
    # Get the shape of the matrix
    rows, cols = M.shape
    
    # Check that the matrix is at least 2x2
    assert rows >= 2 and cols >= 2, "Input matrix must be at least 2x2 in size"
    
    # Get the value at the last row and column
    last_value = M[-1, -1]
    
    # Avoid division by zero by ensuring last_value is not too close to zero
    if abs(last_value) < eps:
        raise ValueError("The value at the last row and column is too close to zero, cannot normalize.")
    
    # Normalize the matrix by dividing all elements by the last element
    normalized_M = M / last_value
    
    # Return the normalized matrix
    return normalized_M

