import numpy as np

def _walsh_hadamard_transform(D, n=None):
    """
    Compute the Walsh-Hadamard Transform of a one-dimensional tensor or array D.
    
    Parameters:
    - D: A one-dimensional numpy array or tensor. Its length must be a power of two.
    - n: An optional integer representing the number of qubits or the size of the transform.
         If not provided, it will be calculated based on the length of D.
    
    Returns:
    - A numpy array of the same shape as D, transformed by the Walsh-Hadamard Transform.
    """
    # Ensure D is a numpy array
    D = np.asarray(D)
    
    # Determine the size of the input
    length = D.shape[0]
    
    # Check if the length of D is a power of two
    if (length & (length - 1)) != 0:
        raise ValueError("The length of D must be a power of two.")
    
    # Calculate n if not provided
    if n is None:
        n = int(np.log2(length))
    
    # Check if n is consistent with the length of D
    if 2**n != length:
        raise ValueError("The length of D must be 2^n for the given n.")
    
    # Define the Hadamard matrix for a single qubit
    H_1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    # Construct the full Hadamard matrix using the Kronecker product
    H_n = H_1
    for _ in range(n - 1):
        H_n = np.kron(H_n, H_1)
    
    # Apply the Hadamard transform
    transformed_D = np.dot(H_n, D)
    
    return transformed_D

