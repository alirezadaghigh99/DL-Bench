import numpy as np

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    Convert a 3x3 rotation matrix to a quaternion (w, x, y, z).

    Parameters:
    - rotation_matrix: A numpy array of shape (..., 3, 3) representing the rotation matrix.
    - eps: A small value to avoid division by zero.

    Returns:
    - A numpy array of shape (..., 4) representing the quaternion in (w, x, y, z) format.
    """
    # Validate input
    if not isinstance(rotation_matrix, np.ndarray):
        raise TypeError("Input rotation_matrix must be a numpy array.")
    
    if rotation_matrix.shape[-2:] != (3, 3):
        raise ValueError("Input rotation_matrix must have shape (..., 3, 3).")
    
    # Prepare output array
    quaternions = np.zeros(rotation_matrix.shape[:-2] + (4,))
    
    # Iterate over the input array if it has more than 2 dimensions
    it = np.nditer(rotation_matrix[..., 0, 0], flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        R = rotation_matrix[idx]
        
        # Calculate the trace of the matrix
        trace = np.trace(R)
        
        if trace > eps:
            s = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        
        quaternions[idx] = np.array([w, x, y, z])
    
    return quaternions

