import numpy as np

def convert_y_list_to_array(y_list):
    """
    Convert a list of numpy arrays to a single numpy array by concatenating along axis 0.
    
    Parameters:
    y_list (list): A list of numpy arrays, each with 2 dimensions.
    
    Returns:
    numpy.ndarray: A single numpy array containing all the outputs concatenated along axis 0.
    
    Raises:
    ValueError: If any array in y_list does not have 2 dimensions.
    """
    # Check if all arrays in y_list have 2 dimensions
    for i, y in enumerate(y_list):
        if y.ndim != 2:
            raise ValueError(f"All y arrays must have 2 dimensions. Array at index {i} has {y.ndim} dimensions.")
    
    # Concatenate all arrays along axis 0
    return np.concatenate(y_list, axis=0)

