import numpy as np

def convert_x_list_to_array(x_list):
    """
    Convert a list of numpy arrays into a single numpy array with an appended fidelity index.

    Parameters:
    x_list (list of np.ndarray): A list of numpy arrays with dimensions (n_points x n_dims).

    Returns:
    np.ndarray: A numpy array with features and fidelity index concatenated.

    Raises:
    ValueError: If any of the arrays in x_list do not have 2 dimensions.
    """
    if not all(isinstance(x, np.ndarray) and x.ndim == 2 for x in x_list):
        raise ValueError("All x arrays must have 2 dimensions")

    # List to hold arrays with appended fidelity index
    arrays_with_fidelity = []

    # Iterate over the list of arrays and their indices
    for fidelity_index, x in enumerate(x_list):
        # Create a column of the fidelity index
        fidelity_column = np.full((x.shape[0], 1), fidelity_index)
        # Append the fidelity index column to the array
        x_with_fidelity = np.hstack((x, fidelity_column))
        # Add the modified array to the list
        arrays_with_fidelity.append(x_with_fidelity)

    # Concatenate all arrays along the first axis (rows)
    result_array = np.vstack(arrays_with_fidelity)

    return result_array

