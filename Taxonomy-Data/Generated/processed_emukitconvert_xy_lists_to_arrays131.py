import numpy as np
from typing import List, Tuple

def convert_xy_lists_to_arrays(x_list: List[np.ndarray], y_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    # Check if the lengths of x_list and y_list are equal
    if len(x_list) != len(y_list):
        raise ValueError("The lengths of x_list and y_list must be equal.")
    
    # Initialize lists to hold the modified x arrays and y arrays
    x_arrays = []
    y_arrays = []
    
    # Iterate over each fidelity level
    for fidelity_index, (x, y) in enumerate(zip(x_list, y_list)):
        # Check if the number of points in x and y are the same
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"The number of points in x_list[{fidelity_index}] and y_list[{fidelity_index}] must be the same.")
        
        # Append the fidelity index as the last column to the x array
        fidelity_column = np.full((x.shape[0], 1), fidelity_index)
        x_with_fidelity = np.hstack((x, fidelity_column))
        
        # Append the modified x array and y array to the lists
        x_arrays.append(x_with_fidelity)
        y_arrays.append(y)
    
    # Concatenate all the x arrays and y arrays
    x_array = np.vstack(x_arrays)
    y_array = np.vstack(y_arrays)
    
    return x_array, y_array