"""
This module provides functions to convert from a list representation of multi-fidelity data to an array representation.

The list based representation is a list of numpy arrays, with a numpy array for every fidelity. The list is ordered
from the lowest fidelity to the highest fidelity.

The array representation is one array for all inputs where the last column of the X array is a zero-based index
indicating the fidelity.
"""
from typing import List, Tuple
import numpy as np

def convert_x_list_to_array(x_list: List) -> np.ndarray:
    """
    Converts list representation of features to array representation
    :param x_list: A list of (n_points x n_dims) numpy arrays ordered from lowest to highest fidelity
    :return: An array of all features with the zero-based fidelity index appended as the last column
    """
    if not np.all([x.ndim == 2 for x in x_list]):
        raise ValueError('All x arrays must have 2 dimensions')
    x_array = np.concatenate(x_list, axis=0)
    indices = []
    for (i, x) in enumerate(x_list):
        indices.append(i * np.ones((len(x), 1)))
    x_with_index = np.concatenate((x_array, np.concatenate(indices)), axis=1)
    return x_with_index

def convert_y_list_to_array(y_list: List) -> np.ndarray:
    """
    Converts list representation of outputs to array representation
    :param y_list: A list of (n_points x n_outputs) numpy arrays representing the outputs
                   ordered from lowest to highest fidelity
    :return: An array of all outputs
    """
    if not np.all([y.ndim == 2 for y in y_list]):
        raise ValueError('All y arrays must have 2 dimensions')
    return np.concatenate(y_list, axis=0)

def convert_xy_lists_to_arrays(x_list: List, y_list: List) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a Python function called convert_xy_lists_to_arrays that takes in two input parameters: x_list and y_list, both of type List. The function returns a Tuple containing two numpy arrays: x_array and y_array. 

The x_list parameter is a list of numpy arrays representing inputs, ordered from lowest to highest fidelity. The y_list parameter is a list of numpy arrays representing outputs, also ordered from lowest to highest fidelity. 

The function first checks if the lengths of x_list and y_list are equal, raising a ValueError if they are not. It then checks if the number of points in each fidelity level is the same for both x_list and y_list, raising a ValueError if they are not. 

The x_array returned contains all inputs across all fidelities with the fidelity index appended as the last column. The y_array returned contains all outputs across all fidelities. 

If the function encounters any errors during the conversion process, it will raise appropriate ValueErrors."""
    if len(x_list) != len(y_list):
        raise ValueError('Different numbers of fidelities between x_list and y_list ({} vs {})'
                          .format(len(x_list), len(y_list)))

    for i, (x, y) in enumerate(zip(x_list, y_list)):
        if len(x) != len(y):
            raise ValueError('Different number of points at fidelity {} for x and y ({} vs {})'
                              .format(i, len(x), len(y)))

    x_array = convert_x_list_to_array(x_list)
    y_array = convert_y_list_to_array(y_list)
    return x_array, y_array
