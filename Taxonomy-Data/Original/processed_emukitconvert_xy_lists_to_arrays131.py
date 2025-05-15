def convert_xy_lists_to_arrays(x_list: List, y_list: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts list representation of targets to array representation
    :param x_list: A list of (n_points x n_dims) numpy arrays ordered from lowest to highest fidelity
    :param y_list: A list of (n_points x n_outputs) numpy arrays representing the outputs
                   ordered from lowest to highest fidelity
    :return: Tuple of (x_array, y_array) where
             x_array contains all inputs across all fidelities with the fidelity index appended as the last column
             and y_array contains all outputs across all fidelities.
    """

    if len(x_list) != len(y_list):
        raise ValueError("Different number of fidelities between x and y")

    # Check same number of points in each fidelity
    n_points_x = np.array([x.shape[0] for x in x_list])
    n_points_y = np.array([y.shape[0] for y in y_list])
    if not np.all(n_points_x == n_points_y):
        raise ValueError("Different number of points in x and y at the same fidelity")

    return convert_x_list_to_array(x_list), convert_y_list_to_array(y_list)