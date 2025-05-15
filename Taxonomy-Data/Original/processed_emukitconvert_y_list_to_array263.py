def convert_y_list_to_array(y_list: List) -> np.ndarray:
    """
    Converts list representation of outputs to array representation
    :param y_list: A list of (n_points x n_outputs) numpy arrays representing the outputs
                   ordered from lowest to highest fidelity
    :return: An array of all outputs
    """
    if not np.all([y.ndim == 2 for y in y_list]):
        raise ValueError("All y arrays must have 2 dimensions")
    return np.concatenate(y_list, axis=0)