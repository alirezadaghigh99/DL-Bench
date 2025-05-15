def convert_x_list_to_array(x_list: List) -> np.ndarray:
    """
    Converts list representation of features to array representation
    :param x_list: A list of (n_points x n_dims) numpy arrays ordered from lowest to highest fidelity
    :return: An array of all features with the zero-based fidelity index appended as the last column
    """
    # First check everything is a 2d array
    if not np.all([x.ndim == 2 for x in x_list]):
        raise ValueError("All x arrays must have 2 dimensions")

    x_array = np.concatenate(x_list, axis=0)
    indices = []
    for i, x in enumerate(x_list):
        indices.append(i * np.ones((len(x), 1)))

    x_with_index = np.concatenate((x_array, np.concatenate(indices)), axis=1)
    return x_with_index