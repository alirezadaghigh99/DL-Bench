def expand_image_array_cnn(image_arr: np.ndarray) -> np.ndarray:
    """
    Checks the sanity of the input image numpy array for cnn and converts the grayscale numpy array to rgb by repeating
    the array thrice along the 3rd dimension if a 2-dimensional image array is provided.

    Args:
        image_arr: Image array.

    Returns:
        A 3-dimensional numpy image array.
    """
    image_arr_shape = image_arr.shape
    if len(image_arr_shape) == 3:
        _check_3_dim(image_arr_shape)
        return image_arr
    elif len(image_arr_shape) == 2:
        image_arr_3dim = _add_third_dim(image_arr)
        return image_arr_3dim
    else:
        _raise_wrong_dim_value_error(image_arr_shape)