def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor.

    Parameters
    ----------
    data : list or numpy array
        The input data to be summed.
    segment_ids : list or numpy array
        The segment IDs for each element in data.
    num_segments : int
        The number of distinct segment IDs.

    Returns
    -------
    list
        A list where each element is the sum of the elements in the corresponding segment.
    """
    # Initialize the result list with zeros
    result = [0] * num_segments
    
    # Iterate over each element in the data
    for i, value in enumerate(data):
        # Add the value to the corresponding segment
        result[segment_ids[i]] += value
    
    return result

