def _min_adj(bits, low, range_len, narrow_range):
    """
    Calculate the minimum adjustment value based on the input parameters.

    Parameters:
    - bits (int): The number of bits used for quantization.
    - low (float): The lower bound of the range.
    - range_len (float): The length of the range.
    - narrow_range (bool): A boolean indicating whether the range is narrow.

    Returns:
    - float: The minimum adjustment value.
    """
    # Calculate the number of quantization levels
    if narrow_range:
        quants_count = (1 << bits) - 1  # 2^bits - 1
    else:
        quants_count = 1 << bits  # 2^bits

    # Calculate the minimum adjustment value
    min_adj = range_len / quants_count

    return min_adj

