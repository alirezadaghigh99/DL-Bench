import numpy as np

def softmax(x: np.ndarray, temperature: float = 1.0, axis: int = None, shift: bool = True) -> np.ndarray:
    """
    Apply the softmax function to an input array.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    temperature : float
        Temperature of the softmax function.
    axis : Optional[int]
        Axis to apply the softmax function. If None, the softmax function is
        applied to all elements of the input array.
    shift : bool
        Whether to shift the input array before applying the softmax function.
        This is useful to avoid numerical issues when the input array contains
        large values, that could result in overflows when applying the exponential
        function.

    Returns
    -------
    np.ndarray
        Softmax function applied to the input array.
    """
    if shift:
        # Subtract the maximum value along the specified axis for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        x = x - x_max

    # Apply the exponential function with temperature scaling
    exp_x = np.exp(x / temperature)

    # Sum of exponentials along the specified axis
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)

    # Compute the softmax values
    softmax_x = exp_x / sum_exp_x

    return softmax_x