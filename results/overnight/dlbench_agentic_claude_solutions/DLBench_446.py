from typing import Optional
import numpy as np
from cleanlab.internal.constants import EPSILON

def softmax(x: np.ndarray, temperature: float=1.0, axis: Optional[int]=None, shift: bool=False) -> np.ndarray:
    """Write a python function Softmax function.

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

The softmax function normalizes the input array by applying the exponential function to each element and dividing by the sum of all exponential values. The temperature parameter can be used to adjust the sensitivity of the softmax function. If shift is set to True, the input array is shifted to avoid numerical issues."""
    x = np.asarray(x, dtype=float) / temperature
    if shift:
        x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
