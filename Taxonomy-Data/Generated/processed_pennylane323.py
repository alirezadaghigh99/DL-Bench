import numpy as np
import torch
import tensorflow as tf

def cast(tensor, dtype):
    """
    Args:
        tensor (tensor_like): tensor to cast
        dtype (str, np.dtype): Any supported NumPy dtype representation; this can be
            a string (``"float64"``), a ``np.dtype`` object (``np.dtype("float64")``), or
            a dtype class (``np.float64``). If ``tensor`` is not a NumPy array, the
            **equivalent** dtype in the dispatched framework is used.

    Returns:
        tensor_like: a tensor with the same shape and values as ``tensor`` and the
        same dtype as ``dtype``
    """
    # Convert dtype to a string if it's not already
    if not isinstance(dtype, str):
        try:
            dtype = np.dtype(dtype).name
        except (AttributeError, TypeError, ImportError):
            dtype = getattr(dtype, "name", dtype)

    # Handle different tensor types
    if isinstance(tensor, (list, tuple, int, float, complex)):
        tensor = np.asarray(tensor)

    if isinstance(tensor, np.ndarray):
        return tensor.astype(dtype)

    elif isinstance(tensor, torch.Tensor):
        # Map NumPy dtype to PyTorch dtype
        torch_dtype = getattr(torch, dtype, None)
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype for PyTorch: {dtype}")
        return tensor.to(torch_dtype)

    elif isinstance(tensor, (tf.Tensor, tf.Variable)):
        # Map NumPy dtype to TensorFlow dtype
        tf_dtype = getattr(tf, dtype, None)
        if tf_dtype is None:
            raise ValueError(f"Unsupported dtype for TensorFlow: {dtype}")
        return tf.cast(tensor, tf_dtype)

    else:
        raise TypeError("Unsupported tensor type")

