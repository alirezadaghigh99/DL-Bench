from typing import Callable, List, TypeVar
import numpy as np
import nncf
from nncf.common.utils.backend import BackendType
TTensor = TypeVar('TTensor')

def create_normalized_mse_func(backend: BackendType) -> Callable[[List[TTensor], List[TTensor]], float]:
    """
    Factory method to create backend-specific implementation of the normalized_nmse.

    :param backend: A backend type.
    :return: The backend-specific implementation of the normalized_nmse.
    """
    if backend == BackendType.OPENVINO:
        return normalized_mse
    raise nncf.UnsupportedBackendError(f'Could not create backend-specific implementation! {backend} backend is not supported!')

def normalized_mse(ref_outputs: List[np.ndarray], approx_outputs: List[np.ndarray]) -> float:
    """Create a Python function `normalized_mse` that computes the normalized mean square error (NMSE) between two lists of NumPy arrays, `ref_outputs` and `approx_outputs`. The NMSE is defined as the mean square error (MSE) between the reference and approximate outputs, normalized by the MSE between the reference output and zero. The function iterates over corresponding elements in `ref_outputs` and `approx_outputs`, computes the NMSE for each pair, and then returns the average NMSE across all pairs as a single float value."""
    nmse = 0.0
    for ref_output, approx_output in zip(ref_outputs, approx_outputs):
        mse = np.mean((ref_output - approx_output) ** 2)
        norm = np.mean(ref_output ** 2)
        nmse += mse / norm
    return nmse / len(ref_outputs)
