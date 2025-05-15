import warnings
from typing import Callable, Iterable, Union, Tuple, List
import tensorflow as tf
from gpflow.quadrature import NDiagGHQuadrature
from gpflow.base import TensorType

def ndiagquad(
    funcs: Union[Callable[..., tf.Tensor], Iterable[Callable[..., tf.Tensor]]],
    H: int,
    Fmu: Union[TensorType, Tuple[TensorType, ...], List[TensorType]],
    Fvar: Union[TensorType, Tuple[TensorType, ...], List[TensorType]],
    logspace: bool = False,
    **Ys: TensorType,
) -> tf.Tensor:
    """
    Compute N Gaussian expectation integrals using Gauss-Hermite quadrature.
    
    :param funcs: A callable or an iterable of callables representing the integrands.
    :param H: Number of Gauss-Hermite quadrature points.
    :param Fmu: Means of the Gaussian distributions.
    :param Fvar: Variances of the Gaussian distributions.
    :param logspace: Whether to compute the log-expectation of exp(funcs).
    :param Ys: Additional deterministic inputs to the integrands.
    :return: The computed expectation with the same shape as Fmu.
    """
    # Issue a deprecation warning
    warnings.warn(
        "The ndiagquad function is deprecated. Please use gpflow.quadrature.NDiagGHQuadrature instead.",
        DeprecationWarning
    )
    
    # Ensure funcs is iterable
    if not isinstance(funcs, Iterable):
        funcs = [funcs]
    
    # Ensure Fmu and Fvar are tuples
    if not isinstance(Fmu, (tuple, list)):
        Fmu = (Fmu,)
    if not isinstance(Fvar, (tuple, list)):
        Fvar = (Fvar,)
    
    # Create the quadrature object
    quadrature = NDiagGHQuadrature(H, Fmu, Fvar)
    
    # Define the function to integrate
    def integrand(*X):
        results = []
        for func in funcs:
            if logspace:
                # Compute log-expectation of exp(func)
                result = tf.exp(func(*X, **Ys))
            else:
                # Compute standard expectation
                result = func(*X, **Ys)
            results.append(result)
        return tf.stack(results, axis=0)
    
    # Perform the quadrature
    result = quadrature(integrand)
    
    # Reshape the result to match the input shape
    result_shape = tf.shape(Fmu[0])
    return tf.reshape(result, result_shape)

