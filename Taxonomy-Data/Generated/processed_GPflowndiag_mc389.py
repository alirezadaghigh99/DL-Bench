import tensorflow as tf
from typing import Callable, Iterable, Union, Optional
from gpflow.base import TensorType
from check_shapes import check_shapes

@check_shapes(
    "Fmu: [N, Din]",
    "Fvar: [N, Din]",
    "Ys.values(): [broadcast N, .]",
    "return: [broadcast n_funs, N, P]",
)
def ndiag_mc(
    funcs: Union[Callable[..., tf.Tensor], Iterable[Callable[..., tf.Tensor]]],
    S: int,
    Fmu: TensorType,
    Fvar: TensorType,
    logspace: bool = False,
    epsilon: Optional[TensorType] = None,
    **Ys: TensorType,
) -> tf.Tensor:
    """
    Computes N Gaussian expectation integrals of one or more functions
    using Monte Carlo samples. The Gaussians must be independent.

    `Fmu`, `Fvar`, `Ys` should all have same shape, with overall size `N`.

    :param funcs: the integrand(s):
        Callable or Iterable of Callables that operates elementwise
    :param S: number of Monte Carlo sampling points
    :param Fmu: array/tensor
    :param Fvar: array/tensor
    :param logspace: if True, funcs are the log-integrands and this calculates
        the log-expectation of exp(funcs)
    :param Ys: arrays/tensors; deterministic arguments to be passed by name
    :return: shape is the same as that of the first Fmu
    """
    if not isinstance(funcs, Iterable):
        funcs = [funcs]

    N, Din = tf.shape(Fmu)
    if epsilon is None:
        epsilon = tf.random.normal((S, N, Din), dtype=Fmu.dtype)

    # Sample from the Gaussian distribution
    samples = Fmu[None, :, :] + tf.sqrt(Fvar)[None, :, :] * epsilon

    # Evaluate each function on the samples
    results = []
    for func in funcs:
        func_values = func(samples, **{k: v[None, ...] for k, v in Ys.items()})
        if logspace:
            # Compute log-expectation of exp(func)
            log_mean = tf.reduce_logsumexp(func_values, axis=0) - tf.math.log(float(S))
            results.append(log_mean)
        else:
            # Compute expectation
            mean = tf.reduce_mean(func_values, axis=0)
            results.append(mean)

    # Stack results for multiple functions
    return tf.stack(results, axis=0)