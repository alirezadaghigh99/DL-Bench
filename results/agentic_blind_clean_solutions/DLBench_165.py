import itertools
import warnings
from functools import wraps
from typing import Callable, Iterable, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from check_shapes import check_shapes
from ..base import AnyNDArray, TensorType
from ..config import default_float
from ..utilities import to_default_float
from .gauss_hermite import NDiagGHQuadrature

@check_shapes('return[0]: [n_quad_points]', 'return[1]: [n_quad_points]')
def hermgauss(n: int) -> Tuple[AnyNDArray, AnyNDArray]:
    (x, w) = np.polynomial.hermite.hermgauss(n)
    (x, w) = (x.astype(default_float()), w.astype(default_float()))
    return (x, w)

@check_shapes('return[0]: [n_quad_points, D]', 'return[1]: [n_quad_points]')
def mvhermgauss(H: int, D: int) -> Tuple[AnyNDArray, AnyNDArray]:
    """
    Return the evaluation locations 'xn', and weights 'wn' for a multivariate
    Gauss-Hermite quadrature.

    The outputs can be used to approximate the following type of integral:
    int exp(-x)*f(x) dx ~ sum_i w[i,:]*f(x[i,:])

    :param H: Number of Gauss-Hermite evaluation points.
    :param D: Number of input dimensions. Needs to be known at call-time.
    :return: eval_locations 'x' (H**DxD), weights 'w' (H**D)
    """
    (gh_x, gh_w) = hermgauss(H)
    x: AnyNDArray = np.array(list(itertools.product(*(gh_x,) * D)))
    w = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)
    return (x, w)

@check_shapes('means: [N, Din]', 'covs: [N, Din, Din]', 'return: [N, Dout...]')
def mvnquad(func: Callable[[tf.Tensor], tf.Tensor], means: TensorType, covs: TensorType, H: int, Din: Optional[int]=None, Dout: Optional[Tuple[int, ...]]=None) -> tf.Tensor:
    """
    Computes N Gaussian expectation integrals of a single function 'f'
    using Gauss-Hermite quadrature.

    :param f: integrand function. Takes one input of shape ?xD.
    :param H: Number of Gauss-Hermite evaluation points.
    :param Din: Number of input dimensions. Needs to be known at call-time.
    :param Dout: Number of output dimensions. Defaults to (). Dout is assumed
        to leave out the item index, i.e. f actually maps (?xD)->(?x*Dout).
    :return: quadratures
    """
    if Din is None:
        Din = means.shape[1]
    if Din is None:
        raise ValueError('If `Din` is passed as `None`, `means` must have a known shape. Running mvnquad in `autoflow` without specifying `Din` and `Dout` is problematic. Consider using your own session.')
    (xn, wn) = mvhermgauss(H, Din)
    N = means.shape[0]
    cholXcov = tf.linalg.cholesky(covs)
    Xt = tf.linalg.matmul(cholXcov, tf.tile(xn[None, :, :], (N, 1, 1)), transpose_b=True)
    X = 2.0 ** 0.5 * Xt + tf.expand_dims(means, 2)
    Xr = tf.reshape(tf.transpose(X, [2, 0, 1]), (-1, Din))
    fevals = func(Xr)
    if Dout is None:
        Dout = tuple((d if type(d) is int else d.value for d in fevals.shape[1:]))
    if any([d is None for d in Dout]):
        raise ValueError('If `Dout` is passed as `None`, the output of `func` must have known shape. Running mvnquad in `autoflow` without specifying `Din` and `Dout` is problematic. Consider using your own session.')
    fX = tf.reshape(fevals, (H ** Din, N) + Dout)
    wr = np.reshape(wn * np.pi ** (-Din * 0.5), (-1,) + (1,) * (1 + len(Dout)))
    return tf.reduce_sum(fX * wr, 0)

@check_shapes('Fmu: [broadcast Din, N...]', 'Fvar: [broadcast Din, N...]', 'Ys.values(): [N...]', 'return: [broadcast Dout, N...]')
def ndiagquad(funcs: Union[Callable[..., tf.Tensor], Iterable[Callable[..., tf.Tensor]]], H: int, Fmu: Union[TensorType, Tuple[TensorType, ...], List[TensorType]], Fvar: Union[TensorType, Tuple[TensorType, ...], List[TensorType]], logspace: bool=False, **Ys: TensorType) -> tf.Tensor:
    """Create a Python function `ndiagquad` that computes N Gaussian expectation integrals using Gauss-Hermite quadrature for one or more functions. The function should accept the following parameters:
- `funcs`: A callable or an iterable of callables representing the integrands, which operate elementwise on the inputs.
- `H`: An integer specifying the number of Gauss-Hermite quadrature points.
- `Fmu`: A tensor or a tuple/list of tensors representing the means of the Gaussian distributions.
- `Fvar`: A tensor or a tuple/list of tensors representing the variances of the Gaussian distributions.
- `logspace`: A boolean indicating whether to compute the log-expectation of `exp(funcs)`.
- `Ys`: Additional named arguments passed as tensors, which represent deterministic inputs to the integrands.

The function should reshape `Fmu` and `Fvar` to ensure they match the expected dimensionality and apply Gauss-Hermite quadrature using the `NDiagGHQuadrature` class. If `logspace` is `True`, it computes the log-expectation of the functions; otherwise, it computes the standard expectation. The result should be returned with the same shape as the input `Fmu`.

### Error Handling:
- A deprecation warning should be issued advising the use of `gpflow.quadrature.NDiagGHQuadrature` instead.

This function is particularly useful in Gaussian process models or other machine learning contexts where expectations with respect to Gaussian distributions are required, and it leverages Gauss-Hermite quadrature for efficient computation.
@check_shapes(
    "Fmu: [broadcast Din, N...]",
    "Fvar: [broadcast Din, N...]",
    "Ys.values(): [N...]",
    "return: [broadcast Dout, N...]",
)
def ndiagquad(
    funcs: Union[Callable[..., tf.Tensor], Iterable[Callable[..., tf.Tensor]]],
    H: int,
    Fmu: Union[TensorType, Tuple[TensorType, ...], List[TensorType]],
    Fvar: Union[TensorType, Tuple[TensorType, ...], List[TensorType]],
    logspace: bool = False,
    **Ys: TensorType,
) -> tf.Tensor:"""
    warnings.warn(
        "ndiagquad is deprecated; use gpflow.quadrature.NDiagGHQuadrature instead.",
        DeprecationWarning,
    )

    if isinstance(Fmu, (tuple, list)):
        Din = len(Fmu)
        shape = tf.shape(Fmu[0])
        Fmu_stacked = tf.stack([tf.reshape(f, (-1,)) for f in Fmu], axis=-1)  # [N, Din]
        Fvar_stacked = tf.stack([tf.reshape(f, (-1,)) for f in Fvar], axis=-1)  # [N, Din]
    else:
        Din = 1
        shape = tf.shape(Fmu)
        Fmu_stacked = tf.reshape(Fmu, (-1, 1))  # [N, 1]
        Fvar_stacked = tf.reshape(Fvar, (-1, 1))  # [N, 1]

    quadrature = NDiagGHQuadrature(Din, H)

    def make_wrapper(f: Callable[..., tf.Tensor]) -> Callable[..., tf.Tensor]:
        def wrapper(X: tf.Tensor, **kwargs: TensorType) -> tf.Tensor:
            # X: [H**Din, N, Din]
            Xs = [X[..., i] for i in range(Din)]  # each [H**Din, N]
            n_quad = tf.shape(X)[0]
            tiled_kwargs = {}
            for name, Y in kwargs.items():
                Y_expanded = Y[tf.newaxis, ...]  # [1, N...]
                multiples = tf.concat([[n_quad], tf.ones_like(tf.shape(Y))], axis=0)
                tiled_kwargs[name] = tf.tile(Y_expanded, multiples)  # [H**Din, N...]
            feval = f(*Xs, **tiled_kwargs)  # [H**Din, N]
            return feval[..., tf.newaxis]  # [H**Din, N, 1]
        return wrapper

    if isinstance(funcs, Iterable):
        wrapped_funcs = [make_wrapper(f) for f in funcs]
        if logspace:
            results = quadrature.logspace(wrapped_funcs, Fmu_stacked, Fvar_stacked, **Ys)
        else:
            results = quadrature(wrapped_funcs, Fmu_stacked, Fvar_stacked, **Ys)
        return [tf.reshape(r, shape) for r in results]
    else:
        wrapped_func = make_wrapper(funcs)
        if logspace:
            result = quadrature.logspace(wrapped_func, Fmu_stacked, Fvar_stacked, **Ys)
        else:
            result = quadrature(wrapped_func, Fmu_stacked, Fvar_stacked, **Ys)
        return tf.reshape(result, shape)

@check_shapes('Fmu: [N, Din]', 'Fvar: [N, Din]', 'Ys.values(): [broadcast N, .]', 'return: [broadcast n_funs, N, P]')
def ndiag_mc(funcs: Union[Callable[..., tf.Tensor], Iterable[Callable[..., tf.Tensor]]], S: int, Fmu: TensorType, Fvar: TensorType, logspace: bool=False, epsilon: Optional[TensorType]=None, **Ys: TensorType) -> tf.Tensor:
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
    (N, D) = (tf.shape(Fmu)[0], tf.shape(Fvar)[1])
    if epsilon is None:
        epsilon = tf.random.normal(shape=[S, N, D], dtype=default_float())
    mc_x = Fmu[None, :, :] + tf.sqrt(Fvar[None, :, :]) * epsilon
    mc_Xr = tf.reshape(mc_x, (S * N, D))
    for (name, Y) in Ys.items():
        D_out = Y.shape[1]
        mc_Yr = tf.tile(Y[None, ...], [S, 1, 1])
        Ys[name] = tf.reshape(mc_Yr, (S * N, D_out))

    def eval_func(func: Callable[..., tf.Tensor]) -> tf.Tensor:
        feval = func(mc_Xr, **Ys)
        feval = tf.reshape(feval, (S, N, -1))
        if logspace:
            log_S = tf.math.log(to_default_float(S))
            return tf.reduce_logsumexp(feval, axis=0) - log_S
        else:
            return tf.reduce_mean(feval, axis=0)
    if isinstance(funcs, Iterable):
        return [eval_func(f) for f in funcs]
    else:
        return eval_func(funcs)
