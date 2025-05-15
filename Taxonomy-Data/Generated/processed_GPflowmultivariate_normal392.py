import tensorflow as tf
from tensorflow_probability import distributions as tfd
from typing import Any

# Define a decorator for shape checking (this is a placeholder, as TensorFlow doesn't have built-in shape checking)
def check_shapes(*args: Any, **kwargs: Any):
    def decorator(func):
        def wrapper(*func_args, **func_kwargs):
            # Here you would implement shape checking logic if needed
            return func(*func_args, **func_kwargs)
        return wrapper
    return decorator

@check_shapes(
    "x: [D, broadcast N]",
    "mu: [D, broadcast N]",
    "L: [D, D]",
    "return: [N]",
)
def multivariate_normal(x: tf.Tensor, mu: tf.Tensor, L: tf.Tensor) -> tf.Tensor:
    """
    Computes the log-density of a multivariate normal.

    :param x: sample(s) for which we want the density
    :param mu: mean(s) of the normal distribution
    :param L: Cholesky decomposition of the covariance matrix
    :return: log densities
    """
    # Ensure x and mu are at least 2D
    x = tf.convert_to_tensor(x)
    mu = tf.convert_to_tensor(mu)
    L = tf.convert_to_tensor(L)

    # Compute the covariance matrix from the Cholesky factor
    cov = tf.matmul(L, L, transpose_b=True)

    # Create a multivariate normal distribution
    mvn = tfd.MultivariateNormalTriL(loc=mu, scale_tril=L)

    # Compute the log probability density
    log_prob = mvn.log_prob(x)

    return log_prob

