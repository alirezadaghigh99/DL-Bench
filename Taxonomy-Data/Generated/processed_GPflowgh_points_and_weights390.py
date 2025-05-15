import tensorflow as tf
import numpy as np
from typing import Tuple

def gh_points_and_weights(n_gh: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Given the number of Gauss-Hermite points n_gh,
    returns the points z and the weights dz to perform the following
    uni-dimensional gaussian quadrature:

    X ~ N(mean, stddev²)
    E[f(X)] = ∫ f(x) p(x) dx = \sum_{i=1}^{n_gh} f(mean + stddev*z_i) dz_i

    :param n_gh: Number of Gauss-Hermite points
    :returns: Points z and weights dz to compute uni-dimensional gaussian expectation
    """
    # Use numpy to get the Gauss-Hermite points and weights
    z, dz = np.polynomial.hermite.hermgauss(n_gh)
    
    # Convert the numpy arrays to TensorFlow tensors
    z_tensor = tf.convert_to_tensor(z, dtype=tf.float32)
    dz_tensor = tf.convert_to_tensor(dz, dtype=tf.float32)
    
    return z_tensor, dz_tensor

