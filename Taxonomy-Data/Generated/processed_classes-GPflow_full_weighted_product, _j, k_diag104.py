import tensorflow as tf
from typing import Optional

# Assuming TensorType is an alias for tf.Tensor
TensorType = tf.Tensor

class Kernel:
    # Placeholder for the Kernel base class
    pass

class ArcCosine(Kernel):
    def __init__(self, weight_variances: tf.Tensor, bias_variance: float):
        self.weight_variances = weight_variances
        self.bias_variance = bias_variance

    def _full_weighted_product(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        if X2 is None:
            X2 = X

        # Compute the weighted product
        weighted_X = X * self.weight_variances
        weighted_X2 = X2 * self.weight_variances

        # Compute the dot product
        dot_product = tf.matmul(weighted_X, weighted_X2, transpose_b=True)

        # Add the bias variance
        weighted_product = dot_product + self.bias_variance

        return weighted_product

