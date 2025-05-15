import tensorflow as tf
from tensorflow.keras.losses import Loss

class SigmoidCrossEntropy(Loss):
    def _compute_tf_loss(self, output, labels):
        """
        Computes the sigmoid cross-entropy loss between logits and labels.

        Args:
            output (tf.Tensor): The logits tensor with shape (batch_size,) or (batch_size, tasks).
            labels (tf.Tensor): The labels tensor with the same shape as output.

        Returns:
            tf.Tensor: A tensor of loss values with shape (batch_size,) or (batch_size, tasks).
        """
        # Compute the sigmoid cross-entropy loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=output)
        
        # Return the loss
        return loss