import tensorflow as tf
from tensorflow.keras.losses import Loss

class SoftmaxCrossEntropy(Loss):
    def _compute_tf_loss(self, output, labels):
        """
        Computes the softmax cross-entropy loss between logits and labels.

        Args:
            output (tf.Tensor): Logits tensor with shape (batch_size, classes) or (batch_size, tasks, classes).
            labels (tf.Tensor): Labels tensor with the same shape as output.

        Returns:
            tf.Tensor: Loss values tensor.
        """
        # Check if the input is 2D or 3D
        if len(output.shape) == 2:
            # 2D case: (batch_size, classes)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output)
        elif len(output.shape) == 3:
            # 3D case: (batch_size, tasks, classes)
            # Reshape to 2D for computation
            batch_size, tasks, classes = output.shape
            output_reshaped = tf.reshape(output, [-1, classes])
            labels_reshaped = tf.reshape(labels, [-1, classes])
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_reshaped, logits=output_reshaped)
            # Reshape back to 3D
            loss = tf.reshape(loss, [batch_size, tasks])
        else:
            raise ValueError("Output and labels must be 2D or 3D tensors.")

        return loss