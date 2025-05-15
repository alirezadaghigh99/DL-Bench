import tensorflow as tf

class Loss:
    # Assuming there are other methods and properties in the base Loss class
    pass

class L2Loss(Loss):
    def _compute_tf_loss(self, output, labels):
        # Ensure the output and labels are of float type
        output = tf.convert_to_tensor(output, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        
        # Check that the shapes of output and labels are consistent
        tf.debugging.assert_shapes([(output, labels.shape)], 
                                   message="Output and labels must have the same shape.")
        
        # Compute the squared differences
        squared_difference = tf.square(output - labels)
        
        # Compute the mean of the squared differences
        loss = tf.reduce_mean(squared_difference)
        
        return loss