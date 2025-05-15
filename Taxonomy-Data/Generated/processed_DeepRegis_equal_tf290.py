import tensorflow as tf

def is_equal_tf(x, y, atol=1e-5):
    """
    Check if two TensorFlow tensors are nearly equal within a specified tolerance.

    Parameters:
    x (tf.Tensor): First tensor to compare.
    y (tf.Tensor): Second tensor to compare.
    atol (float): Absolute tolerance. Default is 1e-5.

    Returns:
    bool: True if tensors are nearly equal, False otherwise.
    """
    # Ensure x and y are tensors
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    
    # Calculate the absolute difference
    difference = tf.math.abs(x - y)
    
    # Check if all elements are within the specified tolerance
    is_nearly_equal = tf.reduce_all(difference <= atol)
    
    # Return the result as a boolean
    return is_nearly_equal.numpy()

