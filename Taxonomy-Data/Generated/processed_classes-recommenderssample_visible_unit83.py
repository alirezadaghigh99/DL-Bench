import tensorflow as tf

class RBM:
    def __init__(self, possible_ratings, visible_units, hidden_units=500, keep_prob=0.7, init_stdv=0.1, learning_rate=0.004, minibatch_size=100, training_epoch=20, display_epoch=10, sampling_protocol=[50, 70, 80, 90, 100], debug=False, with_metrics=False, seed=42):
        self.w = tf.Variable(tf.random.normal([visible_units, hidden_units], stddev=init_stdv), name='weights')
        self.bv = tf.Variable(tf.zeros([visible_units]), name='visible_bias')
        self.v = None  # This should be set to the original input vector during training or inference

    def multinomial_distribution(self, phi):
        # Outputs a tensor representing the probability distribution over the visible units
        return tf.nn.softmax(phi)

    def multinomial_sampling(self, pr):
        # Outputs a tensor representing sampled values of the visible units
        return tf.random.categorical(tf.math.log(pr), num_samples=1)

    def sample_visible_units(self, h):
        # Calculate the linear combination of hidden units and weights, add visible biases
        phi_v = tf.matmul(h, tf.transpose(self.w)) + self.bv
        
        # Calculate the activation probability of the visible units
        pvh = self.multinomial_distribution(phi_v)
        
        # Sample the visible units
        v_ = self.multinomial_sampling(pvh)
        
        # Apply mask to enforce zero values in the reconstructed vector for inactive units
        mask = tf.cast(tf.not_equal(self.v, 0), dtype=tf.float32)
        v_ = tf.multiply(tf.cast(v_, dtype=tf.float32), mask)
        
        return pvh, v_

