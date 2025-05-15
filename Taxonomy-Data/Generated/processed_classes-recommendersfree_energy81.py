import tensorflow as tf

class RBM:
    def __init__(self, possible_ratings, visible_units, hidden_units=500, keep_prob=0.7, init_stdv=0.1, learning_rate=0.004, minibatch_size=100, training_epoch=20, display_epoch=10, sampling_protocol=[50, 70, 80, 90, 100], debug=False, with_metrics=False, seed=42):
        self.bv = tf.Variable(tf.zeros([visible_units]), dtype=tf.float32)  # biases of the visible units
        self.bh = tf.Variable(tf.zeros([hidden_units]), dtype=tf.float32)  # biases of the hidden units
        self.w = tf.Variable(tf.random.normal([visible_units, hidden_units], stddev=init_stdv), dtype=tf.float32)  # weights between visible and hidden units

    def free_energy(self, x):
        # Compute the bias term for visible units
        vbias_term = tf.reduce_sum(tf.multiply(x, self.bv), axis=1)
        
        # Compute the hidden units contribution
        wx_b = tf.matmul(x, self.w) + self.bh
        hidden_term = tf.reduce_sum(tf.math.log(1 + tf.exp(wx_b)), axis=1)
        
        # Free energy is the negative sum of these terms
        free_energy = -vbias_term - hidden_term
        
        return free_energy

