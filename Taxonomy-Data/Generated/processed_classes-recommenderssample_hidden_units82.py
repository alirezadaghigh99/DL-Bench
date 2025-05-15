import tensorflow as tf

class RBM:
    def __init__(self, possible_ratings, visible_units, hidden_units=500, keep_prob=0.7, init_stdv=0.1, learning_rate=0.004, minibatch_size=100, training_epoch=20, display_epoch=10, sampling_protocol=[50, 70, 80, 90, 100], debug=False, with_metrics=False, seed=42):
        self.w = tf.Variable(tf.random.normal([visible_units, hidden_units], stddev=init_stdv), name='weights')
        self.bh = tf.Variable(tf.zeros([hidden_units]), name='hidden_biases')
        self.keep = keep_prob  # dropout keep probability
        self.seed = seed
        tf.random.set_seed(seed)

    def binomial_sampling(self, pr):
        # Outputs a tensor of the same shape as pr, where each element is 1 if the corresponding
        # probability is greater than a sampled uniform random value, and 0 otherwise.
        random_tensor = tf.random.uniform(tf.shape(pr), seed=self.seed)
        return tf.cast(pr > random_tensor, dtype=tf.float32)

    def sample_hidden_units(self, vv):
        # Compute the activation probabilities of the hidden units
        phv = tf.nn.sigmoid(tf.matmul(vv, self.w) + self.bh)
        
        # Apply dropout regularization
        phv_dropout = phv * self.keep
        
        # Sample the hidden units
        h_ = self.binomial_sampling(phv_dropout)
        
        return phv, h_

