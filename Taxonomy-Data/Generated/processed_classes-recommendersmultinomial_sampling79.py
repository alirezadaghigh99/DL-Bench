import tensorflow as tf

class RBM:
    def __init__(self, possible_ratings, visible_units, hidden_units=500, keep_prob=0.7, init_stdv=0.1, learning_rate=0.004, minibatch_size=100, training_epoch=20, display_epoch=10, sampling_protocol=[50, 70, 80, 90, 100], debug=False, with_metrics=False, seed=42):
        self.possible_ratings = possible_ratings
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.keep_prob = keep_prob
        self.init_stdv = init_stdv
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.training_epoch = training_epoch
        self.display_epoch = display_epoch
        self.sampling_protocol = sampling_protocol
        self.debug = debug
        self.with_metrics = with_metrics
        self.seed = seed
        tf.random.set_seed(seed)

    def multinomial_sampling(self, pr):
        """
        Perform multinomial sampling of ratings using a rejection method.

        Args:
            pr (tf.Tensor): A tensor of shape (m, n, r) representing the distribution of ratings.

        Returns:
            tf.Tensor: A tensor of shape (m, n) containing the sampled ratings.
        """
        # Get the shape of the input tensor
        m, n, r = pr.shape

        # Sample from a multinomial distribution
        # We use tf.random.categorical to sample indices based on the probabilities
        pr_reshaped = tf.reshape(pr, [-1, r])  # Reshape to (m*n, r)
        sampled_indices = tf.random.categorical(tf.math.log(pr_reshaped), num_samples=1)  # Sample indices
        sampled_indices = tf.reshape(sampled_indices, [m, n])  # Reshape back to (m, n)

        return sampled_indices

