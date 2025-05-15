    def binomial_sampling(self, pr):
        """Binomial sampling of hidden units activations using a rejection method.

        Basic mechanics:

        1) Extract a random number from a uniform distribution (g) and compare it with
        the unit's probability (pr)

        2) Choose 0 if pr<g, 1 otherwise. It is convenient to implement this condtion using
        the relu function.

        Args:
            pr (tf.Tensor, float32): Input conditional probability.
            g  (numpy.ndarray, float32):  Uniform probability used for comparison.

        Returns:
            tf.Tensor: Float32 tensor of sampled units. The value is 1 if pr>g and 0 otherwise.
        """

        # sample from a Bernoulli distribution with same dimensions as input distribution
        g = tf.convert_to_tensor(
            value=np.random.uniform(size=pr.shape[1]), dtype=tf.float32
        )

        # sample the value of the hidden units
        h_sampled = tf.nn.relu(tf.sign(pr - g))

        return h_sampled