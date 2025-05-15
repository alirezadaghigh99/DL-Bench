    def sample_hidden_units(self, vv):
        """Sampling: In RBM we use Contrastive divergence to sample the parameter space. In order to do that we need
        to initialize the two conditional probabilities:

        P(h|phi_v) --> returns the probability that the i-th hidden unit is active

        P(v|phi_h) --> returns the probability that the  i-th visible unit is active

        Sample hidden units given the visibles. This can be thought of as a Forward pass step in a FFN

        Args:
            vv (tf.Tensor, float32): visible units

        Returns:
            tf.Tensor, tf.Tensor:
            - `phv`: The activation probability of the hidden unit.
            - `h_`: The sampled value of the hidden unit from a Bernoulli distributions having success probability `phv`.
        """

        with tf.compat.v1.name_scope("sample_hidden_units"):

            phi_v = tf.matmul(vv, self.w) + self.bh  # create a linear combination
            phv = tf.nn.sigmoid(phi_v)  # conditional probability of h given v
            phv_reg = tf.nn.dropout(phv, 1 - (self.keep))

            # Sampling
            h_ = self.binomial_sampling(
                phv_reg
            )  # obtain the value of the hidden units via Bernoulli sampling

        return phv, h_