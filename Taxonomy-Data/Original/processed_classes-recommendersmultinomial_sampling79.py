    def multinomial_sampling(self, pr):
        """Multinomial Sampling of ratings

        Basic mechanics:
        For r classes, we sample r binomial distributions using the rejection method. This is possible
        since each class is statistically independent from the other. Note that this is the same method
        used in numpy's random.multinomial() function.

        1) extract a size r array of random numbers from a uniform distribution (g). As pr is normalized,
        we need to normalize g as well.

        2) For each user and item, compare pr with the reference distribution. Note that the latter needs
        to be the same for ALL the user/item pairs in the dataset, as by assumptions they are sampled
        from a common distribution.

        Args:
            pr (tf.Tensor, float32): A distributions of shape (m, n, r), where m is the number of examples, n the number
                 of features and r the number of classes. pr needs to be normalized, i.e. sum_k p(k) = 1 for all m, at fixed n.
            f (tf.Tensor, float32): Normalized, uniform probability used for comparison.

        Returns:
            tf.Tensor: An (m,n) float32 tensor of sampled rankings from 1 to r.
        """
        g = np.random.uniform(size=pr.shape[2])  # sample from a uniform distribution
        f = tf.convert_to_tensor(
            value=g / g.sum(), dtype=tf.float32
        )  # normalize and convert to tensor

        samp = tf.nn.relu(tf.sign(pr - f))  # apply rejection method

        # get integer index of the rating to be sampled
        v_argmax = tf.cast(tf.argmax(input=samp, axis=2), "int32")

        # lookup the rating using integer index
        v_samp = tf.cast(self.ratings_lookup_table.lookup(v_argmax), "float32")

        return v_samp