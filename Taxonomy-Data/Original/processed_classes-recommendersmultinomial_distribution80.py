    def multinomial_distribution(self, phi):
        """Probability that unit v has value l given phi: P(v=l|phi)

        Args:
            phi (tf.Tensor): linear combination of values of the previous layer
            r (float): rating scale, corresponding to the number of classes

        Returns:
            tf.Tensor:
            - A tensor of shape (r, m, Nv): This needs to be reshaped as (m, Nv, r) in the last step to allow for faster sampling when used in the multinomial function.

        """

        numerator = [
            tf.exp(tf.multiply(tf.constant(k, dtype="float32"), phi))
            for k in self.possible_ratings
        ]

        denominator = tf.reduce_sum(input_tensor=numerator, axis=0)

        prob = tf.compat.v1.div(numerator, denominator)

        return tf.transpose(a=prob, perm=[1, 2, 0])