    def free_energy(self, x):
        """Free energy of the visible units given the hidden units. Since the sum is over the hidden units'
        states, the functional form of the visible units Free energy is the same as the one for the binary model.

        Args:
            x (tf.Tensor): This can be either the sampled value of the visible units (v_k) or the input data

        Returns:
            tf.Tensor: Free energy of the model.
        """

        bias = -tf.reduce_sum(input_tensor=tf.matmul(x, tf.transpose(a=self.bv)))

        phi_x = tf.matmul(x, self.w) + self.bh
        f = -tf.reduce_sum(input_tensor=tf.nn.softplus(phi_x))

        F = bias + f  # free energy density per training example

        return F