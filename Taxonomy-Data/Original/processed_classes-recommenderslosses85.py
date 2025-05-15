    def losses(self, vv):
        """Calculate contrastive divergence, which is the difference between
        the free energy clamped on the data (v) and the model Free energy (v_k).

        Args:
            vv (tf.Tensor, float32): empirical input

        Returns:
            obj: contrastive divergence
        """

        with tf.compat.v1.variable_scope("losses"):
            obj = self.free_energy(vv) - self.free_energy(self.v_k)

        return obj