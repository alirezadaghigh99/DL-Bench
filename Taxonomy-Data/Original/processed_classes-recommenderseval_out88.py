    def eval_out(self):
        """Implement multinomial sampling from a trained model"""

        # Sampling
        _, h = self.sample_hidden_units(self.vu)  # sample h

        # sample v
        phi_h = (
            tf.transpose(a=tf.matmul(self.w, tf.transpose(a=h))) + self.bv
        )  # linear combination
        pvh = self.multinomial_distribution(
            phi_h
        )  # conditional probability of v given h

        v = self.multinomial_sampling(pvh)  # sample the value of the visible units

        return v, pvh