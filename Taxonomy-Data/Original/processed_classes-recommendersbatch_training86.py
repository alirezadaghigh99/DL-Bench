    def batch_training(self, num_minibatches):
        """Perform training over input minibatches. If `self.with_metrics` is False,
        no online metrics are evaluated.

        Args:
            num_minibatches (scalar, int32): Number of training minibatches.

        Returns:
            float: Training error per single epoch. If `self.with_metrics` is False, this is zero.
        """

        epoch_tr_err = 0  # initialize the training error for each epoch to zero

        # minibatch loop
        for _ in range(num_minibatches):

            if self.with_metrics:
                _, batch_err = self.sess.run([self.opt, self.rmse])

                # average msr error per minibatch
                epoch_tr_err += batch_err / num_minibatches

            else:
                _ = self.sess.run(self.opt)

        return epoch_tr_err