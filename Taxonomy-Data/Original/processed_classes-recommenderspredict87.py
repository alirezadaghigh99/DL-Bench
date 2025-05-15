    def predict(self, x):
        """Returns the inferred ratings. This method is similar to recommend_k_items() with the
        exceptions that it returns all the inferred ratings

        Basic mechanics:

        The method samples new ratings from the learned joint distribution, together with
        their probabilities. The input x must have the same number of columns as the one used
        for training the model, i.e. the same number of items, but it can have an arbitrary number
        of rows (users).

        Args:
            x (numpy.ndarray, int32): Input user/affinity matrix. Note that this can be a single vector, i.e.
            the ratings of a single user.

        Returns:
            numpy.ndarray, float:
            - A matrix with the inferred ratings.
            - The elapsed time for predediction.
        """

        v_, _ = self.eval_out()  # evaluate the ratings and the associated probabilities
        vp = self.sess.run(v_, feed_dict={self.vu: x})

        return vp