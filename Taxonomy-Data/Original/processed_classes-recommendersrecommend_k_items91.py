    def recommend_k_items(self, x, top_k=10, remove_seen=True):
        """Returns the top-k items ordered by a relevancy score.

        Basic mechanics:

        The method samples new ratings from the learned joint distribution, together with their
        probabilities. The input x must have the same number of columns as the one used for training
        the model (i.e. the same number of items) but it can have an arbitrary number of rows (users).

        A recommendation score is evaluated by taking the element-wise product between the ratings and
        the associated probabilities. For example, we could have the following situation:

        .. code-block:: python

                    rating     probability     score
            item1     5           0.5          2.5
            item2     4           0.8          3.2

        then item2 will be recommended.

        Args:
            x (numpy.ndarray, int32): input user/affinity matrix. Note that this can be a single vector, i.e. the ratings
            of a single user.
            top_k (scalar, int32): the number of items to recommend.

        Returns:
            numpy.ndarray, float:
            - A sparse matrix containing the top_k elements ordered by their score.
            - The time taken to recommend k items.
        """

        # evaluate the ratings and the associated probabilities
        v_, pvh_ = self.eval_out()

        # evaluate v_ and pvh_ on the input data
        vp, pvh = self.sess.run([v_, pvh_], feed_dict={self.vu: x})
        # returns only the probabilities for the predicted ratings in vp
        pv = np.max(pvh, axis=2)

        # evaluate the score
        score = np.multiply(vp, pv)
        # ----------------------Return the results as a P dataframe------------------------------------

        log.info("Extracting top %i elements" % top_k)

        if remove_seen:
            # if true, it removes items from the train set by setting them to zero
            vp[self.seen_mask] = 0
            pv[self.seen_mask] = 0
            score[self.seen_mask] = 0

        top_items = np.argpartition(-score, range(top_k), axis=1)[
            :, :top_k
        ]  # get the top k items

        score_c = score.copy()  # get a copy of the score matrix

        score_c[
            np.arange(score_c.shape[0])[:, None], top_items
        ] = 0  # set to zero the top_k elements

        top_scores = score - score_c  # set to zeros all elements other then the top_k

        return top_scores