    def _full_weighted_product(self, X: TensorType, X2: Optional[TensorType]) -> tf.Tensor:
        if X2 is None:
            return (
                tf.linalg.matmul((self.weight_variances * X), X, transpose_b=True)
                + self.bias_variance
            )

        else:
            D = tf.shape(X)[-1]
            N = tf.shape(X)[-2]
            N2 = tf.shape(X2)[-2]
            batch = tf.shape(X)[:-2]
            batch2 = tf.shape(X2)[:-2]
            rank = tf.rank(X) - 2
            rank2 = tf.rank(X2) - 2
            ones = tf.ones((rank,), tf.int32)
            ones2 = tf.ones((rank2,), tf.int32)

            X = cs(
                tf.reshape(X, tf.concat([batch, ones2, [N, D]], 0)),
                "[batch..., broadcast batch2..., N, D]",
            )
            X2 = cs(
                tf.reshape(X2, tf.concat([ones, batch2, [N2, D]], 0)),
                "[broadcast batch..., batch2..., N2, D]",
            )
            result = cs(
                tf.linalg.matmul((self.weight_variances * X), X2, transpose_b=True)
                + self.bias_variance,
                "[batch..., batch2..., N, N2]",
            )

            indices = tf.concat(
                [
                    tf.range(rank),
                    [rank + rank2],
                    tf.range(rank2) + rank,
                    [rank + rank2 + 1],
                ],
                axis=0,
            )
            return tf.transpose(result, indices)