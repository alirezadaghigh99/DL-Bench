    def _J(self, theta: TensorType) -> TensorType:
        """
        Implements the order dependent family of functions defined in equations
        4 to 7 in the reference paper.
        """
        if self.order == 0:
            return np.pi - theta
        elif self.order == 1:
            return tf.sin(theta) + (np.pi - theta) * tf.cos(theta)
        else:
            assert self.order == 2, f"Don't know how to handle order {self.order}."
            return 3.0 * tf.sin(theta) * tf.cos(theta) + (np.pi - theta) * (
                1.0 + 2.0 * tf.cos(theta) ** 2
            )