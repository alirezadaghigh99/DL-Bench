    def _compute_tf_loss(self, output, labels):
        """Compute the loss function for TensorFlow tensors.

        The inputs are tensors containing the model's outputs and the labels for a
        batch.  The return value should be a tensor of shape (batch_size) or
        (batch_size, tasks) containing the value of the loss function on each
        sample or sample/task.

        Parameters
        ----------
        output: tensor
            the output of the model
        labels: tensor
            the expected output

        Returns
        -------
        The value of the loss function on each sample or sample/task pair
        """
        raise NotImplementedError("Subclasses must implement this")