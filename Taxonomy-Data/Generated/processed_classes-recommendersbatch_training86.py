class RBM:
    def __init__(self, possible_ratings, visible_units, hidden_units=500, keep_prob=0.7, init_stdv=0.1, learning_rate=0.004, minibatch_size=100, training_epoch=20, display_epoch=10, sampling_protocol=[50, 70, 80, 90, 100], debug=False, with_metrics=False, seed=42):
        self.sess = None  # TensorFlow session
        self.opt = None  # optimizer operation for training
        self.rmse = None  # root mean square error operation for evaluation
        self.with_metrics = with_metrics  # flag to determine if metrics should be evaluated
        pass

    def batch_training(self, num_minibatches):
        if self.sess is None or self.opt is None:
            raise ValueError("TensorFlow session or optimizer operation is not initialized.")
        
        total_error = 0.0

        for _ in range(num_minibatches):
            # Run the training operation
            self.sess.run(self.opt)

            if self.with_metrics:
                # Compute the RMSE for the current minibatch
                batch_error = self.sess.run(self.rmse)
                total_error += batch_error

        if self.with_metrics:
            # Calculate the average error over all minibatches
            average_error = total_error / num_minibatches
            return average_error
        else:
            return 0