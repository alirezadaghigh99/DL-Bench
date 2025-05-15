    def save(self, file_path="./rbm_model.ckpt"):
        """Save model parameters to `file_path`

        This function saves the current tensorflow session to a specified path.

        Args:
            file_path (str): output file path for the RBM model checkpoint
                we will create a new directory if not existing.
        """

        f_path = Path(file_path)
        dir_name, file_name = f_path.parent, f_path.name

        # create the directory if it does not exist
        os.makedirs(dir_name, exist_ok=True)

        # save trained model
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, os.path.join(dir_name, file_name))