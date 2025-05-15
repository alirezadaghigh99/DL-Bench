    def load(self, file_path="./rbm_model.ckpt"):
        """Load model parameters for further use.

        This function loads a saved tensorflow session.

        Args:
            file_path (str): file path for RBM model checkpoint
        """

        f_path = Path(file_path)
        dir_name, file_name = f_path.parent, f_path.name

        # load pre-trained model
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, os.path.join(dir_name, file_name))