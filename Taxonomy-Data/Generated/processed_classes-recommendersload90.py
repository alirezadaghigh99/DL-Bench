import tensorflow as tf

class RBM:
    def __init__(self, possible_ratings, visible_units, hidden_units=500, keep_prob=0.7, init_stdv=0.1, learning_rate=0.004, minibatch_size=100, training_epoch=20, display_epoch=10, sampling_protocol=[50, 70, 80, 90, 100], debug=False, with_metrics=False, seed=42):
        self.possible_ratings = possible_ratings
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.keep_prob = keep_prob
        self.init_stdv = init_stdv
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.training_epoch = training_epoch
        self.display_epoch = display_epoch
        self.sampling_protocol = sampling_protocol
        self.debug = debug
        self.with_metrics = with_metrics
        self.seed = seed

        # Initialize TensorFlow session
        self.sess = tf.Session()

        # Define model variables (weights and biases)
        self.weights = tf.Variable(tf.random_normal([self.visible_units, self.hidden_units], stddev=self.init_stdv), name="weights")
        self.hidden_bias = tf.Variable(tf.zeros([self.hidden_units]), name="hidden_bias")
        self.visible_bias = tf.Variable(tf.zeros([self.visible_units]), name="visible_bias")

        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())

    def load(self, file_path="./rbm_model.ckpt"):
        # Create a saver object
        saver = tf.train.Saver()

        # Restore the model from the checkpoint
        try:
            saver.restore(self.sess, file_path)
            print(f"Model parameters loaded from {file_path}")
        except Exception as e:
            print(f"Failed to load model parameters from {file_path}: {e}")

