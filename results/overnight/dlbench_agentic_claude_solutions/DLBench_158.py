import numpy as np
import tensorflow as tf
import logging
import os
from pathlib import Path
tf.compat.v1.disable_eager_execution()
log = logging.getLogger(__name__)

class RBM:
    """Restricted  Boltzmann Machine"""

    def __init__(self, possible_ratings, visible_units, hidden_units=500, keep_prob=0.7, init_stdv=0.1, learning_rate=0.004, minibatch_size=100, training_epoch=20, display_epoch=10, sampling_protocol=[50, 70, 80, 90, 100], debug=False, with_metrics=False, seed=42):
        """Implementation of a multinomial Restricted Boltzmann Machine for collaborative filtering
        in numpy/pandas/tensorflow

        Based on the article by Ruslan Salakhutdinov, Andriy Mnih and Geoffrey Hinton
        https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf

        In this implementation we use multinomial units instead of the one-hot-encoded used in
        the paper. This means that the weights are rank 2 (matrices) instead of rank 3 tensors.

        Basic mechanics:

        1) A computational graph is created when the RBM class is instantiated.
        For an item based recommender this consists of:
        visible units: The number n_visible of visible units equals the number of items
        hidden units : hyperparameter to fix during training

        2) Gibbs Sampling:

        2.1) for each training epoch, the visible units are first clamped on the data

        2.2) The activation probability of the hidden units, given a linear combination of
        the visibles, is evaluated P(h=1|phi_v). The latter is then used to sample the
        value of the hidden units.

        2.3) The probability P(v=l|phi_h) is evaluated, where l=1,..,r are the ratings (e.g.
        r=5 for the movielens dataset). In general, this is a multinomial distribution,
        from which we sample the value of v.

        2.4) This step is repeated k times, where k increases as optimization converges. It is
        essential to fix to zero the original unrated items during the all learning process.

        3) Optimization:
        The free energy of the visible units given the hidden is evaluated at the beginning (F_0)
        and after k steps of Bernoulli sampling (F_k). The weights and biases are updated by
        minimizing the differene F_0 - F_k.

        4) Inference:
        Once the joint probability distribution P(v,h) is learned, this is used to generate ratings
        for unrated items for all users
        """
        self.n_hidden = hidden_units
        self.keep = keep_prob
        self.stdv = init_stdv
        self.learning_rate = learning_rate
        self.minibatch = minibatch_size
        self.epochs = training_epoch + 1
        self.display_epoch = display_epoch
        self.sampling_protocol = sampling_protocol
        self.debug = debug
        self.with_metrics = with_metrics
        self.seed = seed
        np.random.seed(self.seed)
        tf.compat.v1.set_random_seed(self.seed)
        self.n_visible = visible_units
        tf.compat.v1.reset_default_graph()
        self.possible_ratings = possible_ratings
        self.ratings_lookup_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(tf.constant(list(range(len(self.possible_ratings))), dtype=tf.int32), tf.constant(list(self.possible_ratings), dtype=tf.float32)), default_value=0)
        self.generate_graph()
        self.init_metrics()
        self.init_gpu()
        init_graph = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session(config=self.config_gpu)
        self.sess.run(init_graph)

    def binomial_sampling(self, pr):
        """Binomial sampling of hidden units activations using a rejection method.

        Basic mechanics:

        1) Extract a random number from a uniform distribution (g) and compare it with
        the unit's probability (pr)

        2) Choose 0 if pr<g, 1 otherwise. It is convenient to implement this condtion using
        the relu function.

        Args:
            pr (tf.Tensor, float32): Input conditional probability.
            g  (numpy.ndarray, float32):  Uniform probability used for comparison.

        Returns:
            tf.Tensor: Float32 tensor of sampled units. The value is 1 if pr>g and 0 otherwise.
        """
        g = tf.convert_to_tensor(value=np.random.uniform(size=pr.shape[1]), dtype=tf.float32)
        h_sampled = tf.nn.relu(tf.sign(pr - g))
        return h_sampled

    def multinomial_sampling(self, pr):
        """Multinomial Sampling of ratings

        Basic mechanics:
        For r classes, we sample r binomial distributions using the rejection method. This is possible
        since each class is statistically independent from the other. Note that this is the same method
        used in numpy's random.multinomial() function.

        1) extract a size r array of random numbers from a uniform distribution (g). As pr is normalized,
        we need to normalize g as well.

        2) For each user and item, compare pr with the reference distribution. Note that the latter needs
        to be the same for ALL the user/item pairs in the dataset, as by assumptions they are sampled
        from a common distribution.

        Args:
            pr (tf.Tensor, float32): A distributions of shape (m, n, r), where m is the number of examples, n the number
                 of features and r the number of classes. pr needs to be normalized, i.e. sum_k p(k) = 1 for all m, at fixed n.
            f (tf.Tensor, float32): Normalized, uniform probability used for comparison.

        Returns:
            tf.Tensor: An (m,n) float32 tensor of sampled rankings from 1 to r.
        """
        g = np.random.uniform(size=pr.shape[2])
        f = tf.convert_to_tensor(value=g / g.sum(), dtype=tf.float32)
        samp = tf.nn.relu(tf.sign(pr - f))
        v_argmax = tf.cast(tf.argmax(input=samp, axis=2), 'int32')
        v_samp = tf.cast(self.ratings_lookup_table.lookup(v_argmax), 'float32')
        return v_samp

    def multinomial_distribution(self, phi):
        """Probability that unit v has value l given phi: P(v=l|phi)

        Args:
            phi (tf.Tensor): linear combination of values of the previous layer
            r (float): rating scale, corresponding to the number of classes

        Returns:
            tf.Tensor:
            - A tensor of shape (r, m, Nv): This needs to be reshaped as (m, Nv, r) in the last step to allow for faster sampling when used in the multinomial function.

        """
        numerator = [tf.exp(tf.multiply(tf.constant(k, dtype='float32'), phi)) for k in self.possible_ratings]
        denominator = tf.reduce_sum(input_tensor=numerator, axis=0)
        prob = tf.compat.v1.div(numerator, denominator)
        return tf.transpose(a=prob, perm=[1, 2, 0])

    def free_energy(self, x):
        """Free energy of the visible units given the hidden units. Since the sum is over the hidden units'
        states, the functional form of the visible units Free energy is the same as the one for the binary model.

        Args:
            x (tf.Tensor): This can be either the sampled value of the visible units (v_k) or the input data

        Returns:
            tf.Tensor: Free energy of the model.
        """
        bias = -tf.reduce_sum(input_tensor=tf.matmul(x, tf.transpose(a=self.bv)))
        phi_x = tf.matmul(x, self.w) + self.bh
        f = -tf.reduce_sum(input_tensor=tf.nn.softplus(phi_x))
        F = bias + f
        return F

    def placeholder(self):
        """Initialize the placeholders for the visible units"""
        self.vu = tf.compat.v1.placeholder(shape=[None, self.n_visible], dtype='float32')

    def init_parameters(self):
        """Initialize the parameters of the model.

        This is a single layer model with two biases. So we have a rectangular matrix w_{ij} and
        two bias vectors to initialize.

        Args:
            n_visible (int): number of visible units (input layer)
            n_hidden (int): number of hidden units (latent variables of the model)

        Returns:
            tf.Tensor, tf.Tensor, tf.Tensor:
            - `w` of size (n_visible, n_hidden): correlation matrix initialized by sampling from a normal distribution with zero mean and given variance init_stdv.
            - `bv` of size (1, n_visible): visible units' bias, initialized to zero.
            - `bh` of size (1, n_hidden): hidden units' bias, initiliazed to zero.
        """
        with tf.compat.v1.variable_scope('Network_parameters'):
            self.w = tf.compat.v1.get_variable('weight', [self.n_visible, self.n_hidden], initializer=tf.compat.v1.random_normal_initializer(stddev=self.stdv, seed=self.seed), dtype='float32')
            self.bv = tf.compat.v1.get_variable('v_bias', [1, self.n_visible], initializer=tf.compat.v1.zeros_initializer(), dtype='float32')
            self.bh = tf.compat.v1.get_variable('h_bias', [1, self.n_hidden], initializer=tf.compat.v1.zeros_initializer(), dtype='float32')

    def sample_hidden_units(self, vv):
        """Sampling: In RBM we use Contrastive divergence to sample the parameter space. In order to do that we need
        to initialize the two conditional probabilities:

        P(h|phi_v) --> returns the probability that the i-th hidden unit is active

        P(v|phi_h) --> returns the probability that the  i-th visible unit is active

        Sample hidden units given the visibles. This can be thought of as a Forward pass step in a FFN

        Args:
            vv (tf.Tensor, float32): visible units

        Returns:
            tf.Tensor, tf.Tensor:
            - `phv`: The activation probability of the hidden unit.
            - `h_`: The sampled value of the hidden unit from a Bernoulli distributions having success probability `phv`.
        """
        with tf.compat.v1.name_scope('sample_hidden_units'):
            phi_v = tf.matmul(vv, self.w) + self.bh
            phv = tf.nn.sigmoid(phi_v)
            phv_reg = tf.nn.dropout(phv, 1 - self.keep)
            h_ = self.binomial_sampling(phv_reg)
        return (phv, h_)

    def sample_visible_units(self, h):
        """Sample the visible units given the hiddens. This can be thought of as a Backward pass in a FFN
        (negative phase). Each visible unit can take values in [1,rating], while the zero is reserved
        for missing data; as such the value of the hidden unit is sampled from a multinomial distribution.

        Basic mechanics:

        1) For every training example we first sample Nv Multinomial distributions. The result is of the
        form [0,1,0,0,0,...,0] where the index of the 1 element corresponds to the rth rating. The index
        is extracted using the argmax function and we need to add 1 at the end since array indeces starts
        from 0.

        2) Selects only those units that have been sampled. During the training phase it is important to not
        use the reconstructed inputs, so we beed to enforce a zero value in the reconstructed ratings in
        the same position as the original input.

        Args:
            h (tf.Tensor, float32): visible units.

        Returns:
            tf.Tensor, tf.Tensor:
            - `pvh`: The activation probability of the visible unit given the hidden.
            - `v_`: The sampled value of the visible unit from a Multinomial distributions having success probability `pvh`.
        """
        with tf.compat.v1.name_scope('sample_visible_units'):
            phi_h = tf.matmul(h, tf.transpose(a=self.w)) + self.bv
            pvh = self.multinomial_distribution(phi_h)
            v_tmp = self.multinomial_sampling(pvh)
            mask = tf.equal(self.v, 0)
            v_ = tf.compat.v1.where(mask, x=self.v, y=v_tmp)
        return (pvh, v_)

    def gibbs_sampling(self):
        """Gibbs sampling: Determines an estimate of the model configuration via sampling. In the binary
        RBM we need to impose that unseen movies stay as such, i.e. the sampling phase should not modify
        the elements where v=0.

        Args:
            k (scalar, integer): iterator. Number of sampling steps.
            v (tf.Tensor, float32): visible units.

        Returns:
            tf.Tensor, tf.Tensor:
            - `h_k`: The sampled value of the hidden unit at step k, float32.
            - `v_k`: The sampled value of the visible unit at step k, float32.
        """
        with tf.compat.v1.name_scope('gibbs_sampling'):
            self.v_k = self.v
            if self.debug:
                print('CD step', self.k)
            for i in range(self.k):
                (_, h_k) = self.sample_hidden_units(self.v_k)
                (_, self.v_k) = self.sample_visible_units(h_k)

    def losses(self, vv):
        """Calculate contrastive divergence, which is the difference between
        the free energy clamped on the data (v) and the model Free energy (v_k).

        Args:
            vv (tf.Tensor, float32): empirical input

        Returns:
            obj: contrastive divergence
        """
        with tf.compat.v1.variable_scope('losses'):
            obj = self.free_energy(vv) - self.free_energy(self.v_k)
        return obj

    def gibbs_protocol(self, i):
        """Gibbs protocol.

        Basic mechanics:

        If the current epoch i is in the interval specified in the training protocol,
        the number of steps in Gibbs sampling (k) is incremented by one and gibbs_sampling is updated
        accordingly.

        Args:
            i (int): Current epoch in the loop
        """
        with tf.compat.v1.name_scope('gibbs_protocol'):
            epoch_percentage = i / self.epochs * 100
            if epoch_percentage != 0:
                if epoch_percentage >= self.sampling_protocol[self.l] and epoch_percentage <= self.sampling_protocol[self.l + 1]:
                    self.k += 1
                    self.l += 1
                    self.gibbs_sampling()
            if self.debug:
                log.info('percentage of epochs covered so far %f2' % epoch_percentage)

    def data_pipeline(self):
        """Define the data pipeline"""
        self.batch_size = tf.compat.v1.placeholder(tf.int64)
        self.dataset = tf.data.Dataset.from_tensor_slices(self.vu)
        self.dataset = self.dataset.shuffle(buffer_size=50, reshuffle_each_iteration=True, seed=self.seed)
        self.dataset = self.dataset.batch(batch_size=self.batch_size).repeat()
        self.iter = tf.compat.v1.data.make_initializable_iterator(self.dataset)
        self.v = self.iter.get_next()

    def init_metrics(self):
        """Initialize metrics"""
        if self.with_metrics:
            self.rmse = tf.sqrt(tf.compat.v1.losses.mean_squared_error(self.v, self.v_k, weights=tf.where(self.v > 0, 1, 0)))

    def generate_graph(self):
        """Call the different RBM modules to generate the computational graph"""
        log.info('Creating the computational graph')
        self.placeholder()
        self.data_pipeline()
        self.init_parameters()
        log.info('Initialize Gibbs protocol')
        self.k = 1
        self.l = 0
        self.gibbs_sampling()
        obj = self.losses(self.v)
        rate = self.learning_rate / self.minibatch
        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=rate).minimize(loss=obj)

    def init_gpu(self):
        """Config GPU memory"""
        self.config_gpu = tf.compat.v1.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        self.config_gpu.gpu_options.allow_growth = True

    def init_training_session(self, xtr):
        """Initialize the TF session on training data

        Args:
            xtr (numpy.ndarray, int32): The user/affinity matrix for the train set.
        """
        self.sess.run(self.iter.initializer, feed_dict={self.vu: xtr, self.batch_size: self.minibatch})
        self.sess.run(tf.compat.v1.tables_initializer())

    def batch_training(self, num_minibatches):
        """Perform training over input minibatches. If `self.with_metrics` is False,
        no online metrics are evaluated.

        Args:
            num_minibatches (scalar, int32): Number of training minibatches.

        Returns:
            float: Training error per epoch, averaged over the minibatches. Returns 0 if
            `self.with_metrics` is False.
        """
        epoch_tr_err = 0
        if self.with_metrics:
            for _ in range(num_minibatches):
                (_, batch_err) = self.sess.run([self.opt, self.rmse])
                epoch_tr_err += batch_err
            epoch_tr_err /= num_minibatches
        else:
            for _ in range(num_minibatches):
                self.sess.run(self.opt)
        return epoch_tr_err

    def fit(self, xtr):
        """Fit method

        Training in generative models takes place in two steps:

        1) Gibbs sampling
        2) Gradient evaluation and parameters update

        This estimate is later used in the weight update step by minimizing the distance between the
        model and the empirical free energy. Note that while the unit's configuration space is sampled,
        the weights are determined via maximum likelihood (saddle point).

        Main component of the algo; once instantiated, it generates the computational graph and performs
        model training

        Args:
            xtr (numpy.ndarray, integers): the user/affinity matrix for the train set
            xtst (numpy.ndarray, integers): the user/affinity matrix for the test set
        """
        self.seen_mask = np.not_equal(xtr, 0)
        n_users = xtr.shape[0]
        num_minibatches = int(n_users / self.minibatch)
        self.init_training_session(xtr)
        rmse_train = []
        for i in range(self.epochs):
            self.gibbs_protocol(i)
            epoch_tr_err = self.batch_training(num_minibatches)
            if self.with_metrics and i % self.display_epoch == 0:
                log.info('training epoch %i rmse %f' % (i, epoch_tr_err))
            rmse_train.append(epoch_tr_err)
        self.rmse_train = rmse_train

    def eval_out(self):
        """Implement multinomial sampling from a trained model"""
        (_, h) = self.sample_hidden_units(self.vu)
        phi_h = tf.transpose(a=tf.matmul(self.w, tf.transpose(a=h))) + self.bv
        pvh = self.multinomial_distribution(phi_h)
        v = self.multinomial_sampling(pvh)
        return (v, pvh)

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
        (v_, pvh_) = self.eval_out()
        (vp, pvh) = self.sess.run([v_, pvh_], feed_dict={self.vu: x})
        pv = np.max(pvh, axis=2)
        score = np.multiply(vp, pv)
        log.info('Extracting top %i elements' % top_k)
        if remove_seen:
            vp[self.seen_mask] = 0
            pv[self.seen_mask] = 0
            score[self.seen_mask] = 0
        top_items = np.argpartition(-score, range(top_k), axis=1)[:, :top_k]
        score_c = score.copy()
        score_c[np.arange(score_c.shape[0])[:, None], top_items] = 0
        top_scores = score - score_c
        return top_scores

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
        (v_, _) = self.eval_out()
        vp = self.sess.run(v_, feed_dict={self.vu: x})
        return vp

    def save(self, file_path='./rbm_model.ckpt'):
        """Save model parameters to `file_path`

        This function saves the current tensorflow session to a specified path.

        Args:
            file_path (str): output file path for the RBM model checkpoint
                we will create a new directory if not existing.
        """
        f_path = Path(file_path)
        (dir_name, file_name) = (f_path.parent, f_path.name)
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, os.path.join(dir_name, file_name))

    def load(self, file_path='./rbm_model.ckpt'):
        """Load model parameters for further use.

        This function loads a saved tensorflow session.

        Args:
            file_path (str): file path for RBM model checkpoint
        """
        f_path = Path(file_path)
        (dir_name, file_name) = (f_path.parent, f_path.name)
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, os.path.join(dir_name, file_name))
