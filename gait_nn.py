import settings
import os

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

from abc import abstractmethod

slim = tf.contrib.slim

SUMMARY_PATH = settings.LOGDIR_GAIT_PATH
KEY_SUMMARIES = tf.GraphKeys.SUMMARIES

SEED = 0
np.random.seed(SEED)


class GaitNN(object):
    def __init__(self, name, input_tensor, features, num_of_persons, reuse = False, is_train = True,
                 count_of_training_examples = 1000):
        self.input_tensor = input_tensor
        self.is_train = is_train
        self.name = name

        self.FEATURES = features

        net = self.pre_process(input_tensor)
        net, gait_signature, state = self.get_network(net, is_train, reuse)

        self.network = net
        self.gait_signature = gait_signature
        self.state = state

        if is_train:
            # Initialize placeholders
            self.desired_person = tf.placeholder(
                dtype = tf.int32,
                shape = [],
                name = 'desired_person')

            self.desired_person_one_hot = tf.one_hot(self.desired_person, num_of_persons, dtype = tf.float32)
            self.loss = self._sigm_ce_loss()

            self.global_step = tf.Variable(0, name = 'global_step', trainable = False)

            self.learning_rate = tf.placeholder(
                dtype = tf.float32,
                shape = [],
                name = 'learning_rate')

            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps = count_of_training_examples * 2,
                    decay_rate = 0.96,
                    staircase = True)

            self.optimize = layers.optimize_loss(loss = self.loss,
                                                 global_step = self.global_step,
                                                 learning_rate = self.learning_rate,
                                                 summaries = layers.optimizers.OPTIMIZER_SUMMARIES,
                                                 optimizer = tf.train.RMSPropOptimizer,
                                                 learning_rate_decay_fn = _learning_rate_decay_fn,
                                                 clip_gradients = 0.1,
                                                 )

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Initialize summaries
        if name is not None:
            if is_train:
                logdir = os.path.join(SUMMARY_PATH, self.name, 'train')
                self.summary_writer = tf.train.SummaryWriter(logdir)

                self.ALL_SUMMARIES = tf.merge_all_summaries(KEY_SUMMARIES)
            else:
                self.summary_writer_d = {}

                for t in ['avg', 'n', 'b', 's']:
                    logdir = os.path.join(SUMMARY_PATH, self.name, 'val_%s' % t)
                    self.summary_writer_d[t] = tf.train.SummaryWriter(logdir)

        tf.set_random_seed(SEED)

    @staticmethod
    def pre_process(inp):
        return inp / 100.0

    @staticmethod
    def get_arg_scope(is_training):
        weight_decay_l2 = 0.1
        batch_norm_decay = 0.999
        batch_norm_epsilon = 0.0001

        with slim.arg_scope([slim.conv2d, slim.fully_connected, layers.separable_convolution2d],
                            weights_regularizer = slim.l2_regularizer(weight_decay_l2),
                            biases_regularizer = slim.l2_regularizer(weight_decay_l2),
                            weights_initializer = layers.variance_scaling_initializer(),
                            ):
            batch_norm_params = {
                'decay': batch_norm_decay,
                'epsilon': batch_norm_epsilon
            }
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training = is_training):
                with slim.arg_scope([slim.batch_norm],
                                    **batch_norm_params):
                    with slim.arg_scope([slim.conv2d, layers.separable_convolution2d, layers.fully_connected],
                                        activation_fn = tf.nn.elu,
                                        normalizer_fn = slim.batch_norm,
                                        normalizer_params = batch_norm_params) as scope:
                        return scope

    def _sigm_ce_loss(self):
        ce = tf.nn.softmax_cross_entropy_with_logits(logits = self.network, labels = self.desired_person_one_hot)
        loss = tf.reduce_mean(ce)

        return loss

    def train(self, input_tensor, desired_person, learning_rate):
        if not self.is_train:
            raise Exception('Network is not in training mode!')

        self.sess.run(self.optimize, feed_dict = {
            self.input_tensor: input_tensor,
            self.desired_person: desired_person,
            self.learning_rate: learning_rate
        })

    def feed_forward(self, x):
        out, states = self.sess.run([self.gait_signature, self.state], feed_dict = {self.input_tensor: x})

        return out, states

    def write_test_summary(self, err, epoch, t = 'all'):
        loss_summ = tf.Summary()
        loss_summ.value.add(
            tag = 'Classification in percent',
            simple_value = float(err))

        self.summary_writer_d[t].add_summary(loss_summ, epoch)
        self.summary_writer_d[t].flush()

    def write_summary(self, inputs, desired_person, learning_rate, write_frequency = 50):
        step = tf.train.global_step(self.sess, self.global_step)

        if step % write_frequency == 0:
            feed_dict = {
                self.input_tensor: inputs,
                self.desired_person: desired_person,
                self.learning_rate: learning_rate,
            }

            summary, loss = self.sess.run([self.ALL_SUMMARIES, self.loss], feed_dict = feed_dict)
            self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()

    def save(self, checkpoint_path, name):
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        checkpoint_name_path = os.path.join(checkpoint_path, '%s.ckpt' % name)
        all_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'GaitNN')

        saver = tf.train.Saver(all_vars)
        saver.save(self.sess, checkpoint_name_path)

    def restore(self, checkpoint_path):
        all_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'GaitNN')

        saver = tf.train.Saver(all_vars)
        saver.restore(self.sess, checkpoint_path)

    @staticmethod
    def residual_block(net, ch = 256, ch_inner = 128, scope = None, reuse = None, stride = 1):
        """
        Bottleneck v2
        """

        with slim.arg_scope([layers.convolution2d],
                            activation_fn = None,
                            normalizer_fn = None):
            with tf.variable_scope(scope, 'ResidualBlock', reuse = reuse):
                in_net = net

                if stride > 1:
                    net = layers.convolution2d(net, ch, kernel_size = 1, stride = stride)

                in_net = layers.batch_norm(in_net)
                in_net = tf.nn.relu(in_net)
                in_net = layers.convolution2d(in_net, ch_inner, 1)

                in_net = layers.batch_norm(in_net)
                in_net = tf.nn.relu(in_net)
                in_net = layers.convolution2d(in_net, ch_inner, 3, stride = stride)

                in_net = layers.batch_norm(in_net)
                in_net = tf.nn.relu(in_net)
                in_net = layers.convolution2d(in_net, ch, 1, activation_fn = None)

                net = tf.nn.relu(in_net + net)

        return net

    @abstractmethod
    def get_network(self, input_tensor, is_training, reuse = False):
        pass


class GaitNetwork(GaitNN):
    FEATURES = 512

    def __init__(self, name = None, num_of_persons = 0, recurrent_unit = 'GRU', rnn_layers = 1,
                 reuse = False, is_training = False, input_net = None):
        tf.set_random_seed(SEED)

        if num_of_persons <= 0 and is_training:
            raise Exception('Parameter num_of_persons has to be greater than zero when thaining')

        self.num_of_persons = num_of_persons
        self.rnn_layers = rnn_layers
        self.recurrent_unit = recurrent_unit

        if input_net is None:
            input_tensor = tf.placeholder(
                dtype = tf.float32,
                shape = (None, 17, 17, 32),
                name = 'input_image')
        else:
            input_tensor = input_net

        super().__init__(name, input_tensor, self.FEATURES, num_of_persons, reuse, is_training)

    def get_network(self, input_tensor, is_training, reuse = False):
        net = input_tensor

        with tf.variable_scope('GaitNN', reuse = reuse):
            with slim.arg_scope(self.get_arg_scope(is_training)):
                with tf.variable_scope('DownSampling'):
                    with tf.variable_scope('17x17'):
                        net = layers.convolution2d(net, num_outputs = 256, kernel_size = 1)
                        slim.repeat(net, 3, self.residual_block, ch = 256, ch_inner = 64)

                    with tf.variable_scope('8x8'):
                        net = self.residual_block(net, ch = 512, ch_inner = 64, stride = 2)
                        slim.repeat(net, 2, self.residual_block, ch = 512, ch_inner = 128)

                    with tf.variable_scope('4x4'):
                        net = self.residual_block(net, ch = 512, ch_inner = 128, stride = 2)
                        slim.repeat(net, 1, self.residual_block, ch = 512, ch_inner = 256)

                        net = layers.convolution2d(net, num_outputs = 256, kernel_size = 1)
                        net = layers.convolution2d(net, num_outputs = 256, kernel_size = 3)

                with tf.variable_scope('FullyConnected'):
                    # net = tf.reduce_mean(net, [1, 2], name = 'GlobalPool')
                    net = layers.flatten(net)
                    net = layers.fully_connected(net, 512, activation_fn = None, normalizer_fn = None)

                with tf.variable_scope('Recurrent', initializer = tf.contrib.layers.xavier_initializer()):
                    cell_type = {
                        'GRU': tf.nn.rnn_cell.GRUCell,
                        'LSTM': tf.nn.rnn_cell.LSTMCell
                    }

                    cell = cell_type[self.recurrent_unit](self.FEATURES)
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.rnn_layers, state_is_tuple = True)

                    net = tf.expand_dims(net, 0)
                    net, state = tf.nn.dynamic_rnn(cell, net, initial_state = cell.zero_state(1, dtype = tf.float32))
                    net = tf.reshape(net, [-1, self.FEATURES])

                    # Temporal Avg-Pooling
                    gait_signature = tf.reduce_mean(net, 0)

                if is_training:
                    net = tf.expand_dims(gait_signature, 0)
                    net = layers.dropout(net, 0.7)

                    with tf.variable_scope('Logits'):
                        net = layers.fully_connected(net, self.num_of_persons, activation_fn = None,
                                                     normalizer_fn = None)

                return net, gait_signature, state
