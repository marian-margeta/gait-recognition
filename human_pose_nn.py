import tensorflow as tf
import numpy as np
import part_detector
import settings
import utils
import os

from abc import abstractmethod
from functools import lru_cache
from scipy.stats import norm

from inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2

import tensorflow.contrib.layers as layers

slim = tf.contrib.slim

SUMMARY_PATH = settings.LOGDIR_PATH

KEY_SUMMARIES = tf.GraphKeys.SUMMARIES
KEY_SUMMARIES_PER_JOINT = ['summary_joint_%02d' % i for i in range(16)]


class HumanPoseNN(object):
    """
    The neural network used for pose estimation.
    """

    def __init__(self, log_name, heatmap_size, image_size, loss_type = 'SCE', is_training = True):
        tf.set_random_seed(0)

        if loss_type not in { 'MSE', 'SCE' }:
            raise NotImplementedError('Loss function should be either MSE or SCE!')

        self.log_name = log_name
        self.heatmap_size = heatmap_size
        self.image_size = image_size
        self.is_train = is_training
        self.loss_type = loss_type

        # Initialize placeholders
        self.input_tensor = tf.placeholder(
            dtype = tf.float32,
            shape = (None, image_size, image_size, 3),
            name = 'input_image')

        self.present_joints = tf.placeholder(
            dtype = tf.float32,
            shape = (None, 16),
            name = 'present_joints')

        self.inside_box_joints = tf.placeholder(
            dtype = tf.float32,
            shape = (None, 16),
            name = 'inside_box_joints')

        self.desired_heatmap = tf.placeholder(
            dtype = tf.float32,
            shape = (None, heatmap_size, heatmap_size, 16),
            name = 'desired_heatmap')

        self.desired_points = tf.placeholder(
            dtype = tf.float32,
            shape = (None, 2, 16),
            name = 'desired_points')

        self.network = self.pre_process(self.input_tensor)
        self.network, self.feature_tensor = self.get_network(self.network, is_training)

        self.sigm_network = tf.sigmoid(self.network)
        self.smoothed_sigm_network = self._get_gauss_smoothing_net(self.sigm_network, std = 0.7)

        self.loss_err = self._get_loss_function(loss_type)
        self.euclidean_dist = self._euclidean_dist_err()
        self.euclidean_dist_per_joint = self._euclidean_dist_per_joint_err()

        if is_training:
            self.global_step = tf.Variable(0, name = 'global_step', trainable = False)

            self.learning_rate = tf.placeholder(
                dtype = tf.float32,
                shape = [],
                name = 'learning_rate')

            self.optimize = layers.optimize_loss(loss = self.loss_err,
                                                 global_step = self.global_step,
                                                 learning_rate = self.learning_rate,
                                                 optimizer = tf.train.RMSPropOptimizer(self.learning_rate),
                                                 clip_gradients = 2.0
                                                 )

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if log_name is not None:
            self._init_summaries()

    def _init_summaries(self):
        if self.is_train:
            logdir = os.path.join(SUMMARY_PATH, self.log_name, 'train')

            self.summary_writer = tf.summary.FileWriter(logdir)
            self.summary_writer_by_points = [tf.summary.FileWriter(os.path.join(logdir, 'point_%02d' % i))
                                             for i in range(16)]

            tf.scalar_summary('Average euclidean distance', self.euclidean_dist, collections = [KEY_SUMMARIES])

            for i in range(16):
                tf.scalar_summary('Joint euclidean distance', self.euclidean_dist_per_joint[i],
                                  collections = [KEY_SUMMARIES_PER_JOINT[i]])

            self.create_summary_from_weights()

            self.ALL_SUMMARIES = tf.merge_all_summaries(KEY_SUMMARIES)
            self.SUMMARIES_PER_JOINT = [tf.merge_all_summaries(KEY_SUMMARIES_PER_JOINT[i]) for i in range(16)]
        else:
            logdir = os.path.join(SUMMARY_PATH, self.log_name, 'test')
            self.summary_writer = tf.summary.FileWriter(logdir)

    def _get_loss_function(self, loss_type):
        loss_dict = {
            'MSE': self._loss_mse(),
            'SCE': self._loss_cross_entropy()
        }

        return loss_dict[loss_type]

    @staticmethod
    @lru_cache()
    def _get_gauss_filter(size = 15, std = 1.0, kernel_sum = 1.0):
        samples = norm.pdf(np.linspace(-2, 2, size), 0, std)
        samples /= np.sum(samples)
        samples *= kernel_sum ** 0.5

        samples = np.expand_dims(samples, 0)
        weights = np.zeros(shape = (1, size, 16, 1), dtype = np.float32)

        for i in range(16):
            weights[:, :, i, 0] = samples

        return weights

    @staticmethod
    def _get_gauss_smoothing_net(net, size = 15, std = 1.0, kernel_sum = 1.0):
        filter_h = HumanPoseNN._get_gauss_filter(size, std, kernel_sum)
        filter_v = filter_h.swapaxes(0, 1)

        net = tf.nn.depthwise_conv2d(net, filter = filter_h, strides = [1, 1, 1, 1], padding = 'SAME',
                                     name = 'SmoothingHorizontal')

        net = tf.nn.depthwise_conv2d(net, filter = filter_v, strides = [1, 1, 1, 1], padding = 'SAME',
                                     name = 'SmoothingVertical')

        return net

    def generate_output(self, shape, presented_parts, labels, sigma):
        heatmap_dict = {
            'MSE': utils.get_gauss_heat_map(
                shape = shape, is_present = presented_parts,
                mean = labels, sigma = sigma),
            'SCE': utils.get_binary_heat_map(
                shape = shape, is_present = presented_parts,
                centers = labels, diameter = sigma)
        }

        return heatmap_dict[self.loss_type]

    def _adjust_loss(self, loss_err):
        # Shape: [batch, joints]
        loss = tf.reduce_sum(loss_err, [1, 2])

        # Stop error propagation of joints that are not presented
        loss = tf.multiply(loss, self.present_joints)

        # Compute average loss of presented joints
        num_of_visible_joints = tf.reduce_sum(self.present_joints)
        loss = tf.reduce_sum(loss) / num_of_visible_joints

        return loss

    def _loss_mse(self):
        sq = tf.squared_difference(self.sigm_network, self.desired_heatmap)
        loss = self._adjust_loss(sq)

        return loss

    def _loss_cross_entropy(self):
        ce = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.network, labels = self.desired_heatmap)
        loss = self._adjust_loss(ce)

        return loss

    def _joint_highest_activations(self):
        highest_activation = tf.reduce_max(self.smoothed_sigm_network, [1, 2])

        return highest_activation

    def _joint_positions(self):
        highest_activation = tf.reduce_max(self.sigm_network, [1, 2])
        x = tf.argmax(tf.reduce_max(self.smoothed_sigm_network, 1), 1)
        y = tf.argmax(tf.reduce_max(self.smoothed_sigm_network, 2), 1)

        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        a = tf.cast(highest_activation, tf.float32)

        scale_coef = (self.image_size / self.heatmap_size)
        x *= scale_coef
        y *= scale_coef

        out = tf.stack([y, x, a])

        return out

    def _euclidean_dist_err(self):
        # Work only with joints that are presented inside frame
        l2_dist = tf.multiply(self.euclidean_distance(), self.inside_box_joints)

        # Compute average loss of presented joints
        num_of_visible_joints = tf.reduce_sum(self.inside_box_joints)
        l2_dist = tf.reduce_sum(l2_dist) / num_of_visible_joints

        return l2_dist

    def _euclidean_dist_per_joint_err(self):
        # Work only with joints that are presented inside frame
        l2_dist = tf.multiply(self.euclidean_distance(), self.inside_box_joints)

        # Average euclidean distance of presented joints
        present_joints = tf.reduce_sum(self.inside_box_joints, 0)
        err = tf.reduce_sum(l2_dist, 0) / present_joints

        return err

    def _restore(self, checkpoint_path, variables):
        saver = tf.train.Saver(variables)
        saver.restore(self.sess, checkpoint_path)

    def _save(self, checkpoint_path, name, variables):
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        checkpoint_name_path = os.path.join(checkpoint_path, '%s.ckpt' % name)

        saver = tf.train.Saver(variables)
        saver.save(self.sess, checkpoint_name_path)

    def euclidean_distance(self):
        x = tf.argmax(tf.reduce_max(self.smoothed_sigm_network, 1), 1)
        y = tf.argmax(tf.reduce_max(self.smoothed_sigm_network, 2), 1)

        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        dy = tf.squeeze(self.desired_points[:, 0, :])
        dx = tf.squeeze(self.desired_points[:, 1, :])

        sx = tf.squared_difference(x, dx)
        sy = tf.squared_difference(y, dy)

        l2_dist = tf.sqrt(sx + sy)

        return l2_dist

    def feed_forward(self, x):
        out = self.sess.run(self.sigm_network, feed_dict = {
            self.input_tensor: x
        })

        return out

    def heat_maps(self, x):
        out = self.sess.run(self.smoothed_sigm_network, feed_dict = {
            self.input_tensor: x
        })

        return out

    def feed_forward_pure(self, x):
        out = self.sess.run(self.network, feed_dict = {
            self.input_tensor: x
        })

        return out

    def feed_forward_features(self, x):
        out = self.sess.run(self.feature_tensor, feed_dict = {
            self.input_tensor: x,
        })

        return out

    def test_euclidean_distance(self, x, points, present_joints, inside_box_joints):
        err = self.sess.run(self.euclidean_dist, feed_dict = {
            self.input_tensor: x,
            self.desired_points: points,
            self.present_joints: present_joints,
            self.inside_box_joints: inside_box_joints
        })

        return err

    def test_joint_distances(self, x, y):
        err = self.sess.run(self.euclidean_distance(), feed_dict = {
            self.input_tensor: x,
            self.desired_points: y
        })

        return err

    def test_joint_activations(self, x):
        err = self.sess.run(self._joint_highest_activations(), feed_dict = {
            self.input_tensor: x
        })

        return err

    def estimate_joints(self, x):
        out = self.sess.run(self._joint_positions(), feed_dict = {
            self.input_tensor: x
        })

        return out

    def train(self, x, heatmaps, present_joints, learning_rate, is_inside_box):
        if not self.is_train:
            raise Exception('Network is not in train mode!')

        self.sess.run(self.optimize, feed_dict = {
            self.input_tensor: x,
            self.desired_heatmap: heatmaps,
            self.present_joints: present_joints,
            self.learning_rate: learning_rate,
            self.inside_box_joints: is_inside_box
        })

    def write_test_summary(self, epoch, loss):
        loss_sum = tf.Summary()
        loss_sum.value.add(
            tag = 'Average Euclidean Distance',
            simple_value = float(loss))
        self.summary_writer.add_summary(loss_sum, epoch)
        self.summary_writer.flush()

    def write_summary(self, inp, desired_points, heatmaps, present_joints, learning_rate, is_inside_box,
                      write_frequency = 20, write_per_joint_frequency = 100):
        step = tf.train.global_step(self.sess, self.global_step)

        if step % write_frequency == 0:
            feed_dict = {
                self.input_tensor: inp,
                self.desired_points: desired_points,
                self.desired_heatmap: heatmaps,
                self.present_joints: present_joints,
                self.learning_rate: learning_rate,
                self.inside_box_joints: is_inside_box
            }

            summary, loss = self.sess.run([self.ALL_SUMMARIES, self.loss_err], feed_dict = feed_dict)
            self.summary_writer.add_summary(summary, step)

            if step % write_per_joint_frequency == 0:
                summaries = self.sess.run(self.SUMMARIES_PER_JOINT, feed_dict = feed_dict)

                for i in range(16):
                    self.summary_writer_by_points[i].add_summary(summaries[i], step)

                for i in range(16):
                    self.summary_writer_by_points[i].flush()

            self.summary_writer.flush()

    @abstractmethod
    def pre_process(self, inp):
        pass

    @abstractmethod
    def get_network(self, input_tensor, is_training):
        pass

    @abstractmethod
    def create_summary_from_weights(self):
        pass


class HumanPoseIRNetwork(HumanPoseNN):
    """
    The first part of our network that exposes as an extractor of spatial features. It s derived from
    Inception-Resnet-v2 architecture and modified for generating heatmaps - i.e. dense predictions of body joints.
    """

    FEATURES = 32
    IMAGE_SIZE = 299
    HEATMAP_SIZE = 289
    POINT_DIAMETER = 15
    SMOOTH_SIZE = 21

    def __init__(self, log_name = None, loss_type = 'SCE', is_training = False):
        super().__init__(log_name, self.HEATMAP_SIZE, self.IMAGE_SIZE, loss_type, is_training)

    def pre_process(self, inp):
        return ((inp / 255) - 0.5) * 2.0

    def get_network(self, input_tensor, is_training):
        # Load pre-trained inception-resnet model
        with slim.arg_scope(inception_resnet_v2_arg_scope(batch_norm_decay = 0.999, weight_decay = 0.0001)):
            net, end_points = inception_resnet_v2(input_tensor, is_training = is_training)

        # Adding some modification to original InceptionResnetV2 - changing scoring of AUXILIARY TOWER
        weight_decay = 0.0005
        with tf.variable_scope('NewInceptionResnetV2'):
            with tf.variable_scope('AuxiliaryScoring'):
                with slim.arg_scope([layers.convolution2d, layers.convolution2d_transpose],
                                    weights_regularizer = slim.l2_regularizer(weight_decay),
                                    biases_regularizer = slim.l2_regularizer(weight_decay),
                                    activation_fn = None):
                    tf.summary.histogram('Last_layer/activations', net, [KEY_SUMMARIES])

                    # Scoring
                    net = slim.dropout(net, 0.7, is_training = is_training, scope = 'Dropout')
                    net = layers.convolution2d(net, num_outputs = self.FEATURES, kernel_size = 1, stride = 1,
                                               scope = 'Scoring_layer')
                    feature = net
                    tf.summary.histogram('Scoring_layer/activations', net, [KEY_SUMMARIES])

                    # Upsampling
                    net = layers.convolution2d_transpose(net, num_outputs = 16, kernel_size = 17, stride = 17,
                                                         padding = 'VALID', scope = 'Upsampling_layer')

                    tf.summary.histogram('Upsampling_layer/activations', net, [KEY_SUMMARIES])

            # Smoothing layer - separable gaussian filters
            net = super()._get_gauss_smoothing_net(net, size = self.SMOOTH_SIZE, std = 1.0, kernel_sum = 0.2)

            return net, feature

    def restore(self, checkpoint_path, is_pre_trained_imagenet_checkpoint = False):
        all_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'InceptionResnetV2')
        if not is_pre_trained_imagenet_checkpoint:
            all_vars += tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'NewInceptionResnetV2/AuxiliaryScoring')

        super()._restore(checkpoint_path, all_vars)

    def save(self, checkpoint_path, name):
        all_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'InceptionResnetV2')
        all_vars += tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'NewInceptionResnetV2/AuxiliaryScoring')

        super()._save(checkpoint_path, name, all_vars)

    def create_summary_from_weights(self):
        with tf.variable_scope('NewInceptionResnetV2/AuxiliaryScoring', reuse = True):
            tf.summary.histogram('Scoring_layer/biases', tf.get_variable('Scoring_layer/biases'), [KEY_SUMMARIES])
            tf.summary.histogram('Upsampling_layer/biases', tf.get_variable('Upsampling_layer/biases'), [KEY_SUMMARIES])
            tf.summary.histogram('Scoring_layer/weights', tf.get_variable('Scoring_layer/weights'), [KEY_SUMMARIES])
            tf.summary.histogram('Upsampling_layer/weights', tf.get_variable('Upsampling_layer/weights'),
                                 [KEY_SUMMARIES])

        with tf.variable_scope('InceptionResnetV2/AuxLogits', reuse = True):
            tf.summary.histogram('Last_layer/weights', tf.get_variable('Conv2d_2a_5x5/weights'), [KEY_SUMMARIES])
            tf.summary.histogram('Last_layer/beta', tf.get_variable('Conv2d_2a_5x5/BatchNorm/beta'), [KEY_SUMMARIES])
            tf.summary.histogram('Last_layer/moving_mean', tf.get_variable('Conv2d_2a_5x5/BatchNorm/moving_mean'),
                                 [KEY_SUMMARIES])


class PartDetector(HumanPoseNN):
    """
    Architecture of Part Detector network, as was described in https://arxiv.org/abs/1609.01743
    """

    IMAGE_SIZE = 256
    HEATMAP_SIZE = 256
    POINT_DIAMETER = 11

    def __init__(self, log_name = None, init_from_checkpoint = None, loss_type = 'SCE', is_training = False):
        if init_from_checkpoint is not None:
            part_detector.init_model_variables(init_from_checkpoint, is_training)
            self.reuse = True
        else:
            self.reuse = False

        super().__init__(log_name, self.HEATMAP_SIZE, self.IMAGE_SIZE, loss_type, is_training)

    def pre_process(self, inp):
        return inp / 255

    def create_summary_from_weights(self):
        pass

    def restore(self, checkpoint_path):
        all_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope = 'HumanPoseResnet')
        all_vars += tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'NewHumanPoseResnet/Scoring')

        super()._restore(checkpoint_path, all_vars)

    def save(self, checkpoint_path, name):
        all_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'HumanPoseResnet')
        all_vars += tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'NewHumanPoseResnet/Scoring')

        super()._save(checkpoint_path, name, all_vars)

    def get_network(self, input_tensor, is_training):
        net_end, end_points = part_detector.human_pose_resnet(input_tensor, reuse = self.reuse, training = is_training)

        return net_end, end_points['features']
