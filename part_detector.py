import tensorflow as tf
import tensorflow.contrib.layers as layers
import torchfile as th


def init_model_variables(file_path, trainable = True):
    """
    Initialize all model variables of a given torch model. The torch model pre-trained on MPII or MPII+LSP can be
    downloaded from author's pages: https://www.adrianbulat.com/human-pose-estimation

    :param file_path: path to serialized torch model (.th)
    :param trainable: if the loaded variables should be trainable
    """

    def load_conv2(obj, scope = 'Conv'):
        with tf.variable_scope(scope, reuse = False):
            w = obj[b'weight'].swapaxes(0, 3).swapaxes(1, 2).swapaxes(0, 1)
            b = obj[b'bias']

            tf.get_variable('weights', w.shape, initializer = tf.constant_initializer(w), trainable = trainable)
            tf.get_variable('biases', b.shape, initializer = tf.constant_initializer(b), trainable = trainable)

    def load_batch_norm(obj, scope = 'BatchNorm'):
        with tf.variable_scope(scope, reuse = False):
            gamma = obj[b'weight']
            beta = obj[b'bias']
            mean = obj[b'running_mean']
            var = obj[b'running_var']

            tf.get_variable('gamma', gamma.shape, dtype = tf.float32, initializer = tf.constant_initializer(gamma),
                            trainable = trainable)
            tf.get_variable('beta', beta.shape, dtype = tf.float32, initializer = tf.constant_initializer(beta),
                            trainable = trainable)

            tf.get_variable('moving_variance', var.shape, dtype = tf.float32,
                            initializer = tf.constant_initializer(var), trainable = False)
            tf.get_variable('moving_mean', mean.shape, dtype = tf.float32, initializer = tf.constant_initializer(mean),
                            trainable = False)

    def load_bottlenecks(bottlenecks):
        for idx, bottleneck in enumerate(bottlenecks):
            with tf.variable_scope('Bottleneck_%d' % idx, reuse = False):
                connections = bottleneck[b'modules'][0][b'modules']

                res_conn = connections[0][b'modules']
                skip_conn = connections[1][b'modules']

                # Load skip connection
                if idx == 0:
                    # Skip connection involves conv + batch norm
                    load_conv2(skip_conn[0], scope = 'Conv_skip')
                    load_batch_norm(skip_conn[1], scope = 'BatchNorm_skip')

                # Load residual connection
                for l in range(3):
                    load_conv2(res_conn[l * 3], scope = 'Conv_%d' % (l + 1))
                    load_batch_norm(res_conn[l * 3 + 1], scope = 'BatchNorm_%d' % (l + 1))

    file = th.load(file_path)

    with tf.variable_scope('HumanPoseResnet', reuse = False):
        resnet = file[b'modules'][0][b'modules'][1][b'modules']

        with tf.variable_scope('Block_0', reuse = False):
            load_conv2(resnet[0])
            load_batch_norm(resnet[1])

        for i in range(4):
            with tf.variable_scope('Block_%d' % (i + 1), reuse = False):
                load_bottlenecks(resnet[i + 4][b'modules'])

        with tf.variable_scope('Block_5', reuse = False):
            load_conv2(resnet[8])
            # Transpose convolution
            load_conv2(resnet[9], scope = 'Conv2d_transpose')


def human_pose_resnet(net, reuse = False, training = False):
    """
    Architecture of Part Detector network, as was described in https://arxiv.org/abs/1609.01743
    
    :param net: input tensor
    :param reuse: whether reuse variables or not. Use False if the variables are initialized with init_model_variables
    :param training: if the variables should be trainable. It has no effect if the 'reuse' param is set to True
    :return: output tensor and dictionary of named endpoints
    """

    def batch_normalization(input_net, act_f = None, scope = None):
        return layers.batch_norm(input_net, center = True, scale = True, epsilon = 1e-5,
                                 activation_fn = act_f, is_training = training,
                                 scope = scope)

    def conv_2d(input_net, num_outputs, kernel_size, stride = 1, padding_mod = 'SAME', scope = None):
        return layers.convolution2d(input_net, num_outputs = num_outputs, kernel_size = kernel_size,
                                    stride = stride, padding = padding_mod,
                                    activation_fn = None, scope = scope)

    def padding(input_net, w, h):
        return tf.pad(input_net, [[0, 0], [h, h], [w, w], [0, 0]], "CONSTANT")

    def bottleneck(input_net, depth, depth_bottleneck, stride, i):
        with tf.variable_scope('Bottleneck_%d' % i, reuse = reuse):
            res_conv = stride > 1 or stride < 0
            stride = abs(stride)

            # Res connection
            out_net = conv_2d(input_net, num_outputs = depth_bottleneck, kernel_size = 1,
                              stride = 1, padding_mod = 'VALID', scope = 'Conv_1')

            out_net = batch_normalization(out_net, tf.nn.relu, 'BatchNorm_1')

            out_net = padding(out_net, 1, 1)

            out_net = conv_2d(out_net, num_outputs = depth_bottleneck, kernel_size = 3,
                              stride = stride, padding_mod = 'VALID', scope = 'Conv_2')

            out_net = batch_normalization(out_net, tf.nn.relu, 'BatchNorm_2')

            out_net = conv_2d(out_net, num_outputs = depth, kernel_size = 1,
                              stride = 1, padding_mod = 'VALID', scope = 'Conv_3')

            out_net = batch_normalization(out_net, scope = 'BatchNorm_3')

            # Skip connection
            if res_conv:
                input_net = conv_2d(input_net, num_outputs = depth, kernel_size = 1,
                                    stride = stride, padding_mod = 'VALID', scope = 'Conv_skip')

                input_net = batch_normalization(input_net, scope = 'BatchNorm_skip')

            out_net += input_net
            out_net = tf.nn.relu(out_net)

            return out_net

    def repeat_bottleneck(input_net, all_params):
        for i, (depth, depth_bottleneck, stride) in enumerate(all_params):
            input_net = bottleneck(input_net, depth, depth_bottleneck, stride, i)

        return input_net

    end_points = { }

    with tf.variable_scope('HumanPoseResnet', reuse = reuse):
        with tf.variable_scope('Block_0', reuse = reuse):
            net = padding(net, 3, 3)

            net = conv_2d(net, num_outputs = 64, kernel_size = 7, stride = 2, padding_mod = 'VALID')

            net = batch_normalization(net, tf.nn.relu)

            net = padding(net, 1, 1)

            net = layers.max_pool2d(net, 3, 2, padding = 'VALID')

        with tf.variable_scope('Block_1', reuse = reuse):
            net = repeat_bottleneck(net, [(256, 64, -1)] + [(256, 64, 1)] * 2)

        with tf.variable_scope('Block_2', reuse = reuse):
            net = repeat_bottleneck(net, [(512, 128, 2)] + [(512, 128, 1)] * 7)

        with tf.variable_scope('Block_3', reuse = reuse):
            net = repeat_bottleneck(net, [(1024, 256, 2)] + [(1024, 256, 1)] * 35)

        with tf.variable_scope('Block_4', reuse = reuse):
            net = repeat_bottleneck(net, [(2048, 512, -1)] + [(2048, 512, 1)] * 2)

        end_points['resnet_end'] = net
        with tf.variable_scope('Block_5', reuse = reuse):
            net = conv_2d(net, num_outputs = 16, kernel_size = 1, stride = 1, padding_mod = 'VALID')
            end_points['features'] = net

            net = layers.convolution2d_transpose(net, num_outputs = 16, kernel_size = 16, stride = 16,
                                                 activation_fn = None, padding = 'VALID')

            # net = tf.nn.sigmoid(net)

        return net, end_points

# with tf.Graph().as_default():
#     init_model_variables('/home/margeta/data/hp.t7')
#
#     input_tensor = tf.placeholder(tf.float32, shape = (None, 256, 256, 3), name = 'input_image')
#     hp_net = human_pose_resnet(input_tensor, reuse = True, training = False)
#
#     # config = tf.ConfigProto()
#     # config.gpu_options.per_process_gpu_memory_fraction = 0.5
#     # sess = tf.Session(config=config)
#     sess = tf.Session()
#     sess.run(tf.initialize_all_variables())
#     print('Model was loaded!')
#
#     img = np.reshape(th.load('img').swapaxes(0, 1).swapaxes(1, 2), [-1, 256, 256, 3])
#
#     res = sess.run(hp_net, feed_dict = {input_tensor: img})
#
#     res = np.squeeze(res)
#
#     print(res.shape)
#     print(www.shape)
#     print(res[200,160,:])
#     print(www[200,160,:])
#
# img = res[:,:,0]
# fig = plt.figure()
# plt.imshow(img)
# fig.savefig('img.png')
#
