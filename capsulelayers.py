import keras.backend as K
import tensorflow as tf
from keras import initializers, layers, regularizers
from keras.utils.conv_utils import conv_output_length
import numpy as np
from keras.layers import InputSpec
from keras.regularizers import l2


class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()   # config = super().get_config()   # super 调用父类
        return config


class Mask_CID(layers.Layer):

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, a = inputs
            mask = K.argmax(a, 1)
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # mask.shape=[None] 索引值
            mask = K.argmax(x, 1)

        # 从start开始，增量是delta，小于或大于limit. [0, 1, ......]
        increasing = tf.range(start=0, limit=tf.shape(inputs)[0], delta=1)
        m = tf.stack([increasing, tf.cast(mask, tf.int32)], axis=1)
        # increasing.shape=[None]
        # mask.shape=[None]
        # m.shape=[None, 2]
        # inputs.shape=[None, num_capsule, dim_capsule]
        # masked.shape=[None, dim_capsule] 从inputs中选出length最大的vector，因为本文提出的decoder网络是只输入最长的vector
        # x1 = tf.transpose(inputs, (0))
        masked = tf.gather_nd(inputs, m)

        return masked

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape[0], tuple):  # true label provided
            return tuple([None, input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[2]])


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
    
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, axis=1)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvCapsuleLayer3D(layers.Layer):

    def __init__(self, kernel_size, num_capsule, num_atoms, strides=1, padding='valid', routings=3,
                 kernel_initializer='he_normal', **kwargs):
        super(ConvCapsuleLayer3D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.strides = strides
        self.padding = padding
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        # Transform matrix
        self.W = self.add_weight(shape=[self.input_num_atoms, self.kernel_size, self.kernel_size, 1, self.num_capsule * self.num_atoms],
                                 initializer=self.kernel_initializer,
                                 name='W')
        # print(self.W.shape)

        self.b = self.add_weight(shape=[self.num_capsule, self.num_atoms, 1, 1],
                                 initializer=initializers.constant(0.1),
                                 name='b')

        self.built = True

    def call(self, input_tensor, training=None):

        # input_transposed.shape=[None, input_num_capsule, input_num_atoms, input_height, input_width]
        # input_shape=(None, inout_num_capsule, input_num_atoms, input_height, input_width)
        input_transposed = tf.transpose(input_tensor, [0, 3, 4, 1, 2])
        input_shape = K.shape(input_transposed)
        input_tensor_reshaped = K.reshape(input_tensor, [input_shape[0], 1, self.input_num_capsule * self.input_num_atoms, self.input_height, self.input_width])

        input_tensor_reshaped.set_shape((None, 1, self.input_num_capsule * self.input_num_atoms, self.input_height, self.input_width))

        # conv = Conv3D(input_tensor_reshaped, self.W, (self.strides, self.strides),
        #                 padding=self.padding, data_format='channels_first')
        conv = K.conv3d(input_tensor_reshaped, self.W, strides=(self.input_num_atoms, self.strides, self.strides), padding=self.padding, data_format='channels_first')
        # kernel: kernel_shape + [in_channels, out_channels]
        # conv = layers.BatchNormalization(momentum=0.9, name='convcaps_bn')(conv)
        # conv = layers.Activation('relu')(conv)

        votes_shape = K.shape(conv)
        # conv.shape=[None, 256, 32, 7, 7]
        # num_capsule=input_num_capsule=32, num_atoms=input_num_atoms=8
        # num_atoms*num_capsule=input_num_atoms*input_num_capsule=256
        _, _, _, conv_height, conv_width = conv.get_shape()
        conv = tf.transpose(conv, [0, 2, 1, 3, 4]) # [None, 32, 256, 7, 7]
        votes = K.reshape(conv, [input_shape[0], self.input_num_capsule, self.num_capsule, self.num_atoms, votes_shape[3], votes_shape[4]]) # [None, 32, 8, 8, 7, 7]
        votes.set_shape((None, self.input_num_capsule, self.num_capsule, self.num_atoms, conv_height.value, conv_width.value))

        logit_shape = K.stack([input_shape[0], self.input_num_capsule, self.num_capsule, votes_shape[3], votes_shape[4]]) # [None 32 32 7 7]
        biases_replicated = K.tile(self.b, [1, 1, conv_height.value, conv_width.value]) # [32, 8, 7, 7]

        activations = update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_num_capsule,
            output_dim=self.num_capsule,
            num_routing=self.routings)

        a2 = tf.transpose(activations, [0, 3, 4, 1, 2])
        return a2

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-2]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(space[i], self.kernel_size, padding=self.padding, stride=self.strides, dilation=1)
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_capsule, self.num_atoms)

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'num_capsule': self.num_capsule,
            'num_atoms': self.num_atoms,
            'strides': self.strides,
            'padding': self.padding,
            'routings': self.routings,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(ConvCapsuleLayer3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FlattenCaps(layers.Layer):

    def __init__(self, **kwargs):
        super(FlattenCaps, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=4)  # minimum rank of the input is 4.

    def compute_output_shape(self, input_shape):
        # all(iteration) 用于判断给定的迭代参数iteration中的所有元素是否都为True，如果是则返回True，否则返回False
        # 元素除了0、空、None、False外都算True
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "FlattenCaps" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '. '
                             'Make sure to pass a complete "input_shape" '
                             'or "batch_input_shape" argument to the first '
                             'layer in your model.')
        return (input_shape[0], np.prod(input_shape[1:-1]), input_shape[-1])

    def call(self, inputs):
        shape = K.int_shape(inputs)
        return K.reshape(inputs, (-1, np.prod(shape[1:-1]), shape[-1]))


# votes = [None, 32, 8, 8, 7, 7]
# logit_shape = [None 32 32 7 7]
# biases = [32, 8, 7, 7]
def update_routing(votes, biases, logit_shape, num_dims, input_dim, output_dim,
                   num_routing):
    if num_dims == 6:
        votes_t_shape = [3, 0, 1, 2, 4, 5] # 3: self.num_atoms
        r_t_shape = [1, 2, 3, 0, 4, 5]
    elif num_dims == 4:
        votes_t_shape = [3, 0, 1, 2]
        r_t_shape = [1, 2, 3, 0]
    else:
        raise NotImplementedError('Not implemented')

    votes_trans = tf.transpose(votes, votes_t_shape) # [8, None, 32, 8, 7, 7]
    # print(votes_trans.get_shape())
    # votes_trans.get_shape() = (8, None, 32, 32, 2, 2)
    # height=32, width=2, caps=2
    # 应该是: _, _, _, caps, height, width = votes_trans.get_shape()
    # _, _, _, height, width, caps = votes_trans.get_shape()

    def _body(i, logits, activations):
        """Routing while loop."""
        # route: [batch, input_dim, output_dim, ...]
        route = tf.nn.softmax(logits, axis=2)
        # print(route.shape)
        preactivate_unrolled = route * votes_trans
        # print(votes_trans.shape)
        # print(preactivate_unrolled.shape)
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        # print(preact_trans.shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
        # print(preactivate.shape)
        activation = squash(preactivate)
        activations = activations.write(i, activation)

        act_3d = K.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        distances = tf.reduce_sum(votes * act_replicated, axis=3)
        logits += distances
        return (i + 1, logits, activations)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)

    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations = tf.while_loop(
        lambda i, logits, activations: i < num_routing,
        _body,
        loop_vars=[i, logits, activations],
        swap_memory=True)
    a = K.cast(activations.read(num_routing - 1), dtype='float32')
    return K.cast(activations.read(num_routing - 1), dtype='float32')

def PrimaryCap_2D(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D 'n_channels' times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_vector: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, height, width, num_capsule, dim_capsule]
    """

    output = layers.Conv2D(filters=dim_vector * n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(0.0001),
                           name='primarycap_conv2d')(inputs)
    output = layers.BatchNormalization(momentum=0.9, name='primarycap_bn')(output)
    output = layers.Activation('relu', name='primarycap_relu')(output)
    shape = np.shape(output)
    # print(output.shape)

    outputs = layers.Reshape(target_shape=[shape[1].value, shape[2].value, n_channels, dim_vector], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors