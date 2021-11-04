import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.layers.core import Reshape
from keras.regularizers import l2
from capsulelayers import CapsuleLayer, Length, Mask_CID, ConvCapsuleLayer3D, \
    FlattenCaps, PrimaryCap_2D


K.set_image_data_format('channels_last')

class CapsnetBuilder_2D_Deconv(object):
    @staticmethod
    def build(input_shape, n_class, routings, c):
        """
        A Capsule Network
        :param input_shape: data shape, 3d, [width, height, channels]
        :param n_class: number of classes
        :param routings: number of routing iterations
        :return: Two Keras Models, the first one used for training, and the second one for evaluation.
                'eval_model' can also be used for training.
        """
        x = layers.Input(shape=input_shape)

        # Layer 1: Just a conventional Conv2D layer
        conv1 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid',
                              kernel_initializer='he_normal',
                              kernel_regularizer=l2(0.0001),
                              name='conv1')(x)
        conv1 = layers.BatchNormalization(momentum=0.9, name='bn1')(conv1)
        conv1 = layers.Activation('relu')(conv1)

        # Layer 2: Just a conventional Conv2D layer
        conv2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid',
                              kernel_initializer='he_normal',
                              kernel_regularizer=l2(0.0001),
                              name='conv2')(conv1)
        conv2 = layers.BatchNormalization(momentum=0.9, name='bn2')(conv2)
        conv2 = layers.Activation('relu')(conv2)

        # Layer 3: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primarycaps = PrimaryCap_2D(conv2, dim_vector=8, n_channels=16, kernel_size=3, strides=2, padding='valid')

        # Layer 4: 3D Conv Capsule layer. Routing algorithm works here.
        l_skip = ConvCapsuleLayer3D(kernel_size=3, num_capsule=8, num_atoms=8, strides=2, padding='same', routings=3)(
            primarycaps)
        l_skip = FlattenCaps()(l_skip)

        # Layer 5: Capsule layer. Routing algorithm works here.
        digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                                 name='digitcaps')(l_skip)

        # Layer 6: classifier
        out_caps = Length(name='capsnet')(digitcaps)

        # Decoder network.
        y = layers.Input(shape=(n_class,))
        masked_by_y = Mask_CID()([digitcaps, y])  # The true label is used to mask the output of capsule layer.

        decoder = models.Sequential(name='decoder')
        decoder.add(layers.Dense(units=7 * 7 * 16, input_dim=16,
                                 activation='relu',
                                 kernel_initializer='he_normal',
                                 ))
        decoder.add(Reshape((7, 7, 16)))

        decoder.add(layers.BatchNormalization(momentum=0.9))
        decoder.add(layers.Deconvolution2D(filters=64, kernel_size=3, strides=1,
                                           kernel_initializer='he_normal',
                                           kernel_regularizer=l2(0.0001),
                                           # padding='same',
                                           activation='relu',
                                           ))
        decoder.add(layers.Deconvolution2D(filters=32, kernel_size=3, strides=1,
                                           kernel_initializer='he_normal',
                                           kernel_regularizer=l2(0.0001),
                                           activation='relu',
                                           ))
        decoder.add(layers.Deconvolution2D(filters=16, kernel_size=3, strides=1,
                                           kernel_initializer='he_normal',
                                           kernel_regularizer=l2(0.0001),
                                           activation='relu',
                                           ))
        decoder.add(layers.Deconvolution2D(filters=c, kernel_size=3, strides=1,
                                           kernel_initializer='he_normal',
                                           kernel_regularizer=l2(0.0001),
                                           padding='same',
                                           activation='relu',
                                           ))
        decoder.add(Reshape(target_shape=(13, 13, c), name='out_recon'))

        # Models for training and evaluation (prediction)
        train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
        eval_model = models.Model(x, out_caps)

        return train_model, eval_model


    @staticmethod
    def build_capsnet(input_shape, n_class, routings, c):
        return CapsnetBuilder_2D_Deconv.build(input_shape, n_class, routings, c)


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def main():
    model, eval_model = CapsnetBuilder_2D_Deconv.build_capsnet(input_shape=(13, 13, 176), n_class=13, routings=3, c=176)
    model.compile(optimizer=optimizers.Adam(lr=0.001),
                       loss=[margin_loss, 'mae'],
                       loss_weights=[1., 0.01],
                       metrics={'capsnet': 'accuracy'})


    model.summary(positions=[.33, .61, .75, 1.])
    print(model.metrics_names) # ['loss', 'capsnet_loss', 'decoder_loss', 'capsnet_acc']

if __name__ == '__main__':
    main()