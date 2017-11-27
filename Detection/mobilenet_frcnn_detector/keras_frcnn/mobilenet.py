# -*- coding: utf-8 -*-
"""MobileNet model for Keras.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.layers import Flatten, Dense, Input, Conv2D, AveragePooling2D, BatchNormalization, Dropout
from keras.layers import TimeDistributed, DepthwiseConv2D, Activation

from keras import backend as K
from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization
from keras.applications import mobilenet
WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_1_0_224_tf_no_top.h5'
alpha = 1
depth_multiplier = 1
dropout = 1e-3
include_top = True
  # todo : 看需要删除哪些对象

def get_weight_path():
    if K.image_dim_ordering() == 'th':
        print('pretrained weights not available for Mobilenet with theano backend')
        return
    else:
        return WEIGHT_PATH

def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length // 16

    return get_output_length(width), get_output_length(height)


# def relu6(x):
#     return K.relu(x, max_value=6)

from keras.applications import mobilenet
def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), trainable=True):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides, name='conv1', trainable=trainable)(inputs)
    x = FixedBatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation('relu6', name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1, use_bias=False, trainable=True):
    """Adds a depthwise convolution block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),  padding='same',  depth_multiplier=depth_multiplier, strides=strides, use_bias=use_bias,
                        name='conv_dw_%d' % block_id, trainable=trainable)(inputs)
    x = FixedBatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation('relu6', name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=use_bias, strides=(1, 1),
               name='conv_pw_%d' % block_id, trainable=trainable)(x)
    x = FixedBatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation('relu6', name='conv_pw_%d_relu' % block_id)(x)


def _depthwise_conv_block_td(inputs, pointwise_conv_filters, alpha, input_shape,
                          depth_multiplier=1, strides=(1, 1), block_id=1, trainable=True):
    """Adds a depthwise convolution block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = TimeDistributed(DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_multiplier, strides=strides,
                                        use_bias=False, trainable=trainable),
                        input_shape=input_shape, name='conv_dw_%d' % block_id)(inputs)
    x = TimeDistributed(FixedBatchNormalization(axis=channel_axis), name='conv_dw_%d_bn' % block_id)(x)

    x = Activation('relu6', name='conv_dw_%d_relu' % block_id)(x)

    x = TimeDistributed(Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1),
                               trainable=trainable), name='conv_pw_%d' % block_id)(x)
    x = TimeDistributed(FixedBatchNormalization(axis=channel_axis), name='conv_pw_%d_bn' % block_id)(x)
    return Activation('relu6', name='conv_pw_%d_relu' % block_id)(x)


def _depthwise_conv_block_td_2(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1, trainable=True):
    """增加第二个的目的是因为 TimeDistributed模块除了第一个需要制定input shape, 后面的不需要
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = TimeDistributed(DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_multiplier, strides=strides,
                                        use_bias=False, trainable=trainable), name='conv_dw_%d' % block_id)(inputs)
    x = TimeDistributed(FixedBatchNormalization(axis=channel_axis), name='conv_dw_%d_bn' % block_id)(x)

    x = Activation('relu6', name='conv_dw_%d_relu' % block_id)(x)

    x = TimeDistributed(Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1),
                               trainable=trainable), name='conv_pw_%d' % block_id)(x)
    x = TimeDistributed(FixedBatchNormalization(axis=channel_axis), name='conv_pw_%d_bn' % block_id)(x)
    return Activation('relu6', name='conv_pw_%d_relu' % block_id)(x)


def nn_base(input_tensor=None, trainable=True):

    if K.backend() != 'tensorflow':
        raise RuntimeError('Only TensorFlow backend is currently supported, '
                           'as other backends do not support '
                           'depthwise convolution.')
    input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = _conv_block(img_input, 32, alpha, strides=(2, 2), trainable=trainable)
    # Block 2
    x = _depthwise_conv_block(x, 64, alpha, block_id=1, trainable=trainable)

    x = _depthwise_conv_block(x, 128, alpha,
                              strides=(2, 2), block_id=2, trainable=trainable)
    # Block 3
    x = _depthwise_conv_block(x, 128, alpha,  block_id=3, trainable=trainable)

    x = _depthwise_conv_block(x, 256, alpha,
                              strides=(2, 2), block_id=4, trainable=trainable)
    # Block 4
    x = _depthwise_conv_block(x, 256, alpha,  block_id=5, trainable=trainable)

    x = _depthwise_conv_block(x, 512, alpha,
                              strides=(2, 2), block_id=6, trainable=trainable)
    # Blcok 5
    x = _depthwise_conv_block(x, 512, alpha,  block_id=7, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha,  block_id=8, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha,  block_id=9, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha,  block_id=10, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, block_id=11, trainable=trainable)

    return x  # reducing feature map 16 times

def classifier_layers(x, input_shape, trainable=False):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    # (hence a smaller stride in the region that follows the ROI pool)
    if K.backend() == 'tensorflow':
        x = _depthwise_conv_block_td(x, 1024, alpha, input_shape,
                                  strides=(2, 2), block_id=12, trainable=trainable)

    elif K.backend() == 'theano':
        x = _depthwise_conv_block_td(x, 1024, alpha, input_shape,
                                  strides=(1, 1), block_id=12, trainable=trainable)  # theano backend not be tested yet
    x = _depthwise_conv_block_td_2(x, 1024, alpha,  block_id=13, trainable=trainable)

    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout')(x)

    return x


def rpn(base_layers, num_anchors):  # 9 anchors are used here
    x = Conv2D(256, (3, 3), padding='same', activation='relu6', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)
    # x = _depthwise_conv_block(base_layers, 512, alpha, use_bias=True, block_id=1000)  # todo: 换成多少？ 256吗

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=True):  # background and interesting objects
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    if K.backend() == 'tensorflow':
        pooling_regions = 14   # todo: 使用更小的pooling_regions ? 这样会更快吗？
        input_shape = (num_rois, 14, 14, int(512 * alpha))
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois, 512, 7, 7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=trainable)
    out = TimeDistributed(Flatten())(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)

    # key: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]




