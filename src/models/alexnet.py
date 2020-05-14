# Copyright 2020 Yuchen Wong

import numpy as np
import tensorflow as tf

class AlexNet(object):
    def __init__(self, 
            numClass=3,
            dropOutProb=0.5,
            placeHolder,
            pretrainedWeights=None):
        self._numClass = numClass
        self._dropOutProb = dropOutProb
        self._placeHolder = placeHolder
        self._pretrainedWeight = pretrainedWeights

        buildUpModel()

    def buildUpModel(self):

        # 1st layer
        conv1 = conv(self._placeHolder, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        maxpool1 = maxPool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(maxPool1, 2, 2e-05, 0.75, name='norm1')

        # 2nd layer
        conv2 = conv(norm1, 5, 5, 256, 1, 1, name='conv2', groups=2)
        maxPool2 = maxPool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(maxPool2, 2, 2e-05, 0.75, name='norm2')

        # 3rd layer
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')

        # 4th layer
        conv4 = conv(conv3, 3, 3, 256, 1, 1, groups=2, name='conv4')

        # 5th layer
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        maxPool5 = maxPool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th layer
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fullConnect(flattened, 6*6*256, 4096, name='fc6')
        dropout6 = dropOut(fc6, 1.0 - self._dropOutProb)

        # 7th layer
        fc7 = fullConnect(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropOut(fc7, 1.0 - self._dropOutProb)

        # 8th layer
        self._score = fullConnect(dropout7, 4096, self._numClass, relu=False, name='fc8')

        return

def conv(x, kernelHeight, kernelWidth, numFilters, 
        strideY, strideX, name, padding='SAME', groups=1):
    channels = int(x.shape[-1])
    convolution = lambda i, k: tf.nn.conv2d(i, k, strides=[1, strideY, strideX, 1],
            padding=padding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[kernelHeight, kernelWidth, channels/groups, numFilters])
        bias = tf.get_variable('bias', shape=[numFilters])

        if groups == 1:
            conv = convolution(x, weights)
        else:
            inputGroups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weightGroups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            outGroups = [convolution(i, k) for i, k in zip(inputGroups, weightGroups)]

            conv = tf.concat(axis=3, values=outGroups)

        bias = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape().as_list())

        relu = tf.nn.relu(bias, name=scope.name)
        return relu

def dropOut(x, keepProb):
    return tf.nn.dropout(x, keepProb)

def fullConnect(x, numIn, numOut, name, activation='relu'):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("weights", shape=[numIn, numOut], trainable=True)
        b = tf.get_variable("bias", shape=[numOut], trainable=True)

        out = tf.compat.v1.nn.xw_plus_b(x, w, b, name=name)

        if activation == 'relu':
            return rf.nn.relu(out)
        else:
            return out

def lrn(x, depthRadius, bias, alpha, beta, name):
    return tf.nn.local_response_normalization(x, depth_radius=depthRadius,
            alpha=alpha, beta=beta, name=name)

def maxPool(x, kernelHeight, kernelWidth,
        strideY, strideX, padding='SAME', name):
    return tf.nn.max_pool(x, ksize=[1, kernelHeight, kernelWidth, 1],
            stride=[1, strideY, strideX, 1],
            padding=padding, name=name)
