"""
This file contains necessary definitions for Siamese Architecture implementation.
"""

import random
import numpy as np
import time
import tensorflow as tf
import math
import pdb
import sys
import h5py
import scipy.io as sio
from sklearn import *
import matplotlib.pyplot as plt
from PlotROC import Plot_ROC_Fn
from PlotHIST import Plot_HIST_Fn
from PlotPR import Plot_PR_Fn
import re




def _weights_init(name):
    init = tf.constant(params[name][0])
    return tf.get_variable('weights', initializer=init)


def _biases_init(name):
    init = tf.constant(params[name][1])
    return tf.get_variable('biases', initializer=init)


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    # tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def standardalize_Fn(X, mean):
    """
    Standardalizing data
    :param X: Input batch
    :param mean: The specific mean vector.
    :return: The data in specific range.
    """
    X[:, :, :, 0] = (X[:, :, :, 0] - mean[0]) / 255.0
    X[:, :, :, 1] = (X[:, :, :, 1] - mean[1]) / 255.0
    X[:, :, :, 2] = (X[:, :, :, 2] - mean[2]) / 255.0

    return X


def loss(y, distance, batch_size):
    """With this definition the loss will be calculated.
        # loss is the contrastive loss plus the loss caused by
        # weight_dacay parameter(if activated). The kernel of conv_relu is
        # the place for activation of weight decay. The scale of weight_decay
        # loss might not be compatible with the contrastive loss.


        Args:
          y: The labels.
          distance: The distance vector between the output features..
          batch_size: the batch size is necessary because the loss calculation would be over each batch.

        Returns:
          The total loss.
        """

    margin = 1000
    term_1 = y * tf.square(distance)
    # tmp= tf.mul(y,tf.square(d))
    term_2 = (1 - y) * tf.square(tf.maximum((margin - distance), 0))
    Contrastive_Loss = tf.reduce_sum(term_1 + term_2) / batch_size / 2
    tf.add_to_collection('losses', Contrastive_Loss)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# Accuracy computation
def compute_accuracy(prediction, labels):
    """
    Navie way of computation
    """
    return labels[prediction.ravel() < 0.5].mean()


# Extracting the data and label for each batch.
def get_batch(start_idx, end_ind, inputs, labels):
    """Get each batch of data which is necessary for the training and testing phase..

        Args:
          start_idx: The start index of the batch in the whole sample size.
          end_idx: The ending index of the batch in the whole sample size.
          inputs: Train/test data.
          labels: Train/test labels.

        Returns:
          pair_left_part: Batch of left images in pairs.
          pair_right_part: Batch of right images in pairs.
          y: The associated labels to the images pairs.

        """
    num_orientation = inputs.shape[3] / 2
    pair_left_part = inputs[start_idx:end_ind, :, :, 0:num_orientation]
    pair_right_part = inputs[start_idx:end_ind, :, :, num_orientation:]
    y = np.reshape(labels[start_idx:end_ind], (len(range(start_idx, end_ind)), 1))
    return pair_left_part, pair_right_part, y


def max_pool(x, pool_size, stride, name='max_pooling'):
    """This is the max pooling layer..

    ## Note1: The padding is of type 'VALID'. For more information please
    refer to TensorFlow tutorials.
    ## Note2: Variable scope is useful for sharing the variables.

    Args:
      x: The input of the layer which most probably in the output of previous Convolution layer.
      pool_size: The windows size for max pooling operation.
      stride: stride of the max pooling layer

    Returns:
      The resulting feature cube.
    """
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, stride, stride, 1], padding='VALID')


#
def convolution_layer(input, kernel_size, num_outputs, activation, dropout_param, name):
    """
    This layer is mainly "tf.contrib.layers.convolution2d" layer except for the dropout parameter.
    :param input: The input to the convolution layer.
    :param kernel_size: which is a list as [kernel_height, kernel_width].
    :param num_outputs: The number of output feature maps(filters).
    :param activation: The nonlinear activation function.
    :param dropout_param: The dropout parameter which determines the probability that each neuron is kept.
    :param name: The name which might be useful for reusing the variables.

    :return: The output of the convolutional layer which is of size (?,height,width,num_outputs).
    """

    with tf.variable_scope(name):

        conv = tf.contrib.layers.convolution2d(input,
                                               num_outputs,
                                               kernel_size=kernel_size,
                                               stride=[1, 1],
                                               padding='SAME',
                                               activation_fn=activation,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               normalizer_fn=tf.contrib.layers.batch_norm,
                                               normalizer_params=None,
                                               trainable=True
                                               )
        conv = tf.nn.dropout(conv, dropout_param)
        _activation_summary(conv)
        return conv


def fc_layer(input, num_outputs, activation_fn, normalizer_fn, dropout_param, name):
    """
    This layer is mainly "tf.contrib.layers.fully_connected" layer except for the dropout parameter.
    :param input: Input tensor.
    :param num_outputs: Number of neurons in the output of the layer.
    :param activation_fn: The used nonlinear function.
    :param normalizer_fn: Like using batch normalization.
    :param dropout_param: Dropout parameter which determines the probability of keeping each neuron.
    :param name: Name for reusing the variables if necessary.

    :return: Output of the layer of size (?,num_outputs)
    """
    with tf.variable_scope(name):

        fc = tf.contrib.layers.fully_connected(input,
                                               num_outputs,
                                               activation_fn,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               normalizer_fn=normalizer_fn,
                                               trainable=True
                                               )
        fc = tf.nn.dropout(fc, dropout_param)
        _activation_summary(fc)
        return fc


def neural_network(X, dropout_param):
    """This function create each branch os Siamese Architecture.
       Basically there are not two branches. They are the same!!

        Args:
          X: The input image(batch).

        Returns:
          The whole NN model.

        """
    # Get the output features generated by CNN.
    CNN_output = CNN_Structure(X, dropout_param)

    # Flatten
    CNN_output_shape = CNN_output.get_shape()
    shape = CNN_output_shape.as_list()  # a list: [None, 9, 2]
    dim = np.prod(shape[1:])  # dim = prod(9,2) = 18
    CNN_output = tf.reshape(CNN_output, [-1, dim])

    # Fully_Connected layer - 1
    num_outputs = 4096
    activation_fn = tf.nn.relu
    normalizer_fn = tf.contrib.layers.batch_norm
    dropout_param = 0.5
    y_f1 = fc_layer(CNN_output, num_outputs, activation_fn, normalizer_fn, dropout_param, name='fc6')

    # Fully_Connected layer - 2
    num_outputs = 4096
    activation_fn = None
    normalizer_fn = None
    dropout_param = 0.5
    y_f2 = fc_layer(y_f1, num_outputs, activation_fn, normalizer_fn, dropout_param, name='fc7')

    # # Fully_Connected layer - 2
    # num_outputs = 1000
    # activation_fn = None
    # normalizer_fn = None
    # dropout_param = 1
    # y_f3 = fc_layer(y_f2, num_outputs, activation_fn, normalizer_fn, dropout_param, name='fc8')

    return y_f2


def CNN_Structure(x, dropout_param):
    """This is the whole structure of the CNN.

       Nore: Although the dropout left untouched, it can be define for the FC layers output.

         Args:
           X: The input image(batch).

         Returns:
           The output feature vector.

         """
    # No dropout for CNN.
    dropout_param = 1
    ################## SECTION - 1 ##############################

    # Conv_11 layer
    NumFeatureMaps = 64
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    activation = tf.nn.relu
    relu11 = convolution_layer(x, kernel_size, NumFeatureMaps, activation, dropout_param, name = 'conv1_1')

    # Conv_12 layer
    NumFeatureMaps = 64
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    activation = tf.nn.relu
    relu12 = convolution_layer(relu11, kernel_size, NumFeatureMaps, activation, dropout_param, name='conv1_2')

    # Pool_1 layer
    pool_1 = max_pool(relu12, 2, 2, name='pool1')

    ###########################################################
    ##################### SECTION - 2 #########################
    ###########################################################

    # Conv_21 layer
    # Number of feature maps
    NumFeatureMaps = 128
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    activation = tf.nn.relu
    relu21 = convolution_layer(pool_1, kernel_size, NumFeatureMaps, activation, dropout_param, name='conv2_1')

    # Conv_22 layer
    NumFeatureMaps = 128
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    activation = tf.nn.relu
    relu22 = convolution_layer(relu21, kernel_size, NumFeatureMaps, activation, dropout_param, name='conv2_2')

    # Pool_2 layer
    pool_2 = max_pool(relu22, 2, 2, name='pool2')

    ###########################################################
    ##################### SECTION - 3 #########################
    ###########################################################

    # Conv_31 layer
    NumFeatureMaps = 256
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    activation = tf.nn.relu
    relu31 = convolution_layer(pool_2, kernel_size, NumFeatureMaps, activation, dropout_param, name='conv3_1')

    # Conv_32 layer
    NumFeatureMaps = 256
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    activation = tf.nn.relu
    relu32 = convolution_layer(relu31, kernel_size, NumFeatureMaps, activation, dropout_param, name='conv3_2')

    # Conv_33 layer
    NumFeatureMaps = 256
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    activation = tf.nn.relu
    relu33 = convolution_layer(relu32, kernel_size, NumFeatureMaps, activation, dropout_param, name='conv3_3')

    # # Conv_34 layer
    # NumFeatureMaps = 256
    # kernel_height = 3
    # kernel_width = 3
    # kernel_size = [kernel_height, kernel_width]
    # activation = tf.nn.relu
    # relu34 = convolution_layer(relu33, kernel_size, NumFeatureMaps, activation, dropout_param, name='conv3_4')


    # Pool_3 layer
    pool_3 = max_pool(relu33, 2, 2, name='pool3')

    ###########################################################
    ##################### SECTION - 4 #########################
    ###########################################################

    # Conv_41 layer
    NumFeatureMaps = 512
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    activation = tf.nn.relu
    relu41 = convolution_layer(pool_3, kernel_size, NumFeatureMaps, activation, dropout_param, name='conv4_1')

    # Conv_42 layer
    NumFeatureMaps = 512
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    activation = tf.nn.relu
    relu42 = convolution_layer(relu41, kernel_size, NumFeatureMaps, activation, dropout_param, name='conv4_2')

    # Conv_43 layer
    NumFeatureMaps = 512
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    activation = tf.nn.relu
    relu43 = convolution_layer(relu42, kernel_size, NumFeatureMaps, activation, dropout_param, name='conv4_3')

    # # Conv_44 layer
    # NumFeatureMaps = 512
    # kernel_height = 3
    # kernel_width = 3
    # kernel_size = [kernel_height, kernel_width]
    # activation = tf.nn.relu
    # relu44 = convolution_layer(relu43, kernel_size, NumFeatureMaps, activation, dropout_param, name='conv44')

    # Pool_4 layer
    pool_4 = max_pool(relu43, 2, 2, name='pool4')

    ###########################################################
    ##################### SECTION - 5 #########################
    ###########################################################

    # Conv_41 layer
    NumFeatureMaps = 512
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    activation = tf.nn.relu
    relu51 = convolution_layer(pool_4, kernel_size, NumFeatureMaps, activation, dropout_param, name='conv5_1')

    # Conv_42 layer
    NumFeatureMaps = 512
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    activation = tf.nn.relu
    relu52 = convolution_layer(relu51, kernel_size, NumFeatureMaps, activation, dropout_param, name='conv5_2')

    # Conv_43 layer
    NumFeatureMaps = 512
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    activation = tf.nn.relu
    relu53 = convolution_layer(relu52, kernel_size, NumFeatureMaps, activation, dropout_param, name='conv5_3')

    # # Conv_44 layer
    # NumFeatureMaps = 512
    # kernel_height = 3
    # kernel_width = 3
    # kernel_size = [kernel_height, kernel_width]
    # activation = tf.nn.relu
    # relu54 = convolution_layer(relu53, kernel_size, NumFeatureMaps, activation, dropout_param, name='conv54')

    # Pool_4 layer
    pool_5 = max_pool(relu53, 2, 2, name='pool5')

    return pool_5
