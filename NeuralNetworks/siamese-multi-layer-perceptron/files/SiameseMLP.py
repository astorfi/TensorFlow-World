# CNN Siamese Implementation for face recognition.

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
import siamese_structure

"""
Parameters and input data
"""
TRAIN_FILE = h5py.File('data/TRAIN.hdf5', 'r')
TEST_FILE = h5py.File('data/TEST.hdf5', 'r')

# Extracting the data and labels from HDF5.
# Pairs and labels have been saved separately in HDF5 files.
# The number of features and samples are extracted from data.
X_train = TRAIN_FILE['pairs']
y_train = TRAIN_FILE['labels']
X_test = TEST_FILE['pairs']
y_test = TEST_FILE['labels']

# Number of samples.
num_samples = X_train.shape[0]

# Number of features
num_features = X_train.shape[1]

# The third dimenstions must be 2 because we have pairs of features
num_channels = X_train.shape[2]/2


# Defining distance initial vectors for evaluation of the final output of the network.
distance_train_output = np.zeros(y_train.shape)
distance_test_output = np.zeros(y_test.shape)

# Defining the graph of network.
graph = tf.Graph()
with graph.as_default():
    # Some variable defining.
    batch = tf.Variable(0)
    batch_size = 256
    global_step = tf.Variable(0, trainable=False)

    # Learning rate policy.
    starter_learning_rate = 0.001
    num_batches_per_epoch = int(num_samples / batch_size)
    NUM_EPOCHS_PER_DECAY = 1
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    LEARNING_RATE_DECAY_FACTOR = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps,
                                               LEARNING_RATE_DECAY_FACTOR, staircase=True)

    # Adding the larning rate to summary.
    tf.scalar_summary('learning_rate', learning_rate)

    # Defining the place holders.
    feature_L = tf.placeholder(tf.float32, shape=([None, num_features]), name='L')
    feature_R = tf.placeholder(tf.float32, shape=([None, num_features]), name='R')
    labels = tf.placeholder(tf.float32, shape=([None, 1]), name='gt')
    dropout_param = tf.placeholder("float")

    # Sharing the same parameter in order to creating the Siamese Architecture.
    with tf.variable_scope("siamese") as scope:
        output_L = siamese_structure.neural_network(feature_L, dropout_param)
        scope.reuse_variables()
        output_R = siamese_structure.neural_network(feature_R, dropout_param)

    # Defining the distance metric for the outputs of the network.
    distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(output_L, output_R), 2), 1, keep_dims=True))

    # CALCULATION OF LOSS
    loss = siamese_structure.loss(labels, distance, batch_size)

    #TODO: choosing different options for optimizer
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    # optimizer = tf.train.RMSPropOptimizer(0.0001,momentum=0.9,epsilon=1e-6).minimize(loss)

# TODO: Launching the graph!
with tf.Session(graph=graph) as sess:
    num_epoch = 20
    tf.initialize_all_variables().run()

    # Training cycle
    for epoch in range(num_epoch):
        avg_loss = 0.
        avg_acc = 0.
        total_batch = int(num_samples / batch_size)
        start_time = time.time()

        # Loop over all batches
        for i in range(total_batch):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # Fit training using batch data
            input1, input2, y = siamese_structure.get_batch(start_idx, end_idx, X_train, y_train)
            input1_te, input2_te, y_te = X_test[:, :, 0], X_test[:, :, 1], y_test


            # TODO: Running the session and evaluation of three elements.
            _, loss_value, predict = sess.run([optimizer, loss, distance],
                                              feed_dict={feature_L: input1, feature_R: input2, labels: y,
                                                         dropout_param: 0.9})

            # This will be repeated for each epoch but for the moment it is the only way
            # because the training data cannot be fed at once.
            distance_train_evaluation = distance.eval(
                feed_dict={feature_L: X_train[start_idx:end_idx, :, 0],
                           feature_R: X_train[start_idx:end_idx, :, 1],
                           dropout_param: 1.0})
            distance_train_output[start_idx:end_idx, :] = distance_train_evaluation

            # This will be repeated for each epoch but for the moment it is the only way
            # because the test data cannot be fed at once.
            # The upper limit for test data is less that train data
            if end_idx < X_test.shape[0]:
                distance_test_evaluation = distance.eval(
                    feed_dict={feature_L: X_test[start_idx:end_idx, :, 0],
                               feature_R: X_test[start_idx:end_idx, :, 1],
                               dropout_param: 1.0})
                distance_test_output[start_idx:end_idx, :] = distance_test_evaluation
            else:
                if start_idx < X_test.shape[0]:
                    distance_test_evaluation = distance.eval(
                        feed_dict={feature_L: X_test[start_idx:X_test.shape[0], :, 0],
                                   feature_R: X_test[start_idx:X_test.shape[0], :, 1],
                                   dropout_param: 1.0})
                    distance_test_output[start_idx:end_idx, :] = distance_test_evaluation



            # Training output features per batch
            feature1_tr = output_L.eval(feed_dict={feature_L: input1, dropout_param: 1.0})
            feature2_tr = output_R.eval(feed_dict={feature_R: input2, dropout_param: 1.0})


            # Test output features per whole test set
            feature1_te = output_L.eval(feed_dict={feature_L: input1_te, dropout_param: 1.0})
            feature2_te = output_R.eval(feed_dict={feature_R: input2_te, dropout_param: 1.0})

            avg_loss += loss_value
            print("batch %d loss= %f" % (i + 1, loss_value))
        duration = time.time() - start_time
        print(
            'epoch %d  time: %f average_loss %0.5f' % (
                epoch + 1, duration, avg_loss / (total_batch)))

    # TODO: Test model on test samples
    label_test = np.reshape(y_test, (y_test.shape[0], 1))


"""
TRAIN
"""
# Distance between original features before feeding to the network.
distance_original_train = np.sqrt(
    np.sum(np.power(X_train[:, :, 0:num_channels] - X_train[:, :, num_channels:], 2), axis=(1, 2, 3)))

# Change the format from (N,1) to (N,).
distance_train_output = distance_train_output[:, 0]

# Plot ROC for training
Plot_ROC_Fn(y_train, distance_original_train, 'Train', 'Input')
Plot_ROC_Fn(y_train, distance_train_output, 'Train', 'Output')

# Plot Precision-Recall for training
Plot_PR_Fn(y_train, distance_original_train , 'Train', 'Input')
Plot_PR_Fn(y_train, distance_train_output, 'Train', 'Output')

# Plot HIST for training
Plot_HIST_Fn(y_train, distance_original_train, 'Train', 'Input')
Plot_HIST_Fn(y_train, distance_train_output, 'Train', 'Output')

"""
TEST
"""
# Distance between original features before feeding to the network.
distance_original_test = np.sqrt(
    np.sum(np.power(X_test[:, :, 0:num_channels] - X_test[:, :, num_channels:], 2), axis=(1, 2, 3)))

# Change the format from (N,1) to (N,).
distance_test_output = distance_test_output[:, 0]

# Plot ROC for test
Plot_ROC_Fn(label_test, distance_original_test, 'Test', 'Input')
Plot_ROC_Fn(label_test, distance_test_output, 'Test', 'Output')

# Plot Precision-Recall for test
Plot_PR_Fn(label_test, distance_original_test , 'Test', 'Input')
Plot_PR_Fn(label_test, distance_test_output, 'Test', 'Output')

# Plot HIST for test
Plot_HIST_Fn(label_test, distance_original_test, 'Test', 'Input')
Plot_HIST_Fn(label_test, distance_test_output, 'Test', 'Output')
