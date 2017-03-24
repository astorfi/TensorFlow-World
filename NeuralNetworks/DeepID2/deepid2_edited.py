# CNN Siamese Implementation for face recognition.

import random
import numpy as np
import time
import tensorflow as tf
import math
import tables
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
import deepid2_Architecture_Edited

"""
Parameters and input data
"""

# PART 4
# TRAIN_FILE = h5py.File('TRAIN.hdf5', 'r')
# TEST_FILE = h5py.File('TEST.hdf5', 'r')
TRAIN_FILE = h5py.File('train_pairs.hdf5', 'r')
TEST_FILE = h5py.File('test_pairs.hdf5', 'r')

X_train_aux = TRAIN_FILE['aux']
X_train_main = TRAIN_FILE['main']
y_train = TRAIN_FILE['labelCont']
y_frontal_train = TRAIN_FILE['label']
y_profile_train = TRAIN_FILE['label_a']



X_test_aux = TEST_FILE['aux']
X_test_main = TEST_FILE['main']
y_test = TEST_FILE['labelCont']
y_frontal_test = TEST_FILE['label']
y_profile_test = TEST_FILE['label_a']

mean = [50, 52, 61]





N_classes = np.max(y_profile_test) + 1

y_frontal_train = deepid2_Architecture_Edited.Vector_to_Onehot_Fn(y_frontal_train, N_classes)
y_profile_train = deepid2_Architecture_Edited.Vector_to_Onehot_Fn(y_profile_train, N_classes)

y_frontal_test = deepid2_Architecture_Edited.Vector_to_Onehot_Fn(y_frontal_test, N_classes)
y_profile_test = deepid2_Architecture_Edited.Vector_to_Onehot_Fn(y_profile_test, N_classes)

print type(X_train_aux)

print '\n---------------- Train Data -------------------'
# Train Dimensions
num_samples = X_train_aux.shape[0]
num_channels = X_train_aux.shape[1]
width = X_train_aux.shape[2]
height = X_train_aux.shape[3]

print num_samples, num_channels, height, width
print '\n-----------------------------------------------'

print '\n---------------- Test Data -------------------'
# Test Dimensions
num_test_samples = X_test_aux.shape[0]
num_channels2 = X_test_aux.shape[1]
width2 = X_test_aux.shape[2]
height2 = X_test_aux.shape[3]

print num_test_samples, num_channels2, height2, width2
print '-----------------------------------------------'

# sys.exit(1)



class fullprint:
    '''context manager for printing full numpy arrays'''

    def __enter__(self):
        self.opt = np.get_printoptions()
        np.set_printoptions(threshold=np.nan)

    def __exit__(self, type, value, traceback):
        np.set_printoptions(**self.opt)




# Defining distance initial vectors for evaluation of the final output of the network.
distance_train_output = np.zeros([y_train.shape[0],1])
print "distance_train_output",distance_train_output.shape
distance_test_output = np.zeros([y_test.shape[0],1])

print distance_train_output.shape[0]
# sys.exit(1)


# Defining the graph of network.
graph = tf.Graph()
with graph.as_default():
    # Some variable defining.

    # batch = tf.Variable(0)
    batch_size = 16
    # global_step = tf.Variable(0, trainable=False)
    global_step = 0

    #TODO: Defining Learning rate policy.
    starter_learning_rate = 0.001
    num_batches_per_epoch = int(num_samples / batch_size)
    NUM_EPOCHS_PER_DECAY = 1
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    LEARNING_RATE_DECAY_FACTOR = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps,
                                               LEARNING_RATE_DECAY_FACTOR, staircase=True)
    # Adding the larning rate to summary.
    tf.scalar_summary('learning_rate', learning_rate)

    #TODO: Defining place holders
    # Defining the place holders.
    images_Frontal = tf.placeholder(tf.float32, shape=([None, height, width, num_channels]), name='F')
    images_Profile = tf.placeholder(tf.float32, shape=([None, height, width, num_channels]), name='P')
    # images_Frontal = tf.placeholder(tf.float32, shape=([None, num_channels,height, width]), name='F')
    # images_Profile = tf.placeholder(tf.float32, shape=([None, num_channels,height, width]), name='P')



    # Labels specified to verification.
    labels = tf.placeholder(tf.float32, shape=([None, 1]), name='label_contrastive')

    # Specific to classification
    label_F = tf.placeholder(tf.float32, shape=([None, N_classes]), name='label_frontal')
    label_P = tf.placeholder(tf.float32, shape=([None, N_classes]), name='label_profile')



    # # Dropout parameter as a place holder.
    # dropout_param = tf.placeholder(tf.float32,name='dropout')

    # TODO: Calling the network.
    # Extracting the outputs of different branches
    with tf.variable_scope("DeepID2") as scope:
        CNN_output_Frontal, MLP_output_Frontal = deepid2_Architecture_Edited.MLP_Frontal(images_Frontal)
        print MLP_output_Frontal.get_shape()
        CNN_output_Profile, MLP_output_Profile = deepid2_Architecture_Edited.MLP_Profile(images_Profile)
        print MLP_output_Profile.get_shape()





    #TODO: Calculation of total loss
    # Defining the distance metric for the outputs of the network.
    distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(CNN_output_Frontal, CNN_output_Profile), 2), 1, keep_dims=True))

    # CALCULATION OF CONTRASTIVE LOSS
    contrastive_loss = deepid2_Architecture_Edited.Contrastive_Loss(labels, distance, batch_size)
    classification_loss_Frontal = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(MLP_output_Frontal, label_F))
    classification_loss_Profile = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(MLP_output_Profile, label_P))


    # Adding different losses
    loss = deepid2_Architecture_Edited.get_total_loss(contrastive_loss, classification_loss_Frontal, classification_loss_Profile)

    # Just Contrastive Loss
    # loss = deepid2_Architecture_Edited.get_contrastive_loss(contrastive_loss)

    #TODO: choosing different options for optimizer
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    # optimizer = tf.train.RMSPropOptimizer(0.0001,momentum=0.9,epsilon=1e-6).minimize(loss)
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

# sys.exit(1)


    loss_train_values = []

# TODO: Launching the graph!
with tf.Session(graph=graph) as sess:
    num_epoch =5
    sess.run(init)

    # # Uncomment if you want to restore the model
    # saver.restore(sess, "checkpoints/cmu_pie_pair.ckpt")
    # print("Model restored.")
    # Training cycle
    for epoch in range(num_epoch):
        avg_loss = 0.
        avg_acc = 0.
        total_batch = int(num_samples / batch_size)
        start_time = time.time()

        range_val = total_batch #total_batch

        # Loop over all batches
        for i in range(range_val):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # Fit training using batch data
            input1, input2, y_c, y_f, y_p = deepid2_Architecture_Edited.get_batch_edited(start_idx, end_idx, X_train_main, X_train_aux, y_train, y_frontal_train, y_profile_train)


            input1 = deepid2_Architecture_Edited.standardalize_Fn(input1, mean)
            input2 = deepid2_Architecture_Edited.standardalize_Fn(input2, mean)



            #input1_te, input2_te, y_te = X_test[start_idx:end_idx, :, :, 0:num_channels], X_test[start_idx:end_idx, :, :, num_channels:], y_test
            # input1_te, input2_te, y_te = X_test_main[start_idx:end_idx, :, :,:], X_test_aux[start_idx:end_idx, :, :, :], y_test
            #
            # print input1.shape[0], input1.shape[1], input1.shape[2], input1.shape[3]
            # sys.exit(1)
            # TODO: Running the session and evaluation of three elements.
            # _, loss_value, predict = sess.run([optimizer, loss, distance],
            #                                   feed_dict={images_Frontal: input1, images_Profile: input2, labels: y,
            #                                              dropout_param: 0.9})
            _, loss_value, predict = sess.run([optimizer, loss, distance],
                                              feed_dict={images_Frontal: input1, images_Profile: input2, labels: y_c, label_F: y_f, label_P: y_p})

            # loss, acc = sess.run([classification_loss_Frontal, accuracy], feed_dict={images_Frontal: input1_te, images_Profile: input2_te, labels: y_cont_te, label_F: y_f_te, label_P: y_p_te})


            # This will be repeated for each epoch but for the moment it is the only way
            # because the training data cannot be fed at once.



            # # This will be repeated for each epoch but for the moment it is the only way
            # # because the test data cannot be fed at once.
            # # The upper limit for test data is less that train data
            # if end_idx < X_test_aux.shape[0]:
            #     distance_test_evaluation = distance.eval(
            #         feed_dict={images_Frontal: X_test_main[start_idx:end_idx, :, :, 0:num_channels],
            #                    images_Profile: X_test_aux[start_idx:end_idx, :, :, 0:num_channels], #images_Profile: X_test[start_idx:end_idx, :, :, num_channels:],
            #                    dropout_param: 1.0})
            #     distance_test_output[start_idx:end_idx, :] = distance_test_evaluation
            # else:
            #     if start_idx < X_test_aux.shape[0]:
            #         distance_test_evaluation = distance.eval(
            #             feed_dict={images_Frontal: X_test_main[start_idx:X_test_main.shape[0], :, :, 0:num_channels],
            #                        images_Profile: X_test_aux[start_idx:X_test_aux.shape[0], :, :, 0:num_channels], #images_Profile: X_test[start_idx:X_test.shape[0], :, :, num_channels:],
            #
            #                        dropout_param: 1.0})
            #         distance_test_output[start_idx:end_idx, :] = distance_test_evaluation



            # # Training output features per batch
            # feature1_tr = MLP_output_Frontal.eval(feed_dict={images_Frontal: input1, dropout_param: 1.0})
            # feature2_tr = MLP_output_Profile.eval(feed_dict={images_Profile: input2, dropout_param: 1.0})


            # # Test output features per whole test set
            # feature1_te = MLP_output_Frontal.eval(feed_dict={images_Frontal: input1_te, dropout_param: 1.0})
            # feature2_te = MLP_output_Profile.eval(feed_dict={images_Profile: input2_te, dropout_param: 1.0})

            # --------------------------   added ------------------------
            # loss_train_values.append(loss_value)
            # ----------------------------------------------------------

            avg_loss += loss_value
            print("epoch %d, batch %d of %d loss= %f" % (epoch, i + 1,total_batch, loss_value))

        ############ TEST AFTER EACH EPOCH ####################
        batch_size_test = 128
        accuracy_test_frontal = 0
        accuracy_test_profile = 0
        num_samples_test = y_profile_test.shape[0]
        total_batch_test = int(num_samples_test / batch_size_test)
        for j in range(total_batch_test):
            start_idx_test = j * batch_size_test
            end_idx_test = min((j + 1) * batch_size_test,y_frontal_test.shape[0])
            input1_te, input2_te, y_c_te, y_f_te, y_p_te = deepid2_Architecture_Edited.get_batch_edited(start_idx_test, end_idx_test,
                                                                                                        X_test_main,
                                                                                                        X_test_aux,
                                                                                                        y_test,
                                                                                                        y_frontal_test,
                                                                                                        y_profile_test)
            input1_te = deepid2_Architecture_Edited.standardalize_Fn(input1_te, mean)
            input2_te = deepid2_Architecture_Edited.standardalize_Fn(input2_te, mean)


            softmax_output_frontal = MLP_output_Frontal.eval(
                feed_dict={images_Frontal: input1_te,
                           images_Profile: input2_te})

            softmax_output_profile = MLP_output_Profile.eval(
                feed_dict={images_Frontal: input1_te,
                           images_Profile: input2_te})

            # Frontal Classification
            correct_pred = np.equal(np.argmax(softmax_output_frontal, axis=1), np.argmax(y_f_te, axis=1))
            accuracy_frontal = np.sum(correct_pred) / float(batch_size_test)
            print ('test for batch %d of %d,  accuracy frontal=%f' % (j, total_batch_test, accuracy_frontal))
            accuracy_test_frontal += accuracy_frontal

            # Profile Classification
            correct_pred = np.equal(np.argmax(softmax_output_profile, axis=1), np.argmax(y_p_te, axis=1))
            accuracy_profile = np.sum(correct_pred) / float(batch_size_test)
            print ('test for batch %d of %d,  accuracy profile=%f' % (j, total_batch_test, accuracy_profile))
            accuracy_test_profile += accuracy_profile



        print "total accuracy frontal=", accuracy_test_frontal
        print "total accuracy profile=", accuracy_test_profile


        duration = time.time() - start_time
        print(
            'epoch %d  time: %f average_loss %0.5f' % (
                epoch + 1, duration, avg_loss / (total_batch)))


        # # --------------------------   added ------------------------
        # count = 1
        # plt_x_axis = []
        # while count <= range_val:
        #     #print(count)
        #     count = count + 1;
        #     plt_x_axis.append(count)
        # # print plt_x_axis
        #
        # #with fullprint():
        #     #print loss_train_values
        # ----------------------------------------------------------

        save_path = saver.save(sess, 'weights/model'+'_'+str(epoch)+'.ckpt')
        print("Model saved in file: %s" % save_path)


    # TODO: Test model on test samples
    label_test = np.reshape(y_test, (y_test.shape[0], 1))


# --------------------------   added ------------------------
#print plt_x_axis.__len__()
#print loss_train_values.__len__()
# sys.exit(1)
# fig = plt.figure()
# fig.suptitle('Training Loss', fontsize=14, fontweight='bold')
# plt.plot(plt_x_axis, loss_train_values, linewidth=2.0)
# # plt.title('Training Loss') # subplot 211 title
# plt.xlabel('iterations')
# plt.ylabel('loss')
# ----------------------------------------------------------


"""
TRAIN
"""
# # Distance between original features before feeding to the network.
# distance_original_train = np.sqrt(
#     # np.sum(np.power(X_train[:, :, :, 0:num_channels] - X_train[:, :, :, num_channels:], 2), axis=(1, 2, 3)))
#     np.sum(np.power(X_train_main[:, :, :, 0:num_channels] - X_train_aux[:, :, :, 0:num_channels], 2), axis=(1, 2, 3)))




# # Change the format from (N,1) to (N,).
# print distance_train_output.shape
# distance_train_output = distance_train_output[~(distance_train_output==0)]
# print "distance_train_output",distance_train_output.shape,distance_train_output.shape[0]
distance_train_output = distance_train_output[:,0]
# # Plot ROC for training
# Plot_ROC_Fn(y_train, distance_original_train, 'Train', 'Input')
Plot_ROC_Fn(y_train[:distance_train_output.shape[0]], distance_train_output, 'Train', 'Output')#[:160]

# # Plot Precision-Recall for training
# Plot_PR_Fn(y_train, distance_original_train , 'Train', 'Input')
Plot_PR_Fn(y_train, distance_train_output, 'Train', 'Output') #[:160]

# # Plot HIST for training
# Plot_HIST_Fn(y_train, distance_original_train, 'Train', 'Input')
Plot_HIST_Fn(y_train, distance_train_output, 'Train', 'Output') #[:160]
#
# """
# TEST
# """
# # Distance between original features before feeding to the network.
# distance_original_test = np.sqrt(
#     # np.sum(np.power(X_test[:, :, :, 0:num_channels] - X_test[:, :, :, num_channels:], 2), axis=(1, 2, 3)))
#     np.sum(np.power(X_test_main[:, :, :, 0:num_channels] - X_test_aux[:, :, :, 0:num_channels], 2), axis=(1, 2, 3)))
#
# # Change the format from (N,1) to (N,).
# distance_test_output = distance_test_output[:, 0]
#
# # Plot ROC for test
# Plot_ROC_Fn(label_test, distance_original_test, 'Test', 'Input')
# Plot_ROC_Fn(label_test, distance_test_output, 'Test', 'Output')
#
# # Plot Precision-Recall for test
# Plot_PR_Fn(label_test, distance_original_test , 'Test', 'Input')
# Plot_PR_Fn(label_test, distance_test_output, 'Test', 'Output')
#
# # Plot HIST for test
# Plot_HIST_Fn(label_test, distance_original_test, 'Test', 'Input')
# Plot_HIST_Fn(label_test, distance_test_output, 'Test', 'Output')
