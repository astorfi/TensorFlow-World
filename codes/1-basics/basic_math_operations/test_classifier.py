from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from net_structure import net
from input_function import input
from auxiliary import progress_bar
import os
import sys

######################################
######### Necessary Flags ############
######################################
tf.app.flags.DEFINE_string(
    'test_dir', os.path.dirname(os.path.abspath(__file__)) + '/test_logs',
    'Directory where event logs are written to.')

tf.app.flags.DEFINE_string(
    'checkpoint_dir',
    os.path.dirname(os.path.abspath(__file__)) + '/checkpoints',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_integer('num_classes', 10,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_integer('batch_size', np.power(2, 9),
                            'Number of model clones to deploy.')

#########################################
########## status flags #################
#########################################
tf.app.flags.DEFINE_boolean('is_training', False,
                            'Training/Testing.')

tf.app.flags.DEFINE_boolean('allow_soft_placement', True,
                            'Automatically put the variables on CPU if there is no GPU support.')

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Demonstrate which variables are on what device.')

# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS

################################################
################# handling errors!##############
################################################
if not os.path.isabs(FLAGS.checkpoint_dir):
    raise ValueError('You must assign absolute path for --checkpoint_dir')

##########################################
####### Load and Organize Data ###########
##########################################
'''
In this part the input must be prepared.

   1 - The MNIST data will be downloaded.
   2 - The images and labels for both training and testing will be extracted.
   3 - The prepared data format(?,784) is different by the appropriate image shape(?,28,28,1) which needs
        to be fed to the CNN architecture. So it needs to be reshaped.

'''

# Download and get MNIST dataset(available in tensorflow.contrib.learn.python.learn.datasets.mnist)
# It checks and download MNIST if it's not already downloaded then extract it.
# The 'reshape' is True by default to extract feature vectors but we set it to false to we get the original images.
mnist = input_data.read_data_sets("MNIST_data/", reshape=False, one_hot=False)

# The 'input.provide_data' is provided to organize any custom dataset which has specific characteristics.
data = input.provide_data(mnist)

# Dimentionality of train
dimensionality_train = data.train.images.shape

# Dimensions
num_train_samples = dimensionality_train[0]
height = dimensionality_train[1]
width = dimensionality_train[2]
num_channels = dimensionality_train[3]

#######################################
########## Defining Graph ############
#######################################

graph = tf.Graph()
with graph.as_default():
    ###################################
    ########### Parameters ############
    ###################################

    # global step
    global_step = tf.Variable(0, name="global_step", trainable=False)

    ###############################################
    ########### Defining place holders ############
    ###############################################
    image_place = tf.placeholder(tf.float32, shape=([None, height, width, num_channels]), name='image')
    label_place = tf.placeholder(tf.float32, shape=([None, FLAGS.num_classes]), name='gt')
    dropout_param = tf.placeholder(tf.float32)

    ##################################################
    ########### Model + loss + accuracy ##############
    ##################################################

    # MODEL
    arg_scope = net.net_arg_scope(weight_decay=0.0005, is_training=FLAGS.is_training)
    with tf.contrib.framework.arg_scope(arg_scope):
        logits, end_points = net.net_architecture(image_place, num_classes=FLAGS.num_classes,
                                                  dropout_keep_prob=dropout_param,
                                                  is_training=FLAGS.is_training)

    # Define loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_place))

    # Accuracy
    with tf.name_scope('accuracy'):
        # Evaluate the model
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_place, 1))

        # Accuracy calculation
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    ###############################################
    ############ Define Sammaries #################
    ###############################################

    # Image summaries(draw three random images from data in both training and testing phases)
    # The image summaries is only cerated for train summaries and it get three random images from the training set.
    arr = np.random.randint(data.test.images.shape[0], size=(3,))
    tf.summary.image('images', data.test.images[arr], max_outputs=3,
                     collections=['per_epoch_train'])

    # Histogram and scalar summaries sammaries
    # sparsity: This summary is the fraction of zero activation for the output of each layer!
    # activations: This summary is the histogram of activation for the output of each layer!
    # WARNING: tf.summary.histogram can be very time consuming so it will be calculated per epoch!
    for end_point in end_points:
        x = end_points[end_point]
        tf.summary.scalar('sparsity/' + end_point,
                          tf.nn.zero_fraction(x), collections=['test'])

    # Summaries for loss and accuracy
    tf.summary.scalar("loss", loss, collections=['test'])
    tf.summary.scalar("accuracy", accuracy, collections=['test'])
    tf.summary.scalar("global_step", global_step, collections=['test'])

    # Merge all summaries together.
    summary_test_op = tf.summary.merge_all('test')

    ########################################################
    ############ # Defining the tensors list ###############
    ########################################################

    tensors_key = ['loss', 'accuracy', 'global_step', 'image_place', 'label_place',
                   'summary_test_op']
    tensors_values = [loss, accuracy, global_step, image_place, label_place, summary_test_op]
    tensors = dict(zip(tensors_key, tensors_values))

    ############################################
    ############ Run the Session ###############
    ############################################
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(graph=graph, config=session_conf)

    with sess.as_default():

        # The saver op.
        saver = tf.train.Saver()

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        ###################################################################
        ########## Defining the summary writers for test ###########
        ###################################################################

        test_summary_dir = os.path.join(FLAGS.test_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir)
        test_summary_writer.add_graph(sess.graph)

        # The prefix for checkpoint files
        checkpoint_prefix = 'model'

        # Restoring the saved weights.
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, checkpoint_prefix))
        print("Model restored...")

        ###################################################################
        ########## Run the training and loop over the batches #############
        ###################################################################
        total_batch_test = int(data.test.images.shape[0] / FLAGS.batch_size)

        # go through the batches
        test_accuracy = 0
        for batch_num in range(total_batch_test):
            #################################################
            ########## Get the training batches #############
            #################################################

            start_idx = batch_num * FLAGS.batch_size
            end_idx = (batch_num + 1) * FLAGS.batch_size

            # Fit training using batch data
            test_batch_data, test_batch_label = data.test.images[start_idx:end_idx], data.test.labels[
                                                                                     start_idx:end_idx]

            ########################################
            ########## Run the session #############
            ########################################

            # Run session and Calculate batch loss and accuracy
            # When the tensor tensors['global_step'] is evaluated, it will be incremented by one.

            test_batch_accuracy, batch_loss, test_summaries, test_step = sess.run(
                [tensors['accuracy'], tensors['loss'], tensors['summary_test_op'],
                 tensors['global_step']],
                feed_dict={tensors['image_place']: test_batch_data,
                           tensors['label_place']: test_batch_label})
            test_accuracy += test_batch_accuracy

            ########################################
            ########## Write summaries #############
            ########################################

            # Write the summaries
            test_summary_writer.add_summary(test_summaries, global_step=test_step)

            # # Write the specific summaries for training phase.
            # train_summary_writer.add_summary(train_image_summary, global_step=training_step)

            #################################################
            ########## Plot the progressive bar #############
            #################################################

            progress = float(batch_num + 1) / total_batch_test
            progress_bar.print_progress(progress, epoch_num=1, loss=batch_loss)


        ######################################################################
        ########## Calculate the accuracy for the whole test set #############
        ######################################################################
        test_accuracy_total = test_accuracy / float(total_batch_test)
        print("Testing Accuracy= " + \
              "{:.5f}".format(test_accuracy_total))
