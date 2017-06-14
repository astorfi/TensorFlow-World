import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tempfile
import urllib
import pandas as pd
import os
from tensorflow.examples.tutorials.mnist import input_data

######################################
######### Necessary Flags ############
######################################

tf.app.flags.DEFINE_string(
    'train_dir', os.path.dirname(os.path.abspath(__file__)) + '/train_logs',
    'Directory where event logs are written to.')

tf.app.flags.DEFINE_string(
    'checkpoint_dir',
    os.path.dirname(os.path.abspath(__file__)) + '/checkpoints',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_integer('max_num_checkpoint', 10,
                            'Maximum number of checkpoints that TensorFlow will keep.')

tf.app.flags.DEFINE_integer('num_classes', 2,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_integer('batch_size', np.power(2, 9),
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_integer('num_epochs', 10,
                            'Number of epochs for training.')

##########################################
######## Learning rate flags #############
##########################################
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.95, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 1, 'Number of epoch pass to decay learning rate.')

#########################################
########## status flags #################
#########################################
tf.app.flags.DEFINE_boolean('is_training', False,
                            'Training/Testing.')

tf.app.flags.DEFINE_boolean('fine_tuning', False,
                            'Fine tuning is desired or not?.')

tf.app.flags.DEFINE_boolean('online_test', True,
                            'Fine tuning is desired or not?.')

tf.app.flags.DEFINE_boolean('allow_soft_placement', True,
                            'Automatically put the variables on CPU if there is no GPU support.')

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Demonstrate which variables are on what device.')

# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS


################################################
################# handling errors!##############
################################################
if not os.path.isabs(FLAGS.train_dir):
    raise ValueError('You must assign absolute path for --train_dir')

if not os.path.isabs(FLAGS.checkpoint_dir):
    raise ValueError('You must assign absolute path for --checkpoint_dir')

# Download and get MNIST dataset(available in tensorflow.contrib.learn.python.learn.datasets.mnist)
# It checks and download MNIST if it's not already downloaded then extract it.
# The 'reshape' is True by default to extract feature vectors but we set it to false to we get the original images.
mnist = input_data.read_data_sets("MNIST_data/", reshape=True, one_hot=False)

########################
### Data Processing ####
########################
# Organize the data and feed it to associated dictionaries.
data={}

data['train/image'] = mnist.train.images
data['train/label'] = mnist.train.labels
data['test/image'] = mnist.test.images
data['test/label'] = mnist.test.labels

# Get only the samples with zero and one label for training.
index_list_train = []
for sample_index in range(data['train/label'].shape[0]):
    label = data['train/label'][sample_index]
    if label == 1 or label == 0:
        index_list_train.append(sample_index)

# Reform the train data structure.
data['train/image'] = mnist.train.images[index_list_train]
data['train/label'] = mnist.train.labels[index_list_train]


# Get only the samples with zero and one label for test set.
index_list_test = []
for sample_index in range(data['test/label'].shape[0]):
    label = data['test/label'][sample_index]
    if label == 1 or label == 0:
        index_list_test.append(sample_index)

# Reform the test data structure.
data['test/image'] = mnist.test.images[index_list_test]
data['test/label'] = mnist.test.labels[index_list_test]

# Dimentionality of train
dimensionality_train = data['train/image'].shape

# Dimensions
num_train_samples = dimensionality_train[0]
num_features = dimensionality_train[1]

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

    # learning rate policy
    decay_steps = int(num_train_samples / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               FLAGS.learning_rate_decay_factor,
                                               staircase=True,
                                               name='exponential_decay_learning_rate')

    ###############################################
    ########### Defining place holders ############
    ###############################################
    image_place = tf.placeholder(tf.float32, shape=([None, num_features]), name='image')
    label_place = tf.placeholder(tf.int32, shape=([None,]), name='gt')
    label_one_hot = tf.one_hot(label_place, depth=FLAGS.num_classes, axis=-1)
    dropout_param = tf.placeholder(tf.float32)

    ##################################################
    ########### Model + Loss + Accuracy ##############
    ##################################################
    # A simple fully connected with two class and a softmax is equivalent to Logistic Regression.
    logits = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs = FLAGS.num_classes, scope='fc')

    # Define loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot))

    # Accuracy
    with tf.name_scope('accuracy'):
        # Evaluate the model
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))

        # Accuracy calculation
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #############################################
    ########### training operation ##############
    #############################################

    # Define optimizer by its default values
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # 'train_op' is a operation that is run for gradient update on parameters.
    # Each execution of 'train_op' is a training step.
    # By passing 'global_step' to the optimizer, each time that the 'train_op' is run, Tensorflow
    # update the 'global_step' and increment it by one!

    # gradient update.
    with tf.name_scope('train'):
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


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

        # The prefix for checkpoint files
        checkpoint_prefix = 'model'

        # If fie-tuning flag in 'True' the model will be restored.
        if FLAGS.fine_tuning:
            saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, checkpoint_prefix))
            print("Model restored for fine-tuning...")

        ###################################################################
        ########## Run the training and loop over the batches #############
        ###################################################################
        total_batch_test = int(mnist.test.images.shape[0] / FLAGS.batch_size)

        # go through the batches
        test_accuracy = 0
        for epoch in range(FLAGS.num_epochs):
            total_batch_training = int(mnist.train.images.shape[0] / FLAGS.batch_size)

            # go through the batches
            for batch_num in range(total_batch_training):
                #################################################
                ########## Get the training batches #############
                #################################################

                start_idx = batch_num * FLAGS.batch_size
                end_idx = (batch_num + 1) * FLAGS.batch_size

                # Fit training using batch data
                train_batch_data, train_batch_label = mnist.train.images[start_idx:end_idx], mnist.train.labels[
                                                                                             start_idx:end_idx]

                ########################################
                ########## Run the session #############
                ########################################

                # Run optimization op (backprop) and Calculate batch loss and accuracy
                # When the tensor tensors['global_step'] is evaluated, it will be incremented by one.
                batch_loss, _, training_step = sess.run(
                    [loss, train_op,
                     global_step],
                    feed_dict={image_place: train_batch_data,
                               label_place: train_batch_label,
                               dropout_param: 0.5})

                ########################################
                ########## Write summaries #############
                ########################################


                #################################################
                ########## Plot the progressive bar #############
                #################################################

            print("Epoch " + str(epoch + 1) + ", Training Loss= " + \
                  "{:.5f}".format(batch_loss))

        ###########################################################
        ############ Saving the model checkpoint ##################
        ###########################################################

        # # The model will be saved when the training is done.

        # Create the path for saving the checkpoints.
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)

        # save the model
        save_path = saver.save(sess, os.path.join(FLAGS.checkpoint_dir, checkpoint_prefix))
        print("Model saved in file: %s" % save_path)

        ############################################################################
        ########## Run the session for pur evaluation on the test data #############
        ############################################################################

        # The prefix for checkpoint files
        checkpoint_prefix = 'model'

        # Restoring the saved weights.
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, checkpoint_prefix))
        print("Model restored...")

        # Evaluation of the model
        test_accuracy = 100 * sess.run(accuracy, feed_dict={
            image_place: mnist.test.images,
            label_place: mnist.test.labels,
            dropout_param: 1.})

        print("Final Test Accuracy is %% %.2f" % test_accuracy)
