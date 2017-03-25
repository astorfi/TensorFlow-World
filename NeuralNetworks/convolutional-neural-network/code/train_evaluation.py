from __future__ import print_function
import tensorflow as tf
import numpy as np
from auxiliary import progress_bar
import os
import sys

def train(sess, saver, tensors, data, train_dir, finetuning, online_test,
                num_epochs, checkpoint_dir, batch_size):
    """
    This function run the session whether in training or evaluation mode.
    :param sess: The default session.
    :param saver: The saver operator to save and load the model weights.
    :param tensors: The tensors dictionary defind by the gragh.
    :param data: The data structure.
    :param train_dir: The training dir which is a reference for saving the logs and model checkpoints.
    :param finetuning: If fine tuning should be done or random initialization is needed.
    :param num_epochs: Number of epochs for training.
    :param checkpoint_dir: The directory of the checkpoints.
    :param batch_size: The training batch size.

    :return:
             Run the session.
    """

    # The prefix for checkpoint files
    checkpoint_prefix = 'model'

    ###################################################################
    ########## Defining the summary writers for train/test ###########
    ###################################################################

    train_summary_dir = os.path.join(train_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir)
    train_summary_writer.add_graph(sess.graph)

    test_summary_dir = os.path.join(train_dir, "summaries", "test")
    test_summary_writer = tf.summary.FileWriter(test_summary_dir)
    test_summary_writer.add_graph(sess.graph)

    # If fie-tuning flag in 'True' the model will be restored.
    if finetuning:
        saver.restore(sess, os.path.join(checkpoint_dir, checkpoint_prefix))
        print("Model restored for fine-tuning...")

    ###################################################################
    ########## Run the training and loop over the batches #############
    ###################################################################
    for epoch in range(num_epochs):
        total_batch_training = int(data.train.images.shape[0] / batch_size)

        # go through the batches
        for batch_num in range(total_batch_training):
            #################################################
            ########## Get the training batches #############
            #################################################

            start_idx = batch_num * batch_size
            end_idx = (batch_num + 1) * batch_size

            # Fit training using batch data
            train_batch_data, train_batch_label = data.train.images[start_idx:end_idx], data.train.labels[
                                                                                        start_idx:end_idx]

            ########################################
            ########## Run the session #############
            ########################################

            # Run optimization op (backprop) and Calculate batch loss and accuracy
            # When the tensor tensors['global_step'] is evaluated, it will be incremented by one.
            batch_loss, _, train_summaries, training_step = sess.run(
                [tensors['cost'], tensors['train_op'], tensors['summary_train_op'],
                 tensors['global_step']],
                feed_dict={tensors['image_place']: train_batch_data,
                           tensors['label_place']: train_batch_label,
                           tensors['dropout_param']: 0.5})

            ########################################
            ########## Write summaries #############
            ########################################

            # Write the summaries
            train_summary_writer.add_summary(train_summaries, global_step=training_step)

            # # Write the specific summaries for training phase.
            # train_summary_writer.add_summary(train_image_summary, global_step=training_step)

            #################################################
            ########## Plot the progressive bar #############
            #################################################

            progress = float(batch_num + 1) / total_batch_training
            progress_bar.print_progress(progress, epoch_num=epoch + 1, loss=batch_loss)

        # ################################################################
        # ############ Summaries per epoch of training ###################
        # ################################################################
        train_epoch_summaries = sess.run(tensors['summary_epoch_train_op'],
                                         feed_dict={tensors['image_place']: train_batch_data,
                                                    tensors['label_place']: train_batch_label,
                                                    tensors['dropout_param']: 0.5})

        # Put the summaries to the train summary writer.
        train_summary_writer.add_summary(train_epoch_summaries, global_step=training_step)

        #####################################################
        ########## Evaluation on the test data #############
        #####################################################

        if online_test:

            # WARNING: In this evaluation the whole test data is fed. In case the test data is huge this implementation
            #          may lead to memory error. In presense of large testing samples, batch evaluation on testing is
            #          recommended as in the training phase.
            test_accuracy_epoch, test_summaries = sess.run([tensors['accuracy'], tensors['summary_test_op']],
                                                           feed_dict={tensors['image_place']: data.test.images,
                                                                      tensors[
                                                                          'label_place']: data.test.labels,
                                                                      tensors[
                                                                          'dropout_param']: 1.})
            print("Epoch " + str(epoch + 1) + ", Testing Accuracy= " + \
                  "{:.5f}".format(test_accuracy_epoch))

            ###########################################################
            ########## Write the summaries for test phase #############
            ###########################################################

            # Returning the value of global_step if necessary
            current_step = tf.train.global_step(sess, tensors['global_step'])

            # Add the couter of global step for proper scaling between train and test summuries.
            test_summary_writer.add_summary(test_summaries, global_step=current_step)

    ###########################################################
    ############ Saving the model checkpoint ##################
    ###########################################################

    # # The model will be saved when the training is done.

    # Create the path for saving the checkpoints.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # save the model
    save_path = saver.save(sess, os.path.join(checkpoint_dir, checkpoint_prefix))
    print("Model saved in file: %s" % save_path)


    ############################################################################
    ########## Run the session for pur evaluation on the test data #############
    ############################################################################
def evaluation(sess, saver, tensors, data, checkpoint_dir):

        # The prefix for checkpoint files
        checkpoint_prefix = 'model'

        # Restoring the saved weights.
        saver.restore(sess, os.path.join(checkpoint_dir, checkpoint_prefix))
        print("Model restored...")

        # Evaluation of the model
        test_accuracy = 100 * sess.run(tensors['accuracy'], feed_dict={tensors['image_place']: data.test.images,
                                                                       tensors[
                                                                           'label_place']: data.test.labels,
                                                                       tensors[
                                                                           'dropout_param']: 1.})

        print("Final Test Accuracy is %% %.2f" % test_accuracy)
