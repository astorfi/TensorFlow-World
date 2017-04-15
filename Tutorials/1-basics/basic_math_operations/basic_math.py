from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os


######################################
######### Necessary Flags ############
######################################
tf.app.flags.DEFINE_string(
    'log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
    'Directory where event logs are written to.')

tf.app.flags.DEFINE_integer('num_classes', 10,
                            'Number of model clones to deploy.')

#########################################
########## status flags #################
#########################################
tf.app.flags.DEFINE_boolean('is_training', False,
                            'Training/Testing.')

# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS

################################################
################# handling errors!##############
################################################
if not os.path.isabs(FLAGS.log_dir):
    raise ValueError('You must assign absolute path for --log_dir')


# Define constant
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
with tf.Session() as sess:
    writer = tf.summary.FileWriter('log_dir', sess.graph)
    print(sess.run(x))

# Closing the writer.
writer.close()


