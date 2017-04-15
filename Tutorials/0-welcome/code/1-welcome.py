#####################################################
########## Welcome to TensorFlow World ##############
#####################################################

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

# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS

################################################
################# handling errors!##############
################################################

# The user is prompted to input an absolute path.
if not os.path.isabs(FLAGS.log_dir):
    raise ValueError('You must assign absolute path for --log_dir')


# Defining some constant values
a = tf.constant(5.0, name="a")
b = tf.constant(10.0, name="b")

# Some basic operations
x = tf.add(a, b, name="add")
y = tf.div(a, b, name="divide")

# Run the session
with tf.Session() as sess:
    writer = tf.summary.FileWriter('log_dir', sess.graph)
    print("a =", sess.run(a))
    print("b =", sess.run(b))
    print("a + b =", sess.run(x))
    print("a/b =", sess.run(y))

# Closing the writer.
writer.close()


