============================
Welcome to TensorFlow World
============================

.. _this link: https://github.com/astorfi/TensorFlow-World/blob/master/Tutorials/0-welcome/README.rst

The tutorials in this section is just a start for going into TensorFlow world. The source code is available at `this link`_.

We using Tensorboard for visualizing the outcomes. TensorBoard is the graph visualization tools provided by TensorFlow. Using Google’s words: “The computations you'll use TensorFlow for - like training a massive deep neural network - can be complex and confusing. To make it easier to understand, debug, and optimize TensorFlow programs, we've included a suite of visualization tools called TensorBoard.” A simple Tensorboard implementation is used in this tutorial. The details of Tensorboard and its advantages will be presented in more advanced tutorials.


--------------------------
Prepairing the environment
--------------------------

At first we have to import the necessary libraries.

.. code:: python
    
       from __future__ import print_function
       import tensorflow as tf
       import os

Since we are aimed to use Tensorboard, we need a directory to store the information (the operations and their corresponding outputs if desired by the user). These information are exported to ``event files`` by TensorFlow. The even files can be transformed to visual data such that the user be able to evaluate the architecture and the operations. The ``path`` to store these even files is defined as below:

.. code:: python
    
       # The default path for saving event files is the same folder of this python file.
       tf.app.flags.DEFINE_string(
       'log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
       'Directory where event logs are written to.')

       # Store all elemnts in FLAG structure!
       FLAGS = tf.app.flags.FLAGS

The ``os.path.dirname(os.path.abspath(__file__))`` gets the directory name of the current python file. The ``tf.app.flags.FLAGS`` points to all defined flags using the ``FLAGS`` indicator. From now on the flags can be called using ``FLAGS.flag_name``.
