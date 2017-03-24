# Convolutional Neural Network

## Overview

The aim is to design a simple convolutional Neural Network using `TensorFlow`. The tutorial is aimed to sketch a starup model to the the two follwing:

1. Define an organization for the network architecture, training and evaluation phases.
2. Provides a template framework for constructing larger and more complicated models.

## Model Architecture

Two simple `convolutional layers`(each have max pooling) followed by two `fully-cnnected` layers conisdered. The number of output units for the last fully-connected layer is equal to the number of `classes` becasue a `softmax` has been implemented for the classification task.

## Code Organization

The source code is embeded in `code` folder.

| File                | Explanation   |
| ------------------- |:-------------:|
| Model_Functions.py  | The body of the framework which consists of structure and axillary functions |
| classifier.py       | The main file which has to be run |

## Input

The input format is `HDF5` for this implemetation but it basically can be anything as long as it satisfies the shape properties. For each `TRAIN` and `TEST` data, there are attributes call `cube` and `label`. The `cube` is the data of the shape `(Number_Samples,height,width,Number_Channels)` and the `label` is of the form `(Number_Samples,1)` in which each row has the class label. The label matrix should be transform to the form of `(Number_Samples,Number_Classes)` for which each row is associated with a sample and all of the columns are zero except for the one which belongs to the class number. Method `Reform_Fn` does this in the begining.

## Training

As conventional procedure, updating the gradient is done with batches of the data. Moreover `Batch Normalization` has been implemented for each convolutional layer. No `Batch Normalization` is done for the fully-connected layers. For all the convolutional and fully-connected layers, the `drop-out` has been used with the same parameter however this parameter can be customized for each layer in the code. Traditional `GradientDescentOptimizer` has been used.

### Loss Function

`Cross-entropy` loss function has been chosen for the cost of system. The definition is as follows:
```
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
```
However by considering [TensorFlow official documention](https://www.tensorflow.org/versions/r0.11/tutorials/mnist/beginners/index.html#mnist-for-ml-beginners) it is numerically unstable. The reason can be due to the presence of the `log`. If the output of the network provide a very bad prediction and by normalizing that prediction we get `zero` for `y` value, then the loss goes to infinity and this is unstable. Another issue can be the explosion of the exponential. If the output of any of the neurons is large, since in the softmax we do the exponentiation, then the numerator and denominator of the softmax operation can be very large. So a trick can be add a number to all of the unscaled outputs. all the unit output values can be added by `-max{fi{i=0,...,n}}` which is the ngative sign of maximum of all output values. For further reading refer to [CNN for Visual Recognition](http://cs231n.github.io/linear-classify/) Course by stanford. Also please refer to [softmax_regression](http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression) for further details.

Instead the `tf.nn.softmax_cross_entropy_with_logits` on the unnormalized logits (i.e.,  softmax_cross_entropy_with_logits is called on tf.matmul(x, W) + b), this function computes the softmax activation internally which makes it more stable. It's good to take a look at the source code of `TensorFlow` for that however there is a traditional idea to overcome this problem which is add an `epsilon` number with the absolute value of `y` and take it as `y`.
