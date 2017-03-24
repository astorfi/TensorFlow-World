# Siamese Multi-layer-Perceptron

The code is for implementation of the Siamese architecture using Tensorflow. This implementation contains two identical MLP(multi layer perceptron) followed by a contrastive loss cost function. The system platform has to be chosen as follows:
![](Images/siamese.png) 

The Siamese architecture is used for training a similarity metric from data. The method is used for recognition or verification applications where the number of classes is very large and not known during training, and where the number of
training samples for a single class is very small and the class is not important. The idea is to learn a function that maps input patterns into a target space with the impostor pairs(pairs which do not belong to the same class) become as separated as possible and genuine pairs(pairs which belong to the same class) become as close as possible using the simple distance metrics.

## Overview

The aim is to design a simple siamese multi-layer perceptron using `TensorFlow`. The tutorial is aimed to sketch a starup model to the the two follwing:

1. Define an organization for the network architecture, training and evaluation phases.
2. Provides a template framework for constructing larger and more complicated models.

## Model Architecture

Three simple `fully-cnnected` layers are included. 

## Code Organization

The source code is embeded in `code` folder.

| File                  | Explanation   |
| ----------------------|:-------------:|
| siamese_structure.py  | The body of the framework which consists of structure and axillary functions |
| SiameseMLP.py         | The main file which has to be run |

## Input

The input format is `HDF5` for this implemetation but it basically can be anything as long as it satisfies the shape properties. For each `TRAIN` and `TEST` data, there are attributes call `cube` and `label`. The `cube` is the data of the shape `(Number_Samples,number_features,Number_Channels)` and the `label` is of the form `(Number_Samples,1)` in which each row has the class label. `Number_Channels` must be `2` because we are dealing with pairs of features.

## Training

As conventional procedure, updating the gradient is done with batches of the data. Moreover `Batch Normalization` has been implemented for each fully-connected layer. For all the fully-connected layers, the `drop-out` has been used with the same parameter however this parameter can be customized for each layer in the code. Traditional `GradientDescentOptimizer` has been used.



# Example

The code implementation must separate the the data in the output space. Since for this open source project the application of Siamese Architecture is face verification, ROC curve can be good representative of the results. Moreover since the separation in output domain is crucial, the histogram representation of the data at input and output is of great interest.
The examples are present in **Sample-Results** folder.

# Motivation

The reason behind Tensorflow implementation of Siamese Archiecture with MLP is that other implementations like the ones in Caffe does not have too much flexibility and its due to the nature of Caffe. Moreover to the best of our knowldege, there was not any other implemetation on the web to be able to do the tasks related to our under_study application.

# Implementation

The implementation part divided to some parts. Data calling and output demonstration.

## Data Input

The data input type is HDF5. HDF5 is easy to handle and there are lots of documentation about that on the web.
It can be read by using the simple code below:
```
TRAIN_FILE = h5py.File('TRAIN.hdf5', 'r')
TEST_FILE = h5py.File('TEST.hdf5', 'r')
X_train = TRAIN_FILE['pairs']
y_train = TRAIN_FILE['labels']
X_test = TEST_FILE['pairs']
y_test = TEST_FILE['labels']
```

The labels and pairs itself(recall that the application is face verification) should be saved with different names for convenient.

## Plotting Results

Two kind of curves are considered for plotting. ROC curve and Precision-Recall curve. The second one is more precise specifically if there is a huge difference between the number of genuine and impostor pairs. Moreover the histogram of the input and output should be demonstrated as well to reflect the discriminative ability of the structure. For the training phase the output can be depicted by the following code.

```
# Plot ROC for training
Plot_ROC_Fn(label_train, distance_original_train, 'Train', 'Input')
Plot_ROC_Fn(label_train, distance_train, 'Train', 'Output')

# Plot PR for training
Plot_PR_Fn(label_train, distance_original_train , 'Train', 'Input')
Plot_PR_Fn(label_train, distance_train, 'Train', 'Output')


# Plot HIST for training
Plot_HIST_Fn(label_train, distance_original_train, 'Train', 'Input')
Plot_HIST_Fn(label_train, distance_train, 'Train', 'Output')
```





