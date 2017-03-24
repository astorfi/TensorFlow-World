# Siamese Architecture for face recognition

import random
import numpy as np
import time
import tensorflow as tf
import math
import pdb
import sys
import scipy.io as sio
from sklearn import *
import matplotlib.pyplot as plt

def Plot_HIST_Fn(label,distance, phase, status, num_bins = 50):

    dissimilarity = distance[:]
    gen_dissimilarity_original = []
    imp_dissimilarity_original = []
    for i in range(len(label)):
        if label[i] == 1:
            gen_dissimilarity_original.append(dissimilarity[i])
        else:
            imp_dissimilarity_original.append(dissimilarity[i])

    bins = np.linspace(np.amin(distance), np.amax(distance), num_bins)
    fig = plt.figure()
    plt.hist(gen_dissimilarity_original, bins, alpha=0.5, facecolor='blue', normed=False, label='gen_dist_original')
    plt.hist(imp_dissimilarity_original, bins, alpha=0.5, facecolor='red', normed=False, label='imp_dist_original')
    plt.legend(loc='upper right')
    plt.title(phase + '_' + status + '_' + 'OriginalFeatures_Histogram.jpg')
    plt.show()
    fig.savefig(phase + '_' + status + '_' + 'OriginalFeatures_Histogram.jpg')
