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

def Plot_PR_Fn(label,distance,phase,status):

    precision, recall, thresholds = metrics.precision_recall_curve(label, -distance, pos_label=1, sample_weight=None)
    AP = metrics.average_precision_score(label, -distance, average='macro', sample_weight=None)

    # AP(average precision) calculation.
    # This score corresponds to the area under the precision-recall curve.
    print("AP = ", float(("{0:.%ie}" % 1).format(AP)))

    # Plot the ROC
    fig = plt.figure()
    ax = fig.gca()
    lines = plt.plot(recall, precision, label='ROC Curve')
    plt.setp(lines, linewidth=3, color='r')
    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 1, 0.1))
    plt.title(phase + '_' + status + '_' + 'PR.jpg')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # Cutting the floating number
    AP = '%.2f' % AP

    # Setting text to plot
    plt.text(0.5, 0.5, 'AP = ' + str(AP), fontdict=None)
    plt.grid()
    plt.show()
    fig.savefig(phase + '_' + status + '_' + 'PR.jpg')

