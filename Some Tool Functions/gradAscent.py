# -*- coding:utf-8 -*-

import numpy as np
import math
def grad_ascent(data_array, label_array, alpha, max_cycles):
    data_mat = np.mat(data_array)
    label_mat = np.mat(label_array)
    m, n = data_mat.shape
    weights = np.ones((n,1))
    for i in range(max_cycles):
        h = sigmoid(data_mat * weights)
        error = label_mat - h  # size:m*1
        weigh = weights + alpha * data_mat.transpose() * error
    return weights

def sigmoid(x):
    return 1.0/(1+math.exp(-x))
