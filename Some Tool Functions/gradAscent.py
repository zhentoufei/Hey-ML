# -*- coding:utf-8 -*-

import numpy as np
import math


def grad_ascent(data_array, label_array, alpha, max_cycles):
    data_mat = np.mat(data_array)
    label_mat = np.mat(label_array)
    m, n = data_mat.shape
    weights = np.ones((n, 1))
    for i in range(max_cycles):
        h = sigmoid(data_mat * weights)
        error = label_mat - h  # size:m*1
        weights = weights + alpha * data_mat.transpose() * error
    return weights


def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


def classify(test_data, weight):
    data_array, label_array = test_data[0], test_data[1]
    data_mat = np.mat(data_array)
    label_mat = np.mat(label_array)
    h = sigmoid(data_mat * weight)
    m = len(h)
    err = 0.0
    for i in range(m):
        if h[i] > 0.5:
            print("{0} is classified as: 1".format(label_mat[i]))
            if int(label_mat[i]) != 1:
                err+=1
                print "error"
        else:
            print "{0} is classified as: 2".format(label_mat[i])
            if int(label_mat[i]) != 0:
                err+=1
                print "error"
    print "error rate is:", '%.4f'%(err/m)


def do_logistic(train_data, test_data, alpha = 0.07, max_cycle = 10):
    data, label = train_data[0], train_data[1]
    weights = grad_ascent(data, label, alpha, max_cycle)
    classify(test_data, weights)



if __name__ =='__main__':
    print "error rate is:", '%.4f'%(3.0)