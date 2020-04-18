# -*- coding: utf-8 -*-
"""
Created on 2018/9/2 9:30

@author: mick.yi
工具类
"""
import pickle

import numpy as np


def to_categorical(y, num_classes=None):
    """从keras中复制而来
    Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def save_weights(file_path, weights):
    """
    保存权重
    :param file_path:
    :param weights:
    :return:
    """
    f = open(file_path, 'wb')
    pickle.dump(weights, f)
    f.close()


def load_weights(file_path):
    """
    加载权重
    :param file_path:
    :return:
    """
    f = open(file_path, 'rb')
    weights = pickle.load(f)
    return weights
