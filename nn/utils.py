# -*- coding: utf-8 -*-
"""
Created on 2018/9/2 9:30

@author: mick.yi
工具类
"""
import pickle
from six.moves import cPickle
import numpy as np
import os


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


def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_cifar(path):
    """Loads CIFAR10 dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))
    # 归一化
    # x_train = x_train.astype(np.float) / 255. - 1.
    # x_test = x_test.astype(np.float) / 255. - 1.
    mean = np.array([123.680, 116.779, 103.939])
    x_train = x_train.astype(np.float) - mean[:, np.newaxis, np.newaxis]
    x_test = x_test.astype(np.float) - mean[:, np.newaxis, np.newaxis]
    x_train /= 255.
    x_test /= 255
    std = np.array([0.24580306, 0.24236229, 0.2603115])
    x_train /= std[:, np.newaxis, np.newaxis]
    x_test /= std[:, np.newaxis, np.newaxis]
    return (x_train, to_categorical(y_train)), (x_test, to_categorical(y_test))
