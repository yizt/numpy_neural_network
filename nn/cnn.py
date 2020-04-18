# -*- coding: utf-8 -*-
"""
 @File    : cnn.py
 @Time    : 2020/4/18 下午5:54
 @Author  : yizuotian
 @Description    : 卷积网络
"""
import os
import sys
import numpy as np
from six.moves import cPickle

from losses import cross_entropy_loss
from optimizers import SGD
from utils import to_categorical
from vgg import VGG



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
    x_train = x_train.astype(np.float) / 255. - 1.
    x_test = x_test.astype(np.float) / 255. - 1.

    return (x_train, to_categorical(y_train)), (x_test, to_categorical(y_test))


def main(path):
    # 数据加载
    (x_train, y_train), (x_test, y_test) = load_cifar(path)

    # 随机选择训练样本
    train_num = x_train.shape[0]

    def next_batch(batch_size):
        idx = np.random.choice(train_num, batch_size)
        return x_train[idx], y_train[idx]

    # 网络
    vgg = VGG(image_size=32, name='vgg11')
    sgd = SGD(vgg.weights)
    # 训练
    num_steps = 1000
    for step in range(num_steps):
        x, y_true = next_batch(4)
        # 前向传播
        y_predict = vgg.forward(x.astype(np.float))
        # 计算loss
        loss, gradient = cross_entropy_loss(y_predict, y_true)

        # 反向传播
        vgg.backward(gradient)
        # 更新梯度
        sgd.iterate(vgg.weights, vgg.gradients)

        # 打印信息
        print('step:{},loss:{}'.format(step, loss))


def test(path):
    (x_train, y_train), (x_test, y_test) = load_cifar(path)
    print(x_train[0][0])
    print(y_train[0])


if __name__ == '__main__':
    # cifar_root = '/Users/yizuotian/dataset/cifar-10-batches-py'
    # test(cifar_root)
    cifar_root = sys.argv[1]
    main(cifar_root)
