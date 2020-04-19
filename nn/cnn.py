# -*- coding: utf-8 -*-
"""
 @File    : cnn.py
 @Time    : 2020/4/18 下午5:54
 @Author  : yizuotian
 @Description    : 卷积网络
"""
import argparse
import os
import time

from six.moves import cPickle

from losses import cross_entropy_loss
from optimizers import *
from utils import to_categorical, save_weights, load_weights
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
    # x_train = x_train.astype(np.float) / 255. - 1.
    # x_test = x_test.astype(np.float) / 255. - 1.
    mean = np.array([123.680, 116.779, 103.939])
    x_train = x_train.astype(np.float) - mean[:, np.newaxis, np.newaxis]
    x_test = x_test.astype(np.float) - mean[:, np.newaxis, np.newaxis]
    return (x_train, to_categorical(y_train)), (x_test, to_categorical(y_test))


def main(args):
    # 数据加载
    (x_train, y_train), (x_test, y_test) = load_cifar(args.cifar_root)

    # 随机选择训练样本
    train_num = x_train.shape[0]

    def next_batch(batch_size):
        idx = np.random.choice(train_num, batch_size)
        return x_train[idx], y_train[idx]

    # 网络
    vgg = VGG(image_size=32, name='vgg11')
    opt = SGD(vgg.weights, lr=args.lr, decay=1e-3)

    # 加载权重
    if args.checkpoint:
        weights = load_weights(args.checkpoint)
        vgg.load_weights(weights)
        print("load weights done")

    # 训练
    num_steps = args.steps
    for step in range(num_steps):
        x, y_true = next_batch(args.batch_size)
        # 前向传播
        y_predict = vgg.forward(x.astype(np.float32))
        # print('y_pred: min{},max{},mean:{}'.format(np.min(y_predict, axis=-1),
        #                                            np.max(y_predict, axis=-1),
        #                                            np.mean(y_predict, axis=-1)))
        # print('y_pred: {}'.format(y_predict))
        # 计算loss
        loss, gradient = cross_entropy_loss(y_predict, y_true)

        # 反向传播
        vgg.backward(gradient)
        # 更新梯度
        opt.iterate(vgg)

        # 打印信息
        print('{} step:{},loss:{}'.format(time.asctime(time.localtime(time.time())),
                                          step, loss))

        # 保存权重
        if step % 100 == 0:
            save_weights(os.path.join(args.save_dir, 'weights-{:03d}.pkl'.format(step)),
                         vgg.weights)


def test(path):
    (x_train, y_train), (x_test, y_test) = load_cifar(path)
    print(x_train[0][0])
    print(y_train[0])
    vgg = VGG(name='vgg11')
    import utils
    utils.save_weights('./w.pkl', vgg.weights)
    w = utils.load_weights('./w.pkl')
    print(type(w))
    print(w.keys())


if __name__ == '__main__':
    # cifar_root = '/Users/yizuotian/dataset/cifar-10-batches-py'
    # test(cifar_root)

    parse = argparse.ArgumentParser()
    parse.add_argument('-d', '--cifar-root', type=str,
                       default='/Users/yizuotian/dataset/cifar-10-batches-py')
    parse.add_argument('-o', '--save-dir', type=str, default='/tmp')
    parse.add_argument('-c', '--checkpoint', type=str, default=None)
    parse.add_argument('-b', '--batch-size', type=int, default=32)
    parse.add_argument('--lr', type=float, default=2e-2)
    parse.add_argument('-s', '--steps', type=int, default=10000)
    arguments = parse.parse_args()
    main(arguments)
