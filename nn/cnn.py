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

from losses import cross_entropy_loss
from optimizers import *
from utils import load_cifar, save_weights, load_weights
from vgg import VGG


def get_accuracy(net, xs, ys, batch_size=128):
    """

    :param net:
    :param xs:
    :param ys:
    :param batch_size:
    :return:
    """
    scores = np.zeros_like(ys, dtype=np.float)
    num = xs.shape[0]
    for i in range(num // batch_size):
        s = i * batch_size
        e = s + batch_size
        scores[s:e] = net.forward(xs[s:e])
    # 余数处理
    remain = num % batch_size
    if remain > 0:
        scores[-remain:] = net.forward(xs[-remain:])
    # 计算精度
    acc = np.mean(np.argmax(scores, axis=1) == np.argmax(ys, axis=1))
    return acc


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
    # opt = RmsProp(vgg.weights, lr=args.lr, decay=1e-3)
    opt = SGD(vgg.weights, lr=args.lr, decay=args.decay)
    opt.iterations = args.init_step

    # 加载权重
    if args.checkpoint:
        weights = load_weights(args.checkpoint)
        vgg.load_weights(weights)
        print("load weights done")

    # 评估
    if args.eval_only:
        indices = np.random.choice(len(x_test), args.eval_num, replace=False)
        print('{} start evaluate'.format(time.asctime(time.localtime(time.time()))))
        acc = get_accuracy(vgg, x_test[indices], y_test[indices], args.batch_size)
        print('{} acc on test dataset is :{:.3f}'.format(time.asctime(time.localtime(time.time())),
                                                         acc))
        return

    # 训练
    num_steps = args.steps
    for step in range(args.init_step, num_steps):
        x, y_true = next_batch(args.batch_size)
        # 前向传播
        y_predict = vgg.forward(x.astype(np.float))
        # print('y_pred: min{},max{},mean:{}'.format(np.min(y_predict, axis=-1),
        #                                            np.max(y_predict, axis=-1),
        #                                            np.mean(y_predict, axis=-1)))
        # print('y_pred: {}'.format(y_predict))
        acc = np.mean(np.argmax(y_predict, axis=1) == np.argmax(y_true, axis=1))
        # 计算loss
        loss, gradient = cross_entropy_loss(y_predict, y_true)

        # 反向传播
        vgg.backward(gradient)
        # 更新梯度
        opt.iterate(vgg)

        # 打印信息
        print('{} step:{},loss:{:.4f},acc:{:.4f}'.format(time.asctime(time.localtime(time.time())),
                                                         step,
                                                         loss,
                                                         acc))

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
    parse.add_argument('--lr', type=float, default=1e-2)
    parse.add_argument('--decay', type=float, default=1e-3)
    parse.add_argument('-s', '--steps', type=int, default=10000)
    parse.add_argument('--eval-only', action='store_true', default=False)
    parse.add_argument('--eval-num', type=int, default=100)
    parse.add_argument('--init-step', type=int, default=0)
    arguments = parse.parse_args()
    main(arguments)
