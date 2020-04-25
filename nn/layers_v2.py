# -*- coding: utf-8 -*-
"""
 @File    : layers_v2.py
 @Time    : 2020/4/25 上午9:15
 @Author  : yizuotian
 @Description    : v2版前向、反向计算；解决卷积计算速度慢的问题
"""
import time

import numpy as np


def _single_channel_conv_v1(z, K, b=0, padding=(0, 0)):
    """
    当通道卷积操作
    :param z: 卷积层矩阵
    :param K: 卷积核
    :param b: 偏置
    :param padding: padding
    :return: 卷积结果
    """
    padding_z = np.lib.pad(z, ((padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    height, width = padding_z.shape
    k1, k2 = K.shape
    conv_z = np.zeros((1 + (height - k1), 1 + (width - k2)))
    for h in np.arange(height - k1 + 1):
        for w in np.arange(width - k2 + 1):
            conv_z[h, w] = np.sum(padding_z[h:h + k1, w:w + k2] * K)
    return conv_z + b


def _single_channel_conv(z, K, b=0, padding=(0, 0)):
    """
    当通道卷积操作
    :param z: 卷积层矩阵
    :param K: 卷积核
    :param b: 偏置
    :param padding: padding
    :return: 卷积结果
    """
    padding_z = np.lib.pad(z, ((padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    height, width = padding_z.shape
    k1, k2 = K.shape
    oh, ow = (1 + (height - k1), 1 + (width - k2))  # 输出的高度和宽度
    conv_z = np.zeros((1 + (height - k1), 1 + (width - k2)))
    # 遍历卷积比遍历特征高效
    for i in range(k1):
        for j in range(k2):
            conv_z += padding_z[i:i + oh, j:j + ow] * K[i, j]

    return conv_z + b


def test_single_conv():
    """
    两个卷积结果一样，速度相差百倍以上
    :return:
    """
    z = np.random.randn(224, 224)
    K = np.random.randn(3, 3)

    s = time.time()
    o1 = _single_channel_conv_v1(z, K)
    print("v1 耗时:{}".format(time.time() - s))
    s = time.time()
    o2 = _single_channel_conv(z, K)
    print("v2 耗时:{}".format(time.time() - s))

    print(np.allclose(o1, o2))


if __name__ == '__main__':
    test_single_conv()
