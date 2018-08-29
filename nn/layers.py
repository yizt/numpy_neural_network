# -*- coding: utf-8 -*-
"""
Created on 2018/8/19 15:03

@author: mick.yi

定义网络层
"""
import numpy as np


def fc_forword(z, W, b):
    """
    全连接层的前向传播
    :param z: 当前层的输出
    :param W: 当前层的权重
    :param b: 当前层的偏置
    :return: 下一层的输出
    """
    return np.dot(z, W) + b


def fc_backword(next_dz, W, z):
    """
    全连接层的反向传播
    :param next_dz: 下一层的梯度
    :param W: 当前层的权重
    :param z: 当前层的输出
    :return:
    """
    dz = np.dot(next_dz, W.T)  # 当前层的梯度
    dw = np.dot(z.T, next_dz)  # 当前层权重的梯度
    db = np.sum(next_dz, axis=0)  # 当前层偏置的梯度, N个样本的梯度求和
    return dw, db, dz


def _single_channel_conv(z, K, padding=(0, 0), strides=(1, 1)):
    """
    当通道卷积操作
    :param z: 卷积层矩阵
    :param K: 卷积核
    :param padding: padding
    :param strides: 步长
    :return: 卷积结果
    """
    padding_z = np.lib.pad(z, ((padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    hight, width = padding_z.shape
    k1, k2 = K.shape
    assert (hight - k1) % strides[0] == 0, '步长不为1时，步长必须刚好能够被整除'
    assert (width - k2) % strides[1] == 0, '步长不为1时，步长必须刚好能够被整除'
    conv_z = np.zeros((1 + (hight - k1) // strides[0], 1 + (width - k2) // strides[1]))
    for h in np.arange(hight - k1 + 1)[::strides[0]]:
        for w in np.arange(width - k2 + 1)[::strides[1]]:
            conv_z[h // strides[0], w // strides[1]] = np.sum(padding_z[h:h + k1, w:w + k2] * K)
    return conv_z


if __name__ == "__main__":
    z = np.ones((5, 5))
    k = np.ones((3, 3))
    #print(_single_channel_conv(z, k,padding=(1,1)))
    #print(_single_channel_conv(z, k, strides=(2, 2)))
    assert _single_channel_conv(z, k).shape == (3, 3)
    assert _single_channel_conv(z, k, padding=(1, 1)).shape == (5, 5)
    assert _single_channel_conv(z, k, strides=(2, 2)).shape == (2, 2)
    assert _single_channel_conv(z, k, strides=(2, 2),padding=(1, 1)).shape == (3, 3)
    assert _single_channel_conv(z, k, strides=(2, 2), padding=(1, 0)).shape == (3, 2)
    assert _single_channel_conv(z, k, strides=(2, 1), padding=(1, 1)).shape == (3, 5)
    #print(np.lib.pad(x,((1,1),(2,1)), 'constant', constant_values=0))
