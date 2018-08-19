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
    dw = np.dot(z.reshape(1,-1).T, next_dz)  # 当前层权重的梯度
    db = next_dz  # 当前层偏置的梯度
    return dw, db, dz

np.random.randn()