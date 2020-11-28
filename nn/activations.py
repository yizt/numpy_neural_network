# -*- coding: utf-8 -*-
"""
Created on 2018/8/31 20:33

@author: mick.yi
激活层
参考：https://blog.csdn.net/csuyzt/article/details/82320589 中的推导公式
"""
import numpy as np


def sigmoid_forward(z):
    """
    sigmoid激活前向过程
    :param z:
    :return:
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_backward(next_dz, z):
    """
    sigmoid激活反向过程
    :param next_dz:
    :param z:
    :return:
    """
    return sigmoid_forward(z) * (1 - sigmoid_forward(z)) * next_dz


def tanh_forward(z):
    """
    tanh激活前向过程
    :param z:
    :return:
    """
    return np.tanh(z)


def tanh_backward(next_dz, z):
    """
    tanh激活反向过程
    :param next_dz:
    :param z:
    :return:
    """
    return next_dz(1 - np.square(np.tanh(z)))


def relu_forward(z):
    """
    relu前向传播
    :param z: 待激活层
    :return: 激活后的结果
    """
    return np.maximum(0, z)


def relu_backward(next_dz, z):
    """
    relu反向传播
    :param next_dz: 激活后的梯度
    :param z: 激活前的值
    :return:
    """
    dz = np.where(np.greater(z, 0), next_dz, 0)
    return dz


def lrelu_forward(z, alpha=0.1):
    """
    leaky relu前向传播
    :param z: 待激活层
    :param alpha: 常量因子
    :return: 激活后的结果
    """
    return np.where(np.greater(z, 0), z, alpha * z)


def lrelu_backward(next_dz, z, alpha=0.1):
    """
    leaky relu反向传播
    :param next_dz: 激活后的梯度
    :param z: 激活前的值
    :param alpha: 常量因子
    :return:
    """
    dz = np.where(np.greater(z, 0), next_dz, alpha * next_dz)
    return dz


def prelu_forward(z, alpha):
    """
    PReLu 前向传播
    :param z: 输入
    :param alpha:需要学习的参数
    :return:
    """
    return np.where(np.greater(z, 0), z, alpha * z)


def prelu_backwark(next_dz, z, alpha):
    """
    PReLu 后向传播
    :param next_dz: 输出层的梯度
    :param z: 输入
    :param alpha:需要学习的参数
    :return:
    """
    dz = np.where(np.greater(z, 0), next_dz, alpha * next_dz)
    dalpha = np.where(np.greater(z, 0), 0, z * next_dz)
    return dalpha, dz


def elu_forward(z, alpha=0.1):
    """
    elu前向传播
    :param z: 输入
    :param alpha: 常量因子
    :return:
    """
    return np.where(np.greater(z, 0), z, alpha * (np.exp(z) - 1))


def elu_backward(next_dz, z, alpha=0.1):
    """
    elu反向传播
    :param next_dz: 输出层梯度
    :param z: 输入
    :param alpha: 常量因子
    :return:
    """
    return np.where(np.greater(z, 0), next_dz, alpha * next_dz * np.exp(z))
