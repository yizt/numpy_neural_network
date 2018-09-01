# -*- coding: utf-8 -*-
"""
Created on 2018/8/31 20:33

@author: mick.yi
激活层

"""
import numpy as np


def relu_forword(z):
    """
    relu前向传播
    :param z: 待激活层
    :return: 激活后的结果
    """
    return np.max(0, z)


def relu_backword(next_dz,z):
    """
    relu反向传播
    :param next_dz: 激活后的梯度
    :param z: 激活前的值
    :return:
    """
    dz = np.where(np.greater(z, 0), next_dz, 0)
    return dz
