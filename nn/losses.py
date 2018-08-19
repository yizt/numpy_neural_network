# -*- coding: utf-8 -*-
"""
Created on 2018/8/19 15:03

@author: mick.yi

定义损失函数
"""
import numpy as np


def mean_squared_loss(y_predict,y_true):
    """
    均方误差损失函数
    :param y_predict: 预测值,shape (N,d)，N为批量样本数
    :param y_true: 真实值
    :return:
    """
    loss = np.mean(np.square(y_predict-y_true, axis=-1))  # 损失函数值
    dy = np.mean(y_predict - y_true)  # 损失函数关于网络输出的梯度
    return loss, dy
