# -*- coding: utf-8 -*-
"""
Created on 2018/9/4 22:26

@author: mick.yi

优化方法

"""
import numpy as np


class SGD(object):
    """
    小批量梯度下降法
    """

    def __init__(self, weights, lr=0.01, momentum=0.9, decay=1e-5):
        """

        :param weights: 权重，字典类型
        :param lr: 初始学习率
        :param momentum: 动量
        :param decay: 学习率衰减
        """
        self.v = self._copy_weights_to_zeros(weights)
        self.iterations = 0  # 迭代次数
        self.lr = lr
        self.momentum = momentum
        self.decay = decay

    def _copy_weights_to_zeros(self, weights):
        result = {}
        result.keys()
        for key in weights.keys():
            result[key] = np.zeros_like(weights[key])
        return result

    def iterator(self, weights, gradients):
        # 更新学习率
        self.lr /= (1 + self.iterations)
        # 更新动量和梯度
        for key in self.v.keys():
            self.v[key] = self.momentum * self.v[key] + self.lr * gradients[key]
            weights[key] = weights[key] - self.v[key]

        # 更新迭代次数
        self.iterations += 1

        # 返回更新后的权重
        return weights
