# -*- coding: utf-8 -*-
"""
Created on 2018/9/4 22:26

@author: mick.yi

优化方法

"""
import numpy as np


def _copy_weights_to_zeros(self, weights):
    result = {}
    result.keys()
    for key in weights.keys():
        result[key] = np.zeros_like(weights[key])
    return result


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
        self.v = _copy_weights_to_zeros(weights)
        self.iterations = 0  # 迭代次数
        self.lr = self.init_lr = lr
        self.momentum = momentum
        self.decay = decay

    def iterate(self, weights, gradients):
        """
        迭代一次
        :param weights: 当前迭代权重
        :param gradients: 当前迭代梯度
        :return:
        """
        # 更新学习率
        self.lr = self.init_lr / (1 + self.iterations)
        # 更新动量和梯度
        for key in self.v.keys():
            self.v[key] = self.momentum * self.v[key] + self.lr * gradients[key]
            weights[key] = weights[key] - self.v[key]

        # 更新迭代次数
        self.iterations += 1

        # 返回更新后的权重
        return weights


class AdaGrad(object):
    def __init__(self, weights, lr=0.01, epsilon=1e-6, decay=0):
        """

        :param weights: 权重
        :param lr: 学习率
        :param epsilon: 平滑数
        :param decay: 学习率衰减
        """
        self.s = _copy_weights_to_zeros(weights)  # 权重平方和累加量
        self.iterations = 0  # 迭代次数
        self.lr = self.init_lr = lr
        self.epsilon = epsilon
        self.decay = decay

    def iterate(self, weights, gradients):
        """
        迭代一次
        :param weights: 当前迭代权重
        :param gradients: 当前迭代梯度
        :return:
        """
        # 更新学习率
        self.lr = self.init_lr / (1 + self.iterations)

        # 更新权重平方和累加量 和 梯度
        for key in self.s.keys():
            self.s[key] += np.square(weights[key])
            weights[key] -= self.lr * gradients[key] / np.sqrt(self.s[key] + self.epsilon)

        # 更新迭代次数
        self.iterations += 1

        # 返回梯度
        return weights


class RmsProp(object):
    def __init__(self, weights, gamma=0.9, lr=0.01, epsilon=1e-6, decay=0):
        """

        :param weights: 权重
        :param gamma: 指数
        :param lr: 学习率
        :param epsilon: 平滑数
        :param decay: 学习率衰减
        """
        self.s = _copy_weights_to_zeros(weights)  # 权重平方和累加量
        self.gamma = gamma
        self.iterations = 0  # 迭代次数
        self.lr = self.init_lr = lr
        self.epsilon = epsilon
        self.decay = decay

    def iterate(self, weights, gradients):
        """
        迭代一次
        :param weights: 当前迭代权重
        :param gradients: 当前迭代梯度
        :return:
        """
        # 更新学习率
        self.lr = self.init_lr / (1 + self.iterations)

        # 更新权重平方和累加量 和 梯度
        for key in self.s.keys():
            self.s[key] = self.gamma * self.s[key] + (1 - self.gamma) * np.square(weights[key])
            weights[key] -= self.lr * gradients[key] / np.sqrt(self.s[key] + self.epsilon)

        # 更新迭代次数
        self.iterations += 1

        # 返回梯度
        return weights
