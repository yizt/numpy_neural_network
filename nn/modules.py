# -*- coding: utf-8 -*-
"""
 @File    : modules.py
 @Time    : 2020/4/18 上午8:28
 @Author  : yizuotian
 @Description    :
"""
from typing import List

import numpy as np
from layers import *


class BaseModule(object):
    def forward(self, x):
        pass

    def backward(self, in_gradient):
        pass

    def update_gradient(self, lr):
        pass


class Model(BaseModule):
    """
    网络模型
    """

    def __init__(self, layers: List[BaseModule]):
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)

    def backward(self, in_gradient):
        for l in self.layers:
            in_gradient = l.backward(in_gradient)

    def update_gradient(self, lr):
        for l in self.layers:
            l.update_gradient(lr)


class Linear(BaseModule):
    """
    全连接层
    """

    def __init__(self, in_units, out_units):
        """

        :param in_units: 输入神经元数
        :param out_units: 输出神经元数
        """
        self.weight = np.random.randn(in_units, out_units).astype(np.float64)
        self.bias = np.zeros(out_units).astype(np.float64)
        # 权重和偏置的梯度
        self.g_weight = np.zeros_like(self.weight)
        self.g_bias = np.zeros_like(self.bias)
        # 保存输入feature map,求梯度时需要
        self.in_features = None

    def forward(self, x):
        """

        :param x: [B,in_units]
        :return output: [B,out_units]
        """
        self.in_features = x
        output = fc_forward(x, self.weight, self.bias)
        return output

    def backward(self, in_gradient):
        """
        梯度反向传播
        :param in_gradient: 后一层传递过来的梯度，[B,out_units]
        :return out_gradient: 传递给前一层的梯度，[B,in_units]
        """
        self.g_weight, self.g_bias, out_gradient = fc_backward(in_gradient,
                                                               self.weight,
                                                               self.in_features)
        return out_gradient

    def update_gradient(self, lr):
        """
        更新梯度
        :param lr:
        :return:
        """
        self.weight -= self.g_weight * lr
        self.bias -= self.g_bias * lr


def test():
    x = Linear(10, 10)

    x.g_bias
    print(x.backward)
    x.forward(np.random.randn(3, 10))


if __name__ == '__main__':
    test()
