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
from losses import *


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

        return x

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


def test_linear():
    # 实际的权重和偏置
    W = np.array([[3, 7, 4],
                  [5, 2, 6]])
    b = np.array([2, 9, 3])
    # 产生训练样本
    x_data = np.random.randint(0, 10, 1000).reshape(500, 2)
    y_data = np.dot(x_data, W) + b

    def next_sample(batch_size=1):
        idx = np.random.randint(500)
        return x_data[idx:idx + batch_size], y_data[idx:idx + batch_size]

    m = Model([Linear(2, 3)])
    i = 0
    loss = 1
    while loss > 1e-15:
        x, y_true = next_sample(2)  # 获取当前样本
        # 前向传播
        y = m.forward(x)
        # 反向传播更新梯度
        loss, dy = mean_squared_loss(y, y_true)
        m.backward(dy)
        # 更新梯度
        m.update_gradient(0.01)

        # 更新迭代次数
        i += 1
        if i % 1000 == 0:
            print("\n迭代{}次，当前loss:{}, 当前权重:{},当前偏置{}".format(i, loss,
                                                             m.layers[0].weight,
                                                             m.layers[0].bias))


if __name__ == '__main__':
    test()
