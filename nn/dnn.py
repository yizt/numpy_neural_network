# -*- coding: utf-8 -*-
"""
Created on 2018/9/1 22:22

@author: mick.yi

dnn例子

"""
import numpy as np
from losses import cross_entropy_loss, mean_squared_loss
from layers import fc_forward, fc_backward
from activations import relu_forward, relu_backward


class Mnist(object):
    def __init__(self):
        weight_scale = 1e-3
        self.weights = {}
        self.weights["W1"] = weight_scale * np.random.randn(28 * 28, 256)
        self.weights["b1"] = np.zeros(256)

        self.weights["W2"] = weight_scale * np.random.randn(256, 256)
        self.weights["b2"] = np.zeros(256)

        self.weights["W3"] = weight_scale * np.random.randn(256, 10)
        self.weights["b3"] = np.zeros(10)

        # 存放神经元的值
        self.nurons = {}

        # 存放梯度
        self.gradients = {}

    def forward(self, train_data):
        self.nurons["z2"] = fc_forward(train_data, self.weights["W1"], self.weights["b1"])
        self.nurons["z2_relu"] = relu_forward(self.nurons["z2"])
        self.nurons["z3"] = fc_forward(self.nurons["z2_relu"], self.weights["W2"], self.weights["b2"])
        self.nurons["z3_relu"] = relu_forward(self.nurons["z3"])
        self.nurons["y"] = fc_forward(self.nurons["z3_relu"], self.weights["W3"], self.weights["b3"])
        return self.nurons["y"]

    def backward(self, train_data, y_true):
        loss, self.gradients["y"] = cross_entropy_loss(self.nurons["y"], y_true)
        self.gradients["W3"], self.gradients["b3"], self.gradients["z3_relu"] = fc_backward(self.gradients["y"],
                                                                                            self.weights["W3"],
                                                                                            self.nurons["z3_relu"])
        self.gradients["z3"] = relu_backward(self.gradients["z3_relu"], self.nurons["z3"])
        self.gradients["W2"], self.gradients["b2"], self.gradients["z2_relu"] = fc_backward(self.gradients["z3"],
                                                                                            self.weights["W2"],
                                                                                            self.nurons["z2_relu"])
        self.gradients["z2"] = relu_backward(self.gradients["z2_relu"], self.nurons["z2"])
        self.gradients["W1"], self.gradients["b1"], _ = fc_backward(self.gradients["z2"],
                                                                    self.weights["W1"],
                                                                    train_data)
        return loss

    def get_accuracy(self, train_data, y_true):
        score = self.forward(train_data)
        acc = np.mean(np.argmax(score, axis=1) == np.argmax(y_true, axis=1))
        return acc


class LinearRegression(object):
    """
    模拟线性回归的例子 y=Wx+b
    """

    def __init__(self):
        # 实际的权重和偏置
        self.W = np.array([[3, 7, 4],
                           [5, 2, 6]])
        self.b = np.array([2, 9, 3])
        # 产生训练样本
        self.x_data = np.random.randint(0, 10, 1000).reshape(500, 2)
        self.y_data = np.dot(self.x_data, self.W) + self.b

    def next_sample(self, batch_size=1):
        """
        随机产生下一批样本
        :param batch_size:
        :return:
        """
        idx = np.random.randint(500 - batch_size)
        return self.x_data[idx:idx + batch_size], self.y_data[idx:idx + batch_size]

    def train(self):
        # 随机初始化参数
        W1 = np.random.randn(2, 3)
        b1 = np.zeros([3])
        loss = 100.0
        lr = 0.01
        i = 0

        while loss > 1e-15:
            x, y_true = self.next_sample(2)  # 获取当前样本
            # 前向传播
            y = fc_forward(x, W1, b1)
            # 反向传播更新梯度
            loss, dy = mean_squared_loss(y, y_true)
            dw, db, _ = fc_backward(dy, self.W, x)

            # 在一个batch中梯度取均值
            # print(dw)

            # 更新梯度
            W1 -= lr * dw
            b1 -= lr * db

            # 更新迭代次数
            i += 1
            if i % 1000 == 0:
                print("\n迭代{}次，当前loss:{}, 当前权重:{},当前偏置{}".format(i, loss, W1, b1))

                # 打印最终结果
        print("\n迭代{}次，当前loss:{}, 当前权重:{},当前偏置{}".format(i, loss, W1, b1))

        return W1, b1
