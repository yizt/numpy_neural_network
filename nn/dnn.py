# -*- coding: utf-8 -*-
"""
Created on 2018/9/1 22:22

@author: mick.yi

dnn例子

"""
import numpy as np
from losses import cross_entropy_loss
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
