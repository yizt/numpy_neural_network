# -*- coding: utf-8 -*-
"""
 @File    : modules.py
 @Time    : 2020/4/18 上午8:28
 @Author  : yizuotian
 @Description    :
"""
from typing import List

from activations import *
from layers import *
from losses import *

pyximport.install()
from clayers import *


class BaseModule(object):
    def __init__(self, name=''):
        """

        :param name: 层名
        """
        self.name = name
        self.weights = dict()  # 权重参数字典
        self.gradients = dict()  # 梯度字典
        self.in_features = None  # 输入的feature map

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

    def __init__(self, layers: List[BaseModule], **kwargs):
        super(Model, self).__init__(**kwargs)
        self.layers = layers
        # 收集所有权重和梯度
        for l in self.layers:
            self.weights.update(l.weights)
            self.gradients.update(l.gradients)

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
            # print('forward layer:{},feature:{}'.format(l.name, np.max(x)))
        # 网络结果返回
        return x

    def backward(self, in_gradient):
        # 反向传播
        for l in self.layers[::-1]:
            in_gradient = l.backward(in_gradient)
            # print('backward layer:{},gradient:{}'.format(l.name, np.max(in_gradient)))

    def update_gradient(self, lr):
        for l in self.layers:
            l.update_gradient(lr)


class Linear(BaseModule):
    """
    全连接层
    """

    def __init__(self, in_units, out_units, **kwargs):
        """

        :param in_units: 输入神经元数
        :param out_units: 输出神经元数
        """
        super(Linear, self).__init__(**kwargs)
        self.weight = np.random.randn(in_units, out_units).astype(np.float) * np.sqrt(2 / in_units / out_units)
        self.bias = np.zeros(out_units).astype(np.float)
        # 权重对应的梯度
        self.g_weight = np.zeros_like(self.weight)
        self.g_bias = np.zeros_like(self.bias)
        # 权重和梯度的字典
        self.weights = {"{}_weight".format(self.name): self.weight,
                        "{}_bias".format(self.name): self.bias}

        self.gradients = {"{}_weight".format(self.name): self.weight,
                          "{}_bias".format(self.name): self.bias}

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


class Conv2D(BaseModule):
    """
    2D卷积层
    """

    def __init__(self, in_filters, out_filters, kernel=(3, 3), padding=(1, 1), stride=(1, 1), **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        # 权重参数
        self.weight = np.random.randn(in_filters, out_filters, *kernel).astype(np.float) * np.sqrt(2 / in_filters)
        self.bias = np.zeros(out_filters).astype(np.float)
        # 梯度
        self.g_weight = np.zeros_like(self.weight)
        self.g_bias = np.zeros_like(self.bias)
        # 权重和梯度的字典
        self.weights = {"{}_weight".format(self.name): self.weight,
                        "{}_bias".format(self.name): self.bias}

        self.gradients = {"{}_weight".format(self.name): self.weight,
                          "{}_bias".format(self.name): self.bias}

    def forward(self, x):
        """

        :param x: [B,in_filters,H,W]
        :return output:  [B,out_filters,H,W]
        """
        self.in_features = x
        output = conv_forward(x, self.weight, self.bias, self.padding, self.stride)
        return output

    def backward(self, in_gradient):
        """

        :param in_gradient: 后一层传递过来的梯度，[B,out_filters,H,W]
        :return out_gradient: 传递给前一层的梯度，[B,in_filters,H,W]
        """
        self.g_weight, self.g_bias, out_gradient = conv_backward(in_gradient,
                                                                 self.weight,
                                                                 self.in_features,
                                                                 self.padding, self.stride)
        return out_gradient

    def update_gradient(self, lr):
        self.weight -= self.g_weight * lr
        self.bias -= self.g_bias * lr


class ReLU(BaseModule):
    def __init__(self, **kwargs):
        super(ReLU, self).__init__(**kwargs)

    def forward(self, x):
        self.in_features = x
        return relu_forward(x)

    def backward(self, in_gradient):
        """

        :param in_gradient: 后一层传递过来的梯度
        :return out_gradient: 传递给前一层的梯度
        """
        out_gradient = relu_backward(in_gradient, self.in_features)
        return out_gradient


class MaxPooling2D(BaseModule):
    """
    最大池化层
    """

    def __init__(self, kernel=(2, 2), stride=(2, 2), padding=(0, 0), **kwargs):
        """

        :param kernel: 池化尺寸
        :param stride: 步长
        :param padding: padding
        :param kwargs:
        """
        super(MaxPooling2D, self).__init__(**kwargs)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """

        :param x: [B,C,H,W]
        :return output : [B,C,H',W']
        """
        self.in_features = x
        output = max_pooling_forward(x, self.kernel, self.stride, self.padding)
        return output

    def backward(self, in_gradient):
        """

        :param in_gradient: 后一层传递过来的梯度
        :return out_gradient: 传递给前一层的梯度
        """
        out_gradient = max_pooling_backward(in_gradient,
                                            self.in_features,
                                            self.kernel,
                                            self.stride,
                                            self.padding)
        return out_gradient


class GlobalAvgPooling2D(BaseModule):
    """
    全局平均池化
    """

    def __init__(self, **kwargs):
        super(GlobalAvgPooling2D, self).__init__(**kwargs)

    def forward(self, x):
        """

        :param x: [B,C,H,W]
        :return output : [B,C,H',W']
        """
        self.in_features = x
        output = global_avg_pooling_forward(x)
        return output

    def backward(self, in_gradient):
        """

        :param in_gradient: 后一层传递过来的梯度
        :return out_gradient: 传递给前一层的梯度
        """
        out_gradient = global_avg_pooling_backward(in_gradient,
                                                   self.in_features)
        return out_gradient


class Flatten(BaseModule):
    """
    打平层
    """

    def __init__(self, **kwargs):
        super(Flatten, self).__init__(**kwargs)

    def forward(self, x):
        self.in_features = x
        return flatten_forward(x)

    def backward(self, in_gradient):
        """

        :param in_gradient: 后一层传递过来的梯度
        :return out_gradient: 传递给前一层的梯度
        """
        out_gradient = flatten_backward(in_gradient, self.in_features)
        return out_gradient


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
    print(m.layers[0].weights)


if __name__ == '__main__':
    test_linear()
