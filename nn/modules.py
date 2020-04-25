# -*- coding: utf-8 -*-
"""
 @File    : modules.py
 @Time    : 2020/4/18 上午8:28
 @Author  : yizuotian
 @Description    :
"""
from typing import List

from activations import *
from layers import fc_forward, fc_backward, global_avg_pooling_forward, flatten_forward, flatten_backward
from layers_v2 import conv_forward, conv_backward, max_pooling_forward, max_pooling_backward, \
    global_avg_pooling_backward
from losses import *
from optimizers import *


# pyximport.install()
# from clayers import *


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

    def load_weights(self, weights):
        """
        加载权重
        :param weights:
        :return:
        """
        for key in self.weights.keys():
            self.weights[key] = weights[key]


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

    def load_weights(self, weights):
        """
        加载模型权重
        :param weights:
        :return:
        """
        # 逐层加载权重
        for l in self.layers:
            l.load_weights(weights)


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
        # 权重参数
        weight = np.random.randn(in_units, out_units) * np.sqrt(2 / in_units)
        bias = np.zeros(out_units)
        # 权重对应的梯度
        g_weight = np.zeros_like(weight)
        g_bias = np.zeros_like(bias)
        # 权重和梯度的字典
        self.weights = {"{}_weight".format(self.name): weight,
                        "{}_bias".format(self.name): bias}

        self.gradients = {"{}_weight".format(self.name): g_weight,
                          "{}_bias".format(self.name): g_bias}

    @property
    def weight(self):
        return self.weights["{}_weight".format(self.name)]

    @property
    def bias(self):
        return self.weights["{}_bias".format(self.name)]

    def set_gradient(self, name, gradient):
        """
        更新梯度
        :param name: weight 或 bias 中一个
        :param gradient:
        :return:
        """
        self.gradients["{}_{}".format(self.name, name)] = gradient

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
        g_weight, g_bias, out_gradient = fc_backward(in_gradient,
                                                     self.weight,
                                                     self.in_features)
        self.set_gradient('weight', g_weight)
        self.set_gradient('bias', g_bias)
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
        fan_in = in_filters * kernel[0] * kernel[1]  # 输入参数量
        fan_out = out_filters * kernel[0] * kernel[1]  # 输入参数量
        weight = np.random.randn(in_filters,
                                 out_filters,
                                 *kernel) * np.sqrt(2 / (fan_in + fan_out))
        bias = np.zeros(out_filters)
        # 梯度
        g_weight = np.zeros_like(weight)
        g_bias = np.zeros_like(bias)
        # 权重和梯度的字典
        self.weights = {"{}_weight".format(self.name): weight,
                        "{}_bias".format(self.name): bias}

        self.gradients = {"{}_weight".format(self.name): g_weight,
                          "{}_bias".format(self.name): g_bias}

    @property
    def weight(self):
        return self.weights["{}_weight".format(self.name)]

    @property
    def bias(self):
        return self.weights["{}_bias".format(self.name)]

    def set_gradient(self, name, gradient):
        """
        更新梯度
        :param name: weight 或 bias 中一个
        :param gradient:
        :return:
        """
        self.gradients["{}_{}".format(self.name, name)] = gradient

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
        g_weight, g_bias, out_gradient = conv_backward(in_gradient,
                                                       self.weight,
                                                       self.in_features,
                                                       self.padding, self.stride)
        self.set_gradient('weight', g_weight)
        self.set_gradient('bias', g_bias)
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
    x_data = np.random.randn(500, 2)
    y_data = np.dot(x_data, W) + b

    def next_sample(batch_size=1):
        idx = np.random.randint(500)
        return x_data[idx:idx + batch_size], y_data[idx:idx + batch_size]

    fc_layer = Linear(2, 3, name='fc1')
    # fc_layer.weights['fc1_weight'] *= 1e-2  # 单层权重初始化要小
    m = Model([fc_layer])
    sgd = SGD(m.weights, lr=1e-3)
    i = 0
    loss = 1
    while loss > 1e-15:
        x, y_true = next_sample(4)  # 获取当前样本
        # 前向传播
        y = m.forward(x)
        # 反向传播更新梯度
        loss, dy = mean_squared_loss(y, y_true)
        m.backward(dy)
        # 更新梯度
        sgd.iterate(m)

        # 更新迭代次数
        i += 1
        if i % 10000 == 0:
            print("y_pred：{},y_true:{}".format(y, y_true))
            print("\n迭代{}次，当前loss:{}, 当前权重:{},当前偏置{},梯度:{}".format(i, loss,
                                                                   m.layers[0].weight,
                                                                   m.layers[0].bias,
                                                                   m.layers[0].gradients))
            # print(m.weights)

    print('迭代{}次,当前权重:{} '.format(i, m.layers[0].weights))


if __name__ == '__main__':
    test_linear()
