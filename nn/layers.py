# -*- coding: utf-8 -*-
"""
Created on 2018/8/19 15:03

@author: mick.yi

定义网络层
"""
import numpy as np
import pyximport

pyximport.install()
from clayers import conv_forward


def fc_forward(z, W, b):
    """
    全连接层的前向传播
    :param z: 当前层的输出,形状 (N,ln)
    :param W: 当前层的权重
    :param b: 当前层的偏置
    :return: 下一层的输出
    """
    return np.dot(z, W) + b


def fc_backward(next_dz, W, z):
    """
    全连接层的反向传播
    :param next_dz: 下一层的梯度
    :param W: 当前层的权重
    :param z: 当前层的输出
    :return:
    """
    N = z.shape[0]
    dz = np.dot(next_dz, W.T)  # 当前层的梯度
    dw = np.dot(z.T, next_dz)  # 当前层权重的梯度
    db = np.sum(next_dz, axis=0)  # 当前层偏置的梯度, N个样本的梯度求和
    return dw / N, db / N, dz


def _single_channel_conv(z, K, b=0, padding=(0, 0), strides=(1, 1)):
    """
    当通道卷积操作
    :param z: 卷积层矩阵
    :param K: 卷积核
    :param b: 偏置
    :param padding: padding
    :param strides: 步长
    :return: 卷积结果
    """
    padding_z = np.lib.pad(z, ((padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    height, width = padding_z.shape
    k1, k2 = K.shape
    assert (height - k1) % strides[0] == 0, '步长不为1时，步长必须刚好能够被整除'
    assert (width - k2) % strides[1] == 0, '步长不为1时，步长必须刚好能够被整除'
    conv_z = np.zeros((1 + (height - k1) // strides[0], 1 + (width - k2) // strides[1]))
    for h in np.arange(height - k1 + 1)[::strides[0]]:
        for w in np.arange(width - k2 + 1)[::strides[1]]:
            conv_z[h // strides[0], w // strides[1]] = np.sum(padding_z[h:h + k1, w:w + k2] * K)
    return conv_z + b


def _remove_padding(z, padding):
    """
    移除padding
    :param z: (N,C,H,W)
    :param paddings: (p1,p2)
    :return:
    """
    if padding[0] > 0 and padding[1] > 0:
        return z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
    elif padding[0] > 0:
        return z[:, :, padding[0]:-padding[0], :]
    elif padding[1] > 0:
        return z[:, :, :, padding[1]:-padding[1]]
    else:
        return z


def conv_forward_bak(z, K, b, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param K: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
    :param b: 偏置,形状(D,)
    :param padding: padding
    :param strides: 步长
    :return: 卷积结果
    """
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)
    N, _, height, width = padding_z.shape
    C, D, k1, k2 = K.shape
    assert (height - k1) % strides[0] == 0, '步长不为1时，步长必须刚好能够被整除'
    assert (width - k2) % strides[1] == 0, '步长不为1时，步长必须刚好能够被整除'
    conv_z = np.zeros((N, D, 1 + (height - k1) // strides[0], 1 + (width - k2) // strides[1]))
    for n in np.arange(N):
        for d in np.arange(D):
            for h in np.arange(height - k1 + 1)[::strides[0]]:
                for w in np.arange(width - k2 + 1)[::strides[1]]:
                    conv_z[n, d, h // strides[0], w // strides[1]] = np.sum(
                        padding_z[n, :, h:h + k1, w:w + k2] * K[:, d]) + b[d]
    return conv_z


def _insert_zeros(dz, strides):
    """
    想多维数组最后两位，每个行列之间增加指定的个数的零填充
    :param dz: (N,D,H,W),H,W为卷积输出层的高度和宽度
    :param strides: 步长
    :return:
    """
    _, _, H, W = dz.shape
    pz = dz
    if strides[0] > 1:
        for h in np.arange(H - 1, 0, -1):
            for o in np.arange(strides[0] - 1):
                pz = np.insert(pz, h, 0, axis=2)
    if strides[1] > 1:
        for w in np.arange(W - 1, 0, -1):
            for o in np.arange(strides[1] - 1):
                pz = np.insert(pz, w, 0, axis=3)
    return pz


def conv_backward(next_dz, K, z, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积层的反向过程
    :param next_dz: 卷积输出层的梯度,(N,D,H,W),H,W为卷积输出层的高度和宽度
    :param K: 当前层卷积核，(C,D,k1,k2)
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param padding: padding
    :param strides: 步长
    :return:
    """
    N, C, H, W = z.shape
    C, D, k1, k2 = K.shape

    # 卷积核梯度
    # dK = np.zeros((C, D, k1, k2))
    padding_next_dz = _insert_zeros(next_dz, strides)

    # 卷积核高度和宽度翻转180度
    flip_K = np.flip(K, (2, 3))
    # 交换C,D为D,C；D变为输入通道数了，C变为输出通道数了
    swap_flip_K = np.swapaxes(flip_K, 0, 1)
    # 增加高度和宽度0填充
    ppadding_next_dz = np.lib.pad(padding_next_dz, ((0, 0), (0, 0), (k1 - 1, k1 - 1), (k2 - 1, k2 - 1)), 'constant',
                                  constant_values=0)
    dz = conv_forward(ppadding_next_dz,
                      swap_flip_K,
                      np.zeros((C,), dtype=np.float))

    # 求卷积和的梯度dK
    swap_z = np.swapaxes(z, 0, 1)  # 变为(C,N,H,W)与
    dK = conv_forward(swap_z, padding_next_dz, np.zeros((D,), dtype=np.float))

    # 偏置的梯度
    db = np.sum(np.sum(np.sum(next_dz, axis=-1), axis=-1), axis=0)  # 在高度、宽度上相加；批量大小上相加

    # 把padding减掉
    dz = _remove_padding(dz, padding)  # dz[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]

    return dK / N, db / N, dz


def max_pooling_forward_bak(z, pooling, strides=(2, 2), padding=(0, 0)):
    """
    最大池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = z.shape
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)

    # 输出的高度和宽度
    out_h = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
    out_w = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1

    pool_z = np.zeros((N, C, out_h, out_w), dtype=np.float32)

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_z[n, c, i, j] = np.max(padding_z[n, c,
                                                strides[0] * i:strides[0] * i + pooling[0],
                                                strides[1] * j:strides[1] * j + pooling[1]])
    return pool_z


def max_pooling_backward_bak(next_dz, z, pooling, strides=(2, 2), padding=(0, 0)):
    """
    最大池化反向过程
    :param next_dz：损失函数关于最大池化输出的损失
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = z.shape
    _, _, out_h, out_w = next_dz.shape
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)
    # 零填充后的梯度
    padding_dz = np.zeros_like(padding_z)

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    # 找到最大值的那个元素坐标，将梯度传给这个坐标
                    flat_idx = np.argmax(padding_z[n, c,
                                         strides[0] * i:strides[0] * i + pooling[0],
                                         strides[1] * j:strides[1] * j + pooling[1]])
                    h_idx = strides[0] * i + flat_idx // pooling[1]
                    w_idx = strides[1] * j + flat_idx % pooling[1]
                    padding_dz[n, c, h_idx, w_idx] += next_dz[n, c, i, j]
    # 返回时剔除零填充
    return _remove_padding(padding_dz, padding)  # padding_z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]


def avg_pooling_forward(z, pooling, strides=(2, 2), padding=(0, 0)):
    """
    平均池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = z.shape
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)

    # 输出的高度和宽度
    out_h = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
    out_w = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1

    pool_z = np.zeros((N, C, out_h, out_w), dtype=np.float32)

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_z[n, c, i, j] = np.mean(padding_z[n, c,
                                                 strides[0] * i:strides[0] * i + pooling[0],
                                                 strides[1] * j:strides[1] * j + pooling[1]])
    return pool_z


def avg_pooling_backward(next_dz, z, pooling, strides=(2, 2), padding=(0, 0)):
    """
    平均池化反向过程
    :param next_dz：损失函数关于最大池化输出的损失
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = z.shape
    _, _, out_h, out_w = next_dz.shape
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)
    # 零填充后的梯度
    padding_dz = np.zeros_like(padding_z)

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    # 每个神经元均分梯度
                    padding_dz[n, c,
                    strides[0] * i:strides[0] * i + pooling[0],
                    strides[1] * j:strides[1] * j + pooling[1]] += next_dz[n, c, i, j] / (pooling[0] * pooling[1])
    # 返回时剔除零填充
    return _remove_padding(padding_dz, padding)  # padding_z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]


def global_max_pooling_forward(z):
    """
    全局最大池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :return:
    """
    return np.max(np.max(z, axis=-1), -1)


def global_max_pooling_backward(next_dz, z):
    """
    全局最大池化反向过程
    :param next_dz: 全局最大池化梯度，形状(N,C)
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :return:
    """
    N, C, H, W = z.shape
    dz = np.zeros_like(z)
    for n in np.arange(N):
        for c in np.arange(C):
            # 找到最大值所在坐标，梯度传给这个坐标
            idx = np.argmax(z[n, c, :, :])
            h_idx = idx // W
            w_idx = idx % W
            dz[n, c, h_idx, w_idx] = next_dz[n, c]
    return dz


def global_avg_pooling_forward(z):
    """
    全局平均池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :return:
    """
    return np.mean(np.mean(z, axis=-1), axis=-1)


def global_avg_pooling_backward(next_dz, z):
    """
    全局平均池化反向过程
    :param next_dz: 全局最大池化梯度，形状(N,C)
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :return:
    """
    N, C, H, W = z.shape
    dz = np.zeros_like(z)
    for n in np.arange(N):
        for c in np.arange(C):
            # 梯度平分给相关神经元
            dz[n, c, :, :] += next_dz[n, c] / (H * W)
    return dz


def flatten_forward(z):
    """
    将多维数组打平，前向传播
    :param z: 多维数组,形状(N,d1,d2,..)
    :return:
    """
    N = z.shape[0]
    return np.reshape(z, (N, -1))


def flatten_backward(next_dz, z):
    """
    打平层反向传播
    :param next_dz:
    :param z:
    :return:
    """
    return np.reshape(next_dz, z.shape)


def main():
    z = np.ones((5, 5))
    k = np.ones((3, 3))
    b = 3
    # print(_single_channel_conv(z, k,padding=(1,1)))
    # print(_single_channel_conv(z, k, strides=(2, 2)))
    assert _single_channel_conv(z, k).shape == (3, 3)
    assert _single_channel_conv(z, k, padding=(1, 1)).shape == (5, 5)
    assert _single_channel_conv(z, k, strides=(2, 2)).shape == (2, 2)
    assert _single_channel_conv(z, k, strides=(2, 2), padding=(1, 1)).shape == (3, 3)
    assert _single_channel_conv(z, k, strides=(2, 2), padding=(1, 0)).shape == (3, 2)
    assert _single_channel_conv(z, k, strides=(2, 1), padding=(1, 1)).shape == (3, 5)

    dz = np.ones((1, 1, 3, 3))
    assert _insert_zeros(dz, (1, 1)).shape == (1, 1, 3, 3)
    print(_insert_zeros(dz, (3, 2)))
    assert _insert_zeros(dz, (1, 2)).shape == (1, 1, 3, 5)
    assert _insert_zeros(dz, (2, 2)).shape == (1, 1, 5, 5)
    assert _insert_zeros(dz, (2, 4)).shape == (1, 1, 5, 9)

    z = np.ones((8, 16, 5, 5))
    k = np.ones((16, 32, 3, 3))
    b = np.ones((32))
    assert conv_forward(z, k, b).shape == (8, 32, 3, 3)
    print(conv_forward(z, k, b)[0, 0])

    print(np.argmax(np.array([[1, 2], [3, 4]])))


def test_conv():
    # 测试卷积
    z = np.random.randn(3, 3, 28, 28).astype(np.float)
    K = np.random.randn(3, 4, 3, 3).astype(np.float) * 1e-3
    b = np.zeros(4).astype(np.float)

    next_z = conv_forward(z, K, b)
    y_true = np.ones_like(next_z)

    from nn.losses import mean_squared_loss
    for i in range(10000):
        # 前向
        next_z = conv_forward(z, K, b)
        # 反向
        loss, dy = mean_squared_loss(next_z, y_true)
        dK, db, _ = conv_backward(dy, K, z)
        K -= 0.001 * dK
        b -= 0.001 * db

        if i % 10 == 0:
            print("i:{},loss:{},mindy:{},maxdy:{}".format(i, loss, np.mean(dy), np.max(dy)))

        if np.allclose(y_true, next_z):
            print("yes")
            break


def test_conv_and_max_pooling():
    # 测试卷积和最大池化
    z = np.random.randn(3, 3, 28, 28).astype(np.float)
    K = np.random.randn(3, 4, 3, 3).astype(np.float) * 1e-3
    b = np.zeros(4).astype(np.float)

    next_z = conv_forward(z, K, b)
    y_pred = max_pooling_forward_bak(next_z, pooling=(2, 2))
    y_true = np.ones_like(y_pred)

    from nn.losses import mean_squared_loss
    for i in range(10000):
        # 前向
        next_z = conv_forward(z, K, b)
        y_pred = max_pooling_forward_bak(next_z, pooling=(2, 2))
        # 反向
        loss, dy = mean_squared_loss(y_pred, y_true)
        next_dz = max_pooling_backward_bak(dy, next_z, pooling=(2, 2))
        dK, db, _ = conv_backward(next_dz, K, z)
        K -= 0.001 * dK
        b -= 0.001 * db

        if i % 10 == 0:
            print("i:{},loss:{},mindy:{},maxdy:{}".format(i, loss, np.mean(dy), np.max(dy)))

        if np.allclose(y_true, y_pred):
            print("yes")
            break


if __name__ == "__main__":
    # main()
    test_conv()
