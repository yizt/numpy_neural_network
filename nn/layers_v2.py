# -*- coding: utf-8 -*-
"""
 @File    : layers_v2.py
 @Time    : 2020/4/25 上午9:15
 @Author  : yizuotian
 @Description    : v2版前向、反向计算；解决卷积计算速度慢的问题
"""
import time

import numpy as np
import pyximport

from layers import _insert_zeros, _remove_padding

pyximport.install()
from clayers_v2 import conv_forward as c_conv_forward


def _single_channel_conv_v1(z, K, b=0, padding=(0, 0)):
    """
    当通道卷积操作
    :param z: 卷积层矩阵
    :param K: 卷积核
    :param b: 偏置
    :param padding: padding
    :return: 卷积结果
    """
    padding_z = np.lib.pad(z, ((padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    height, width = padding_z.shape
    k1, k2 = K.shape
    conv_z = np.zeros((1 + (height - k1), 1 + (width - k2)))
    for h in np.arange(height - k1 + 1):
        for w in np.arange(width - k2 + 1):
            conv_z[h, w] = np.sum(padding_z[h:h + k1, w:w + k2] * K)
    return conv_z + b


def _single_channel_conv(z, K, b=0, padding=(0, 0)):
    """
    当通道卷积操作
    :param z: 卷积层矩阵
    :param K: 卷积核
    :param b: 偏置
    :param padding: padding
    :return: 卷积结果
    """
    padding_z = np.lib.pad(z, ((padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    height, width = padding_z.shape
    k1, k2 = K.shape
    oh, ow = (1 + (height - k1), 1 + (width - k2))  # 输出的高度和宽度
    conv_z = np.zeros((1 + (height - k1), 1 + (width - k2)))
    # 遍历卷积比遍历特征高效
    for i in range(k1):
        for j in range(k2):
            conv_z += padding_z[i:i + oh, j:j + ow] * K[i, j]

    return conv_z + b


def conv_forward_v1(z, K, b, padding=(0, 0)):
    """
    多通道卷积前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param K: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
    :param b: 偏置,形状(D,)
    :param padding: padding
    :return: conv_z: 卷积结果[N,D,oH,oW]
    """
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)
    N, _, height, width = padding_z.shape
    C, D, k1, k2 = K.shape
    oh, ow = (1 + (height - k1), 1 + (width - k2))  # 输出的高度和宽度
    conv_z = np.zeros((N, D, oh, ow))
    for n in np.arange(N):
        for d in np.arange(D):
            for h in np.arange(oh):
                for w in np.arange(oh):
                    conv_z[n, d, h, w] = np.sum(
                        padding_z[n, :, h:h + k1, w:w + k2] * K[:, d]) + b[d]
    return conv_z


def _conv_forward_old(z, K, b, padding=(0, 0)):
    """
    占用太多内存反而慢
    多通道卷积前向过程;
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param K: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
    :param b: 偏置,形状(D,)
    :param padding: padding
    :return: conv_z: 卷积结果[N,D,oH,oW]
    """
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)
    N, _, height, width = padding_z.shape
    C, D, k1, k2 = K.shape
    oh, ow = (1 + (height - k1), 1 + (width - k2))  # 输出的高度和宽度

    # 扩维
    padding_z = padding_z[:, :, np.newaxis, :, :]  # 扩维[N,C,1,H,W] 与K [C,D,K1,K2] 可以广播
    K = K[:, :, :, :, np.newaxis, np.newaxis]
    conv_z = np.zeros((N, C, D, oh, ow))

    # 批量卷积
    for i in range(k1):
        for j in range(k2):
            # [N,C,1,oh,ow]*[C,D,1,1] =>[N,C,D,oh,ow]
            conv_z += padding_z[:, :, :, i:i + oh, j:j + ow] * K[:, :, i, j]

    conv_z = np.sum(conv_z, axis=1)  # [N, C, D, oh, ow] => [N, D, oh, ow]
    # 增加偏置 [N, D, oh, ow]+[D, 1, 1]
    conv_z += b[:, np.newaxis, np.newaxis]
    return conv_z


def _conv_forward(z, K, b, padding=(0, 0)):
    """
    多通道卷积前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param K: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
    :param b: 偏置,形状(D,)
    :param padding: padding
    :return: conv_z: 卷积结果[N,D,oH,oW]
    """
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)
    N, _, height, width = padding_z.shape
    C, D, k1, k2 = K.shape
    oh, ow = (1 + (height - k1), 1 + (width - k2))  # 输出的高度和宽度

    # 扩维
    padding_z = padding_z[:, :, np.newaxis, :, :]  # 扩维[N,C,1,H,W] 与K [C,D,K1,K2] 可以广播
    conv_z = np.zeros((N, D, oh, ow))

    # 批量卷积
    if k1 * k2 < oh * ow * 10:
        K = K[:, :, :, :, np.newaxis, np.newaxis]
        for c in range(C):
            for i in range(k1):
                for j in range(k2):
                    # [N,1,oh,ow]*[D,1,1] =>[N,D,oh,ow]
                    conv_z += padding_z[:, c, :, i:i + oh, j:j + ow] * K[c, :, i, j]
    else:  # 大卷积核，遍历空间更高效
        # print('大卷积核，遍历空间更高效')
        for c in range(C):
            for h in range(oh):
                for w in range(ow):
                    # [N,1,k1,k2]*[D,k1,k2] =>[N,D,k1,k2] => [N,D]
                    conv_z[:, :, h, w] += np.sum(padding_z[:, c, :, h:h + k1, w:w + k2] * K[c], axis=(2, 3))

    # 增加偏置 [N, D, oh, ow]+[D, 1, 1]
    conv_z += b[:, np.newaxis, np.newaxis]
    return conv_z


def conv_forward(z, K, b, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param K: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
    :param b: 偏置,形状(D,)
    :param padding: padding
    :param strides: 步长
    :return: conv_z: 卷积结果[N,D,oH,oW]
    """
    # 长宽方向步长
    sh, sw = strides
    # origin_conv_z = _conv_forward(z, K, b, padding)
    origin_conv_z = c_conv_forward(z, K, b, padding)  # 使用cython
    # 步长为1时的输出卷积尺寸
    N, D, oh, ow = origin_conv_z.shape
    if sh * sw == 1:
        return origin_conv_z
    # 高度方向步长大于1
    elif sw == 1:
        conv_z = np.zeros((N, D, oh // sh, ow))
        for i in range(oh // sh):
            conv_z[:, :, i, :] = origin_conv_z[:, :, i * sh, :]
        return conv_z
    # 宽度方向步长大于1
    elif sh == 1:
        conv_z = np.zeros((N, D, oh, ow // sw))
        for j in range(ow // sw):
            conv_z[:, :, :, j] = origin_conv_z[:, :, :, j * sw]
        return conv_z
    # 高度宽度方向步长都大于1
    else:
        conv_z = np.zeros((N, D, oh // sh, ow // sw))
        for i in range(oh // sh):
            for j in range(ow // sw):
                conv_z[:, :, i, j] = origin_conv_z[:, :, i * sh, j * sw]
        return conv_z


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

    # 偏置的梯度,[N,D,H,W]=>[D]
    db = np.sum(next_dz, axis=(0, 2, 3))  # 在高度、宽度上相加；批量大小上相加

    # 把padding减掉
    dz = _remove_padding(dz, padding)  # dz[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]

    return dK / N, db / N, dz


def max_pooling_forward(z, pooling, strides=(2, 2), padding=(0, 0)):
    """
    最大池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = z.shape
    pad_h, pad_w = padding
    sh, sw = strides
    kh, kw = pooling
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant',
                           constant_values=0)

    # 输出的高度和宽度
    out_h = (H + 2 * pad_h - kh) // sh + 1
    out_w = (W + 2 * pad_w - kw) // sw + 1

    pool_z = np.zeros((N, C, out_h, out_w), dtype=np.float)

    for i in np.arange(out_h):
        for j in np.arange(out_w):
            pool_z[:, :, i, j] = np.max(padding_z[:, :, sh * i:sh * i + kh, sw * j:sw * j + kw],
                                        axis=(2, 3))
    return pool_z


def max_pooling_backward(next_dz, z, pooling, strides=(2, 2), padding=(0, 0)):
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
    pad_h, pad_w = padding
    sh, sw = strides
    kh, kw = pooling
    _, _, out_h, out_w = next_dz.shape
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant',
                           constant_values=0)
    # 零填充后的梯度
    padding_dz = np.zeros_like(padding_z)
    zeros = np.zeros((N, C, sh, sw))
    for i in np.arange(out_h):
        for j in np.arange(out_w):
            # 找到最大值的那个元素坐标，将梯度传给这个坐标
            cur_padding_z = padding_z[:, :, sh * i:sh * i + kh, sw * j:sw * j + kw]
            cur_padding_dz = padding_dz[:, :, sh * i:sh * i + kh, sw * j:sw * j + kw]
            max_val = np.max(cur_padding_z, axis=(2, 3))  # [N,C]
            cur_padding_dz += np.where(cur_padding_z == max_val[:, :, np.newaxis, np.newaxis],
                                       next_dz[:, :, i:i + 1, j:j + 1],
                                       zeros)
    # 返回时剔除零填充
    return _remove_padding(padding_dz, padding)  # padding_z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]


def global_avg_pooling_backward(next_dz, z):
    """
    全局平均池化反向过程
    :param next_dz: 全局最大池化梯度，形状(N,C)
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :return dz: [N,C,H,W]
    """
    _, _, H, W = z.shape
    dz = np.zeros_like(z)
    # 梯度平分给相关神经元
    dz += next_dz[:, :, np.newaxis, np.newaxis] / (H * W)
    return dz


def test_single_conv():
    """
    两个卷积结果一样，速度相差百倍以上
    :return:
    """
    z = np.random.randn(224, 224)
    K = np.random.randn(3, 3)

    s = time.time()
    o1 = _single_channel_conv_v1(z, K)
    print("v1 耗时:{}".format(time.time() - s))
    s = time.time()
    o2 = _single_channel_conv(z, K)
    print("v2 耗时:{}".format(time.time() - s))

    print(np.allclose(o1, o2))


def test_conv():
    """
    两个卷积结果一样，速度相差几十倍
    :return:
    """
    z = np.random.randn(4, 3, 112, 112)
    K = np.random.randn(3, 32, 3, 3)
    b = np.random.randn(32)
    s = time.time()
    o1 = conv_forward_v1(z, K, b)
    print("v1 耗时:{}".format(time.time() - s))
    s = time.time()
    o2 = _conv_forward(z, K, b)
    print("v2 耗时:{}".format(time.time() - s))

    import pyximport
    pyximport.install()
    from clayers_v2 import conv_forward as c_conv_forward
    s = time.time()
    o3 = c_conv_forward(z, K, b)
    print("cython v2 耗时:{}".format(time.time() - s))

    print(np.allclose(o1, o2), np.allclose(o2, o3))


def test_conv_backward():
    """
    卷积反向传播
    :return:
    """
    z = np.random.randn(4, 3, 224, 224)
    K = np.random.randn(3, 64, 3, 3)
    next_dz = np.random.randn(4, 64, 224, 224)

    from layers import conv_backward as conv_backward_v1
    s = time.time()
    dk1, db1, dz1 = conv_backward_v1(next_dz, K, z, padding=(1, 1))
    print("v1 耗时:{}".format(time.time() - s))
    s = time.time()
    dk2, db2, dz2 = conv_backward(next_dz, K, z, padding=(1, 1))
    print("v2 耗时:{}".format(time.time() - s))

    print(np.allclose(dk1, dk2),
          np.allclose(db1, db2),
          np.allclose(dz1, dz2))


def test_max_pooling():
    """
    池化结果一样，速度差数十倍
    :return:
    """
    z = np.random.randn(4, 24, 224, 224)
    from layers import max_pooling_forward_bak
    s = time.time()
    o1 = max_pooling_forward_bak(z, (2, 2))
    print("max pooling v1 耗时:{}".format(time.time() - s))
    s = time.time()
    o2 = max_pooling_forward(z, (2, 2))
    print("max pooling v2 耗时:{}".format(time.time() - s))

    print(np.allclose(o1, o2))


def test_max_pooling_backward():
    """
    池化梯度结果一样，速度差约十倍
    :return:
    """
    next_dz = np.random.randn(4, 24, 112, 112)
    z = np.random.randn(4, 24, 224, 224)
    from layers import max_pooling_backward_bak
    s = time.time()
    o1 = max_pooling_backward_bak(next_dz, z, (2, 2))
    print("max pooling backward v1 耗时:{}".format(time.time() - s))
    s = time.time()
    o2 = max_pooling_backward(next_dz, z, (2, 2))
    print("max pooling backward v2 耗时:{}".format(time.time() - s))

    print(np.allclose(o1, o2))


def test_global_avg_pooling_backward():
    """
    池化结果一样，速度差约十倍
    :return:
    """
    next_dz = np.random.randn(32, 512)
    z = np.random.randn(32, 512, 7, 7)
    from layers import global_avg_pooling_backward as global_avg_pooling_backward_v1
    s = time.time()
    o1 = global_avg_pooling_backward_v1(next_dz, z)
    print("global avg pooling backward v1 耗时:{}".format(time.time() - s))
    s = time.time()
    o2 = global_avg_pooling_backward(next_dz, z)
    print("global avg pooling backward v2 耗时:{}".format(time.time() - s))

    print(np.allclose(o1, o2))


if __name__ == '__main__':
    # test_single_conv()
    # test_conv()
    test_conv_backward()
    # test_max_pooling()
    # test_max_pooling_backward()
    # test_global_avg_pooling_backward()
