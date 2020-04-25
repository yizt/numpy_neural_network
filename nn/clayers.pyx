# -*- coding: utf-8 -*-
"""
Created on 2018/9/10 16:31

@author: mick.yi

用cython重写部分层，提高运行效率

"""
cimport cython
import numpy as np
cimport numpy as np


cpdef conv_forward(np.ndarray[double, ndim=4] z,
                   np.ndarray[double, ndim=4] K,
                   np.ndarray[double, ndim=1] b,
                   tuple padding=(0, 0),
                   tuple strides=(1, 1)):
    """
    多通道卷积前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param K: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
    :param b: 偏置,形状(D,)
    :param padding: padding
    :param strides: 步长
    :return: 卷积结果
    """
    cdef np.ndarray[double, ndim= 4] padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]),
                                                                 (padding[1], padding[1])), 'constant', constant_values=0)
    cdef unsigned int N = padding_z.shape[0]
    cdef unsigned int height = padding_z.shape[2]
    cdef unsigned int  width = padding_z.shape[3]
    cdef unsigned int C = K.shape[0]
    cdef unsigned int D = K.shape[1]

    cdef unsigned int k1 = K.shape[2]
    cdef unsigned int k2 = K.shape[3]

    cdef unsigned int s0 = strides[0]
    cdef unsigned int s1 = strides[1]

    assert (height - k1) % s0 == 0, '步长不为1时，步长必须刚好能够被整除'
    assert (width - k2) % s1 == 0, '步长不为1时，步长必须刚好能够被整除'
    cdef np.ndarray[double, ndim= 4] conv_z = np.zeros((N, D, 1 + (height - k1) // s0, 1 + (width - k2) // s1)).astype(np.float)
    cdef unsigned int n, d, h, w
    for n in np.arange(N):
        for d in np.arange(D):
            for h in np.arange(height - k1 + 1)[::s0]:
                for w in np.arange(width - k2 + 1)[::s1]:
                    conv_z[n, d, h // s0, w // s1] = np.sum(padding_z[n, :, h:h + k1, w:w + k2] * K[:, d]) + b[d]

    return conv_z



def _remove_padding(np.ndarray[double, ndim=4] z, tuple padding):
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


cpdef max_pooling_forward(np.ndarray[double, ndim=4] z,
                        tuple pooling,
                        tuple strides=(2, 2),
                        tuple padding=(0, 0)):
    """
    最大池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    cdef unsigned int N = z.shape[0]
    cdef unsigned int C = z.shape[1]
    cdef unsigned int H = z.shape[2]
    cdef unsigned int W = z.shape[3]
    # 零填充
    cdef np.ndarray[double, ndim= 4] padding_z = np.lib.pad(z, ((0, 0), (0, 0),
                                                                 (padding[0], padding[0]), (padding[1], padding[1])),
                                                             'constant', constant_values=0)

    # 输出的高度和宽度
    cdef unsigned int out_h = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
    cdef unsigned int out_w = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1

    cdef np.ndarray[double, ndim= 4] pool_z = np.zeros((N, C, out_h, out_w)).astype(np.float)

    cdef unsigned int n, c, i, j
    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_z[n, c, i, j] = np.max(padding_z[n, c,
                                                          strides[0] * i:strides[0] * i + pooling[0],
                                                          strides[1] * j:strides[1] * j + pooling[1]])
    return pool_z


cpdef max_pooling_backward(np.ndarray[double, ndim=4] next_dz,
                         np.ndarray[double, ndim=4] z,
                         tuple pooling,
                         tuple strides=(2, 2),
                         tuple padding=(0, 0)):
    """
    最大池化反向过程
    :param next_dz：损失函数关于最大池化输出的损失
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    cdef unsigned int N = z.shape[0]
    cdef unsigned int C = z.shape[1]
    cdef unsigned int H = z.shape[2]
    cdef unsigned int W = z.shape[3]
    cdef unsigned int out_h = next_dz.shape[2]
    cdef unsigned int out_w = next_dz.shape[3]
    # 零填充
    cdef np.ndarray[double, ndim = 4] padding_z = np.lib.pad(z, ((0, 0), (0, 0),
                                                                (padding[0], padding[0]),
                                                                (padding[1], padding[1])),
                                                            'constant', constant_values=0)
    # 零填充后的梯度
    cdef np.ndarray[double, ndim = 4] padding_dz = np.zeros_like(padding_z).astype(np.float)

    cdef unsigned int n, c, i, j
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
