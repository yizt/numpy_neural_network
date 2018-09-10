# -*- coding: utf-8 -*-
"""
Created on 2018/9/10 16:31

@author: mick.yi

用cython重写部分层，提高运行效率

"""
cimport cython
import numpy as np
cimport numpy as np


def conv_forward(np.ndarray[double, ndim=4] z,
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

    assert (height - k1) % strides[0] == 0, '步长不为1时，步长必须刚好能够被整除'
    assert (width - k2) % strides[1] == 0, '步长不为1时，步长必须刚好能够被整除'
    cdef np.ndarray[double, ndim= 4] conv_z = np.zeros((N, D, 1 + (height - k1) // strides[0], 1 + (width - k2) // strides[1]))
    for n in np.arange(N):
        for d in np.arange(D):
            for h in np.arange(height - k1 + 1)[::strides[0]]:
                for w in np.arange(width - k2 + 1)[::strides[1]]:
                    conv_z[n, d, h // strides[0], w // strides[1]] = np.sum(padding_z[n, :, h:h + k1, w:w + k2] * K[:, d]) + b[d]
    return conv_z

