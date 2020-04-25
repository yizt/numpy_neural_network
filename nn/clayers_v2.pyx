# -*- coding: utf-8 -*-
"""
 @File    : clayers_v2.pyx
 @Time    : 2020/4/25 下午6:47
 @Author  : yizuotian
 @Description    : 高效卷积的cython版本
"""
cimport cython
import numpy as np
cimport numpy as np

def conv_forward(np.ndarray[double, ndim=4] z,
                   np.ndarray[double, ndim=4] K,
                   np.ndarray[double, ndim=1] b,
                   tuple padding=(0, 0)):
    """
    多通道卷积前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param K: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
    :param b: 偏置,形状(D,)
    :param padding: padding
    :return: conv_z: 卷积结果[N,D,oH,oW]
    """
    cdef np.ndarray[double, ndim= 4] padding_z = np.lib.pad(z,
                                ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                                'constant',
                                constant_values=0)
    cdef unsigned int N = padding_z.shape[0]
    cdef unsigned int height = padding_z.shape[2]
    cdef unsigned int width = padding_z.shape[3]
    cdef unsigned int C = K.shape[0]
    cdef unsigned int D = K.shape[1]

    cdef unsigned int k1 = K.shape[2]
    cdef unsigned int k2 = K.shape[3]

    # 输出的高度和宽度
    cdef unsigned int oh = 1 + (height - k1)
    cdef unsigned int ow = 1 + (width - k2)

    # 扩维
    cdef np.ndarray[double, ndim= 5] padding_z_e = padding_z[:, :, np.newaxis, :, :]  # 扩维[N,C,1,H,W] 与K [C,D,K1,K2] 可以广播
    cdef np.ndarray[double, ndim= 4] conv_z = np.zeros((N, D, oh, ow))

    # 批量卷积
    cdef unsigned int c, i, j, h, w
    cdef np.ndarray[double, ndim= 6] K_e = K[:, :, :, :, np.newaxis, np.newaxis]
    if k1 * k2 < oh * ow * 10:
        for c in range(C):
            for i in range(k1):
                for j in range(k2):
                    # [N,1,oh,ow]*[D,1,1] =>[N,D,oh,ow]
                    conv_z += padding_z_e[:, c, :, i:i + oh, j:j + ow] * K_e[c, :, i, j]
    else:  # 大卷积核，遍历空间更高效
        # print('大卷积核，遍历空间更高效')
        for c in range(C):
            for h in range(oh):
                for w in range(ow):
                    # [N,1,k1,k2]*[D,k1,k2] =>[N,D,k1,k2] => [N,D]
                    conv_z[:, :, h, w] += np.sum(padding_z_e[:, c, :, h:h + k1, w:w + k2] * K[c], axis=(2, 3))

    # 增加偏置 [N, D, oh, ow]+[D, 1, 1]
    conv_z += b[:, np.newaxis, np.newaxis]
    return conv_z