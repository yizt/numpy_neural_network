

[TOC]

## Max Pooling

### 前向过程

参考[池化层的反向传播](0_2_5-池化层的反向传播-MaxPooling、AveragePooling、GlobalAveragePooling.md)中公式(1)


```python
import numpy as np
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
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)

    # 输出的高度和宽度
    out_h = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
    out_w = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1

    pool_z = np.zeros((N, C, out_h, out_w))

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_z[n, c, i, j] = np.max(padding_z[n, c,
                                                          strides[0] * i:strides[0] * i + pooling[0],
                                                          strides[1] * j:strides[1] * j + pooling[1]])
    return pool_z
```

### 反向过程

参考[池化层的反向传播](0_2_5-池化层的反向传播-MaxPooling、AveragePooling、GlobalAveragePooling.md)中公式(6)


```python
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
    return _remove_padding(padding_dz, padding) 

```

## Average Pooling

### 前向过程

参考[池化层的反向传播](0_2_5-池化层的反向传播-MaxPooling、AveragePooling、GlobalAveragePooling.md)中公式(2)


```python
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

    pool_z = np.zeros((N, C, out_h, out_w))

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_z[n, c, i, j] = np.mean(padding_z[n, c,
                                                           strides[0] * i:strides[0] * i + pooling[0],
                                                           strides[1] * j:strides[1] * j + pooling[1]])
    return pool_z
```

### 反向过程

参考[池化层的反向传播](0_2_5-池化层的反向传播-MaxPooling、AveragePooling、GlobalAveragePooling.md)中公式(9)


```python
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
    return _remove_padding(padding_dz, padding)  
```

## Global Max Pooling

### 前向过程

参考[池化层的反向传播](0_2_5-池化层的反向传播-MaxPooling、AveragePooling、GlobalAveragePooling.md)中公式(3)


```python
def global_max_pooling_forward(z):
    """
    全局最大池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :return:
    """
    return np.max(np.max(z, axis=-1), -1)
```

### 反向过程

参考[池化层的反向传播](0_2_5-池化层的反向传播-MaxPooling、AveragePooling、GlobalAveragePooling.md)中公式(10)


```python
def global_max_pooling_forward(next_dz, z):
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
```

## Global Average Pooling

### 前向过程

参考[池化层的反向传播](0_2_5-池化层的反向传播-MaxPooling、AveragePooling、GlobalAveragePooling.md)中公式(4)


```python
def global_avg_pooling_forward(z):
    """
    全局平均池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :return:
    """
    return np.mean(np.mean(z, axis=-1), axis=-1)
```

### 反向过程

参考[池化层的反向传播](0_2_5-池化层的反向传播-MaxPooling、AveragePooling、GlobalAveragePooling.md)中公式(12)


```python
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
```

## Cython加速

对于最大池化层的前向和后向过程使用Cython编译加速,实际测试发现耗时减少约20%,貌似提升效果不大;对Cython使用不精通，哪位大佬知道如何改进，请不吝赐教，感谢！！


```python
%load_ext Cython
```

    The Cython extension is already loaded. To reload it, use:
      %reload_ext Cython



```cython
%%cython
cimport cython
cimport numpy as np
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

    cdef np.ndarray[double, ndim= 4] pool_z = np.zeros((N, C, out_h, out_w)).astype(np.float64)

    cdef unsigned int n, c, i, j
    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_z[n, c, i, j] = np.max(padding_z[n, c,
                                                          strides[0] * i:strides[0] * i + pooling[0],
                                                          strides[1] * j:strides[1] * j + pooling[1]])
    return pool_z
```



```cython
%%cython
cimport cython
cimport numpy as np
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
    cdef np.ndarray[double, ndim = 4] padding_dz = np.zeros_like(padding_z).astype(np.float64)

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
```



