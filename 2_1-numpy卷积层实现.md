[TOC]


## 前向传播

  参考 [卷积层的反向传播-多通道、无padding、步长1](0_2_4-卷积层的反向传播-多通道、有padding、步长不为1.md)中公式(4) 



```python
import numpy as np
def conv_forward(z, K, b, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param K: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
    :param b: 偏置,形状(D,)
    :param padding: padding
    :param strides: 步长
    :return: 卷积结果
    """
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    N, _, height, width = padding_z.shape
    C, D, k1, k2 = K.shape
    assert (height - k1) % strides[0] == 0, '步长不为1时，步长必须刚好能够被整除'
    assert (width - k2) % strides[1] == 0, '步长不为1时，步长必须刚好能够被整除'
    conv_z = np.zeros((N, D, 1 + (height - k1) // strides[0], 1 + (width - k2) // strides[1]))
    for n in np.arange(N):
        for d in np.arange(D):
            for h in np.arange(height - k1 + 1)[::strides[0]]:
                for w in np.arange(width - k2 + 1)[::strides[1]]:
                    conv_z[n, d, h // strides[0], w // strides[1]] = np.sum(padding_z[n, :, h:h + k1, w:w + k2] * K[:, d]) + b[d]
    return conv_z
```

## 反向传播


参考[卷积层的反向传播-多通道、无padding、步长1](0_2_4-卷积层的反向传播-多通道、有padding、步长不为1.md)中公式(5),(6)和(9) 

首先定义两个内部函数,一个是在反向中对于步长大于1的卷积核,对输出层梯度行列(高度和宽宽)之间插入零;另一个是对于padding不为零的卷积核，在对输入层求梯度后，剔除padding


```python
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
```


```python
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
```


```python
def conv_backward(next_dz, K, z, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积层的反向过程
    :param next_dz: 卷积输出层的梯度,(N,D,H',W'),H',W'为卷积输出层的高度和宽度
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
    ppadding_next_dz = np.lib.pad(padding_next_dz, ((0, 0), (0, 0), (k1 - 1, k1 - 1), (k2 - 1, k2 - 1)), 'constant', constant_values=0)
    dz = conv_forward(ppadding_next_dz.astype(np.float64), swap_flip_K.astype(np.float64), np.zeros((C,), dtype=np.float64))

    # 求卷积核的梯度dK
    swap_z = np.swapaxes(z, 0, 1)  # 变为(C,N,H,W)与
    dK = conv_forward(swap_z.astype(np.float64), padding_next_dz.astype(np.float64), np.zeros((D,), dtype=np.float64))

    # 偏置的梯度
    db = np.sum(np.sum(np.sum(next_dz, axis=-1), axis=-1), axis=0)  # 在高度、宽度上相加；批量大小上相加

    # 把padding减掉
    dz = _remove_padding(dz, padding)  # dz[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]

    return dK / N, db / N, dz
```

说明：
a)求卷积核的梯度dk时，输出就是维度是卷积核的维度(C,D,k1,k2),卷积核是输出层的梯度，其维度为(N,D,H',W'),而输入z的维度是(N,C,H,W);这样是无法做卷积的，需要对输入z在做坐标轴交换变为(C,N,H,W)
b)求输入层的梯度dz时，需要最后将padding去除

##  测试卷积

   以下实现一个简单的卷积层,输入为随机数组成的多维向量，训练前向反向过程，使得输出为数值全为1的多维向量


```python
# 随机初始化输入及卷积核和偏置
z = np.random.randn(3, 3, 28, 28).astype(np.float64)
K = np.random.randn(3, 4, 3, 3).astype(np.float64) 
b = np.zeros(4).astype(np.float64)
```

设目标输出是全为1的多维向量,第一维是批量、第二维是通道、最后两维是高度和宽度


```python
# 没有padding,输入的高度和宽度是28*28,卷积核是3*3,输出高度和宽度就是28-3+1=26
y_true = np.ones((3,4,26,26))
```

使用均方差测试训练结果


```python
from nn.losses import mean_squared_loss
for i in range(1000):
    # 前向
    next_z = conv_forward(z, K, b)
    # 反向
    loss, dy = mean_squared_loss(next_z, y_true)
    dK, db, _ = conv_backward(dy, K, z)
    # 更新梯度
    K -= 0.001 * dK
    b -= 0.001 * db

    # 打印损失
    print("step:{},loss:{}".format(i, loss))

    if np.allclose(y_true, next_z):
        print("yes")
        break
```

    step:0,loss:564.3438940391925
    step:1,loss:66.17172343756643
    step:2,loss:9.198997813124956
    step:3,loss:1.4069519751440192
    step:4,loss:0.2291942712009542
    step:5,loss:0.03908173551298978
    step:6,loss:0.0069035446737729855
    step:7,loss:0.0012548420117566993
    step:8,loss:0.0002336212264014327
    step:9,loss:4.439647286022715e-05
    step:10,loss:8.588536282082359e-06
    step:11,loss:1.6875253762932353e-06
    step:12,loss:3.3613308537260323e-07
    step:13,loss:6.776104204755497e-08
    step:14,loss:1.380456978130391e-08
    step:15,loss:2.838444613389672e-09
    step:16,loss:5.883808721904564e-10
    step:17,loss:1.228345750173361e-10
    yes


可以看到经过很少的十几步迭代就收敛了，下面来看看训练后输出结果,可以看到结果已经很接近1了;说卷积层的前向后向过程是正确的


```python
print("min:{},max:{},avg:{}".format(np.min(next_z),np.max(next_z),np.average(next_z)))
```

    min:0.9999917939446406,max:1.0000086638248415,avg:0.9999999546434823


## Cython加速

对于卷积层的前向过程使用Cython编译加速,实际测试发现耗时减少约20%,貌似提升效果不大;对Cython使用不精通，哪位大佬知道如何改进，请不吝赐教，感谢！！



```python
%load_ext Cython
```

```python
%%cython
cimport cython
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
    cdef np.ndarray[double, ndim= 4] conv_z = np.zeros((N, D, 1 + (height - k1) // s0, 1 + (width - k2) // s1))
    cdef unsigned int n, d, h, w
    for n in np.arange(N):
        for d in np.arange(D):
            for h in np.arange(height - k1 + 1)[::s0]:
                for w in np.arange(width - k2 + 1)[::s1]:
                    conv_z[n, d, h // s0, w // s1] = np.sum(padding_z[n, :, h:h + k1, w:w + k2] * K[:, d]) + b[d]

    return conv_z
```

