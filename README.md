# numpy_neuron_network
仅使用numpy从头构建神经网络, 包括如下内容(持续更新中....)

0. 网络中梯度反向传播公式推导


1. 层：FC层,卷积层,池化层,Flatten
2. 激活函数: ReLU、LeakyReLU、PReLU、ELU、SELU
3. 损失函数：均方差、交叉熵
4. 模型的保存、部署
5. 案例学习：线性回归、图像分类
6. 迁移学习、模型精调
7. 进阶：RNN、LSTM、GRU、BN

[TOC]

## 运行工程

环境：python 3.6.x

依赖：numpy>=1.15.0、Cython、jupyter

a) 下载

```shell
git clone https://github.com/yizt/numpy_neuron_network
```



b) 编译nn/clayers.pyx

```shell
cd numpy_neuron_network
python setup.py build_ext -i
```

c) 启动工程,所有的notebook都可以直接运行

```shell
jupyter notebook --allow-root --ip 0.0.0.0
```





## 基础知识

[0_1-全连接层、损失函数的反向传播](0_1-全连接层、损失函数的反向传播.md) 、[csdn地址](https://blog.csdn.net/csuyzt/article/details/81839388)

[0_2_1-卷积层的反向传播-单通道、无padding、步长1](0_2_1-卷积层的反向传播-单通道、无padding、步长1.md) 、[csdn地址](https://blog.csdn.net/csuyzt/article/details/81952377)

[0_2_2-卷积层的反向传播-多通道、无padding、步长1](0_2_2-卷积层的反向传播-多通道、无padding、步长1.md) 、[csdn地址](https://blog.csdn.net/csuyzt/article/details/82026408)

[0_2_3-卷积层的反向传播-多通道、无padding、步长不为1](0_2_3-卷积层的反向传播-多通道、无padding、步长不为1.md) 、[csdn地址](https://blog.csdn.net/csuyzt/article/details/82120173)

[0_2_4-卷积层的反向传播-多通道、有padding、步长不为1](0_2_4-卷积层的反向传播-多通道、有padding、步长不为1.md) 、[csdn地址](https://blog.csdn.net/csuyzt/article/details/82632918)

[0_2_5-池化层的反向传播-MaxPooling、AveragePooling、GlobalAveragePooling、GlobalMaxPooling](0_2_5-池化层的反向传播-MaxPooling、AveragePooling、GlobalAveragePooling.md) 、[csdn地址](https://blog.csdn.net/csuyzt/article/details/82633051)

[0_3-激活函数的反向传播-ReLU、LeakyReLU、PReLU、ELU、SELU](0_3-激活函数的反向传播-ReLU、LeakyReLU、PReLU、ELU、SELU.md) 、[csdn地址](https://blog.csdn.net/csuyzt/article/details/82320589)

[0_4-优化方法-SGD、AdaGrad、RMSProp、Adadelta、Adam](0_4-优化方法-SGD、AdaGrad、RMSProp、Adadelta、Adam.md) 、[csdn地址](https://blog.csdn.net/csuyzt/article/details/82633173)





## DNN练习

[1_1_1-全连接神经网络做线性回归](1_1_1-全连接神经网络做线性回归.md) 、[csdn地址](https://blog.csdn.net/csuyzt/article/details/81841817)

[1_1_2-全连接神经网络做mnist手写数字识别](1_1_2-全连接神经网络做mnist手写数字识别.md) 、[csdn地址](https://blog.csdn.net/csuyzt/article/details/82320663)



## CNN练习

[2_1-numpy卷积层实现](2_1-numpy卷积层实现.md) 、[csdn地址](https://blog.csdn.net/csuyzt/article/details/82668153)

[2_2-numpy池化层实现]() 、[csdn地址](https://blog.csdn.net/csuyzt/article/details/82668941)

[2_3-numpy-cnn-mnist手写数字识别](2_3-numpy-cnn-mnist手写数字识别.md) 、[csdn地址](https://blog.csdn.net/csuyzt/article/details/82669885)

2_4-对抗神经网络 、[csdn地址]()



## 其它

模型的保存、部署

精调网络





## 进阶

5-1-RNN反向传播

5-2-LSTM反向传播

5-3-GRU反向传播

5-4-RNN、LSTM、GRU实现

5-5-案例-lstm连续文字识别



6-1-Batch Normalization反向传播

6-2-Batch Normalization实现









