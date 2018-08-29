

[TOC]

## 依赖知识

a) 熟悉[全连接层、损失函数的反向传播](0_1-全连接层、损失函数的反向传播.md)

b) 熟悉[卷积层的反向传播-单通道、无padding、步长1](0_2_1-卷积层的反向传播-单通道、无padding、步长1.md)

c) 熟悉以上两点的依赖知识



## 约定说明

a) $l$ 代表网络的第$l$ 层, $z^l$ 代表第$l$ 层卷积，$z_{d,i,j}^l$ 代表第$l$ 层卷积第$d$ 通道$(i,j)$ 位置的值; $z^l$ 的通道数为$C^l$, 高度和宽度分别为$H^l,\hat W^l$ ($\color{red}{避免与权重相同}$) 

b) $W^{l-1},b^{l-1}$ 代表连接第$l-1$ 层和第$l$ 层的卷积核权重和偏置; 卷积核的维度为$(k_1^{l-1},k_2^{l-1})$ 。

c) 记损失函数L关于第$l$ 层卷积的输出$z^l$ 的偏导为$\delta^l = \frac {\partial L} {\partial z^l}  \ \ \ (3)$   



## 前向传播

​      根据以上约定，卷积核权重$W^{l-1} \in \Bbb R^{k_1^{l-1} \times k_2^{l-1} \times C^{l-1} \times C^{l}}$ ,偏置$b^{l-1} \in \Bbb R^{C^l}$ ,每个输出通道一个偏置。

​      则有第$l$ 层卷积层,第$d$个通道输出为:
$$
\begin{align}
&z^l_{d,i,j} = \sum_{c=1}^{C^{l-1}}\sum_{m=0}^{k_1^{l-1}-1} \sum_{n=0}^{k_2^{l-1}-1} W_{m,n,c,d}^{l-1} z_{c,i+m,j+n}^{l-1} + b^{l-1}_d  & i \in [0,H^l-1], j\in [0,\hat W^l-1]\tag 4
\end{align}
$$
其中：$H^l = H^{l-1} - k_1^{l-1} + 1;\ \ \ \ \  \hat W^l = \hat W^{l-1} - k_2^{l-1} + 1 $ ; 注意前后通道直接相当于全连接，即前后两个卷积层直接所有通道都互相连接。



## 反向传播

### 权重梯度

a) 首先来看损失函数$L$关于第$l-1$层权重$W^{l-1}$和偏置$b^{l-1}$的梯度：
$$
\begin{align}
&\frac {\partial L} {\partial W_{m,n,c,d}^{l-1}} 
= \sum_i \sum_j \frac {\partial L} {\partial z^l_{d,i,j}} * \frac {\partial z^l_{d,i,j}} {\partial W_{m,n,c,d}^{l-1}} &//l层的d通道每个神经元都有梯度传给权重W^{l-1}_{m,n,c,d}\\
&=\sum_i \sum_j \delta^l_{d,i,j} * \frac {\partial ( \sum_{c=1}^{C^{l-1}}\sum_{m=0}^{k_1^{l-1}-1} \sum_{n=0}^{k_2^{l-1}-1} W_{m,n,c,d}^{l-1} z_{c,i+m,j+n}^{l-1} + b^{l-1}_d  )} {\partial W^{l-1}_{m,n,c,d}}  \\
&=\sum_i \sum_j \delta^l_{d,i,j} * z^{l-1}_{c,i+m,j+n} \tag 5
\end{align} \\
$$
​     对比公式(5)和[单通道](0_2_1-卷积层的反向传播-单通道、无padding、步长1.md)中公式(4),可以发现,损失函数$L$关于第$l-1$层权重$W^{l-1}_{:,:c,d}$梯度就是以$\delta^l_d$ 为卷积核在$z^{l-1}_c$上做卷积的结果(这里没有偏置项),单通道对单通道的卷积。



b) 损失函数$L$关于第$l-1$层偏置$b^{l-1}$的梯度同
$$
\begin{align}
\frac {\partial L} {\partial b^{l-1}_d} =\sum_i \sum_j \delta^l_{d,i,j}  \tag 6
\end{align}
$$

### l-1层梯度

​       由[单通道](0_2_1-卷积层的反向传播-单通道、无padding、步长1.md) 可知第$l$层的第$d$个通道传给第$l-1$层$c$通道的梯度为:
$$
\sum_{m=0}^{k_1^{l-1}-1} \sum_{n=0}^{k_2^{l-1}-1}rot_{180^\circ} W^{l-1}_{m,n,c,d}p\delta^{l}_{d,i+m,j+n}  \tag 7
$$

​       而l层的每个通道都有梯度返回给第l-1层的第c个通道，因此有:
$$
\delta^{l-1}_{c,i,j}=\sum_{d=1}^{C^l}\sum_{m=0}^{k_1^{l-1}-1} \sum_{n=0}^{k_2^{l-1}-1}rot_{180^\circ} W^{l-1}_{m,n,c,d}p\delta^{l}_{d,i+m,j+n} \tag 8
$$

​        其中：
$$
p\delta^l_{d,i,j}=\begin{cases}
\delta^l_{d,i-k_1^{l-1}+1,j-k_2^{l-1}+1}  & i \in[k_1^{l-1}-1,H^l+k_1^{l-1}-2] 且j \in [k_2^{l-1}-1,\hat W^l+k_2^{l-1}-2] \\
0 & i,j其它情况 \tag {12}
\end{cases}
$$

