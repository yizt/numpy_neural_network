[TOC]



## 依赖知识

a) 了解神经网络的基础知识,熟悉卷积网络

b) 熟悉导数的链式法则及常见函数的导数

c) 熟悉常见的优化方法，梯度下降，随机梯度下降等

d) 熟悉矩阵和向量的乘加运算

e) 熟悉[全连接层、损失函数的反向传播](0_1-全连接层、损失函数的反向传播.md)



## 卷积定义

​          对于输入图像$I$ , 使用维度为$k_1 \times k_2$ 的滤波器$K$ ,卷积的定义如下：

$$
\begin{align}
(I * K)_{ij} &= \sum_{m = 0}^{k_1 - 1} \sum_{n = 0}^{k_2 - 1} I(i+m, j+n)K(m,n) \tag {1}
\end{align}
$$

注意：这里的卷积跟数学中定义的卷积不是完全一致的，数学中这叫协相关; 卷积和协相关的区别详见[deep learning](https://github.com/exacity/deeplearningbook-chinese) 第九章282页。神经网络中一般都把公式(1)的定义当做卷积。



## CNN卷积网络

​         卷积网络包含一系列的卷积层，每层由输入特征图$I$，一堆滤波器$K$ 和偏置$b$ . 假设输入的高度、宽度、通道数分别为$H,W,C$; 则$I \in \Bbb R^{H \times W \times C}$ , 输出$D$ 个通道的卷积层，则有卷积核$K \in \Bbb R^{k_1 \times k_2 \times C \times D}$ ,偏置$b \in \Bbb R^D$ ,每个输出通道一个偏置。则其中一个输出通道的可以如下表示：
$$
\begin{align}
(I \ast K)_{ij} &= \sum_{m = 0}^{k_1 - 1} \sum_{n = 0}^{k_2 - 1} \sum_{c = 1}^{C} K_{m,n,c} \cdot I_{i+m, j+n, c} + b \tag {2}
\end{align}
$$
​          

​           有心读者肯定会疑惑公式(2),没有体现padding和卷积核的步长。由于卷积相对复杂，没有办法一次性说明的非常清楚，计划分几次来逐步说明；本文接下来将推导最简单的卷积反向传播公式。假定输入输出通道都为1，即$C=D=1$, 且卷积核的padding=0,步长为1。



## 约定说明

a) $l$ 代表网络的第$l$ 层, $z^l$ 代表第$l$ 层卷积，$z_{i,j}^l$ 代表第$l$ 层卷积的$(i,j)$ 位置的值; $z^l$ 的高度和宽度分别为$H^l,\hat W^l$ ($\color{red}{避免与权重相同}$)

b) $W^{l-1},b^{l-1}$ 代表连接第$l-1$ 层和第$l$ 层的卷积核权重和偏置; 卷积核的维度为$(k_1^{l-1},k_2^{l-1})$ 。

c) 记损失函数L关于第$l$ 层卷积的输出$z^l$ 的偏导为$\delta^l = \frac {\partial L} {\partial z^l}  \ \ \ (3)$   

根据以上约定第$l$ 层卷积输出为:
$$
\begin{align}
&z^l_{i,j} = \sum_{m=0}^{k_1^{l-1}-1} \sum_{n=0}^{k_2^{l-1}-1} W_{m,n}^{l-1} z_{i+m,j+n}^{l-1} + b^{l-1}  & i \in [0,H^l-1], j\in [0,\hat W^l-1]\tag 4
\end{align}
$$
其中：$H^l = H^{l-1} - k_1^{l-1} + 1;\ \ \ \ \  \hat W^l = \hat W^{l-1} - k_2^{l-1} + 1 $



## 误差反向传播

### 权重梯度

a) 首先来看损失函数$L$关于第$l-1$层权重$W^{l-1}$和偏置$b^{l-1}$的梯度
$$
\begin{align}
&\frac {\partial L} {\partial W_{m,n}^{l-1}} 
= \sum_i \sum_j \frac {\partial L} {\partial z^l_{i,j}} * \frac {\partial z^l_{i,j}} {\partial W_{m,n}^{l-1}} &//l层的每个神经元都有梯度传给权重W^{l-1}_{m,n}\\
&=\sum_i \sum_j \delta^l_{i,j} * \frac {\partial ( \sum_{m=0}^{k_1^{l-1}} \sum_{n=0}^{k_2^{l-1}} W_{m,n}^{l-1} z_{i+m,j+n}^{l-1} + b^{l-1})} {\partial W^{l-1}_{m,n}} \\
&=\sum_i \sum_j \delta^l_{i,j} * z^{l-1}_{i+m,j+n} \tag 5
\end{align} \\
$$


​        对比公式(5)和公式(4),可以发现,损失函数$L$关于第$l-1$层权重$W^{l-1}$ 的梯度就是以$\delta^l$ 为卷积核在$z^{l-1}$上做卷积的结果(这里没有偏置项)。多么简介对称呀!!!。



b) 同理可得
$$
\begin{align}
\frac {\partial L} {\partial b^{l-1}} =\sum_i \sum_j \delta^l_{i,j}  \tag 6
\end{align}
$$

###  l-1层梯度

​     然后再来看看损失函数关于第$l-1$层输出的梯度
$$
\begin{align}
&\delta^{l-1}_{i^{\prime},j^{\prime}}=\frac {\partial L} {\partial z_{i^{\prime},j^{\prime}}^{l-1}} 
= \sum_i \sum_j \frac {\partial L} {\partial z^l_{i,j}} * \frac {\partial z^l_{i,j}} {\partial z_{i^{\prime},j^{\prime}}^{l-1}} \\
&=\sum_i \sum_j \delta^l_{i,j} * \frac {\partial ( \sum_{m=0}^{k_1^{l-1}} \sum_{n=0}^{k_2^{l-1}} W_{m,n}^{l-1} z_{i+m,j+n}^{l-1} + b^{l-1})} {\partial z_{i^{\prime},j^{\prime}}^{l-1}} &//当i=i^{\prime}-m, j=j^{\prime}-n时有梯度W^{l-1}_{m,n}\\
&=\sum_i \sum_j \delta^l_{i,j}  W^{l-1}_{m,n} &//此时m=i^{\prime}-i ,n=j^{\prime}-j\\
&=\sum_m \sum_n \delta^l_{i^{\prime}-m,j^{\prime}-n}W^{l-1}_{m,n}   \ \ \ \ \ \ \  (7) &//此时i=i^{\prime}-m \in[0,H^l-1],j=j^{\prime}-n \in [0,\hat W^l-1]  \\
&=\sum_i \sum_j \delta^l_{i,j}  W^{l-1}_{i^{\prime}-i,j^{\prime}-j}  \ \ \ \ \ \ \  (8) &//需要满足i^{\prime}-i \in [0,k_1^{l-1}-1],j^{\prime}-j \in [0,k_2^{l-1}-1]
\end{align}
$$

​           

​            约束条件:$i^{\prime}-i \in [0,k_1^{l-1}-1],j^{\prime}-j \in [0,k_2^{l-1}-1] $ 

​            变换一下就是:$ i \in [i^{\prime}+1-k_1^{l-1},i^{\prime}],j \in [j^{\prime}+1-k_2^{l-1},j^{\prime}] \tag 9$

​            同时$i,j$ 需要满足公式(4)的约束条件:
$$
i\in [0,H^l-1], j\in [0,\hat W^l-1] \tag {10}
$$
​            因此有
$$
\begin{cases}
i \in [\max(0,i^{\prime}+1-k_1^{l-1}),\min(H^l-1,i^{\prime})] \\
j \in [\max(0,j^{\prime}+1-k_2^{l-1}),\min(\hat W^l-1,j^{\prime})]   \tag {11}
\end{cases}
$$
​          下面来看一个例子，对于l-1层 $5 \times 5$ 的卷积层，卷积核$3 \times 3$ , 则输出的l层卷积大小为5-3-1=3，也就是$3 \times 3$ , 此时有：
$$
\begin{cases}
i \in [\max(0,i^{\prime}-2),\min(2,i^{\prime})] \\
j \in [\max(0,j^{\prime}-2,\min(2,j^{\prime})]  
\end{cases}
$$
根据公式(7)及其约束条件有：
$$
\begin{align}
&\delta^{l-1}_{0,0} =\delta^{l}_{0,0}W^{l-1}_{0,0} &i \in [0,0],j \in [0,0] \\
&\delta^{l-1}_{0,1} =\delta^{l}_{0,1}W^{l-1}_{0,0} + \delta^{l}_{0,0}W^{l-1}_{0,1} &i \in [0,0],j \in [0,1] \\
&\delta^{l-1}_{0,2} =\delta^{l}_{0,2}W^{l-1}_{0,0} + \delta^{l}_{0,1}W^{l-1}_{0,1} +\delta^{l}_{0,0}W^{l-1}_{0,2} &i \in [0,0],j \in [0,2] \\
&\delta^{l-1}_{1,0} =\delta^{l}_{1,0}W^{l-1}_{0,0} + \delta^{l}_{0,0}W^{l-1}_{1,0} &i \in [0,1],j \in [0,0] \\
&\delta^{l-1}_{1,1} =\delta^{l}_{1,1}W^{l-1}_{0,0} + \delta^{l}_{0,1}W^{l-1}_{1,0} +\delta^{l}_{1,0}W^{l-1}_{0,1} + \delta^{l}_{0,0}W^{l-1}_{1,1} &i \in [0,1],j \in [0,1] \\
&\delta^{l-1}_{1,2} = \sum_i \sum_j \delta^l_{i,j}  W^{l-1}_{i^{\prime}-i,j^{\prime}-j} & i \in [0,1],j \in [0,2] \\
&... ... \\
&\delta^{l-1}_{2,2} = \sum_i \sum_j \delta^l_{i,j}  W^{l-1}_{i^{\prime}-i,j^{\prime}-j} & i \in [0,2],j \in [0,2] \\
\end{align}
$$

​           等价于以下的卷积


$$
\delta^{l-1}=\left(
\begin{align}
&0, &&0,&&0,&&0,&&0,&&0,&&0 \\
&0, &&0,&&0,&&0,&&0,&&0,&&0 \\
&0,&&0,&&\delta^{l}_{0,0},&&\delta^{l}_{0,1},&&\delta^{l}_{0,2},&&0,&&0\\
&0,&&0,&&\delta^{l}_{1,0},&&\delta^{l}_{1,1},&&\delta^{l}_{1,2},&&0,&&0\\
&0,&&0,&&\delta^{l}_{2,0},&&\delta^{l}_{2,1},&&\delta^{l}_{2,2},&&0,&&0\\
&0,&&0, &&0,&&0,&&0,&&0,&&0 \\
&0,&&0, &&0,&&0,&&0,&&0,&&0
\end{align}
\right) *
\left(
\begin{array}
aW^{l-1}_{2,2},& W^{l-1}_{2,1},& W^{l-1}_{2,0}\\
W^{l-1}_{1,2},& W^{l-1}_{11},& W^{l-1}_{1,0}\\
W^{l-1}_{0,2},& W^{l-1}_{01},& W^{l-1}_{0,0}\\
\end{array}
\right)
$$
​           即以$W^{l-1}$ 翻转$180^\circ$ 的矩阵为卷积核在$\delta^l$ 加上padding=2的矩阵上做卷积的结果。

a) 设$rot_{180^\circ}W^{l-1}$ 为以$W^{l-1}$ 翻转$180^\circ$ 的矩阵后的矩阵

b) 设$p\delta^l$ 为$\delta^l$ 加上padding高宽为卷积核高宽减1即$(k_1^{l-1}-1,k_2^{l-1}-1)$后的梯度矩阵，可知其高度为$H^l+2k_1^{l-1} -2 = H^{l-1} +k_1^{l-1} -1$ ;相应的宽度为$\hat W^{l-1} +k_2^{l-1} -1$ 

c) 卷积核$rot_{180^\circ}W^{l-1}$ 的大小为$(k_1^{l-1},k_1^{l-1})$,在上做完卷积后的长宽刚好与$\delta^{l-1}$ 的高度和宽度一样，即$(H^{l-1},\hat W^{l-1})$ 。

d)  $p\delta^l$ 和$\delta^l$ 的关系如下：
$$
p\delta^l_{i,j}=\begin{cases}
\delta^l_{i-k_1^{l-1}+1,j-k_2^{l-1}+1}  & i \in[k_1^{l-1}-1,H^l+k_1^{l-1}-2] 且j \in [k_2^{l-1}-1,\hat W^l+k_2^{l-1}-2] \\
0 & i,j其它情况 \tag {12}
\end{cases}
$$


​        接下来将证明这个卷积就是$\delta^{l-1}$ 

根据公式(4) 卷积后的$(i,j)$ 位置的值为：
$$
\begin{align}
&\sum_{m=0}^{k_1^{l-1}-1} \sum_{n=0}^{k_2^{l-1}-1}rot_{180^\circ} W^{l-1}_{m,n}p\delta^{l}_{i+m,j+n} \\
&=\sum_{m=0}^{k_1^{l-1}-1} \sum_{n=0}^{k_2^{l-1}-1}W_{k^{l-1}_1-1-m,k_2^{l-1}-1-n}\ p\delta^{l}_{i+m,j+n} \ \ \ \ \ \ //将翻转180^\circ改回来 \\
&=\sum_{m^{\prime}=0}^{k_1^{l-1}-1} \sum_{n^{\prime}=0}^{k_2^{l-1}-1}W_{m^{\prime},n^{\prime}}\ p\delta^{l}_{i+k^{l-1}_1-1-m^{\prime},j+k_2^{l-1}-1-n^{\prime}} \ \ \ \ \ //m^{\prime} +m =k_1^{l-1} -1,n^{\prime}+n=k_2^{l-1}-1 \\
&=\sum_{m=0}^{k_1^{l-1}-1} \sum_{n=0}^{k_2^{l-1}-1}W_{m,n}\ p\delta^{l}_{i+k^{l-1}_1-1-m,j+k_2^{l-1}-1-n}    \ \ \ \ \ //将下标改回来  \\
&=
\sum_{m=0}^{k_1^{l-1}-1} \sum_{n=0}^{k_2^{l-1}-1}W_{m,n}\ \begin{cases} \delta^{l}_{i-m,j-n}  &//i-m \in [0,H^l-1] 且j-n \in[0,\hat W^l-1] \\
0    &//i-m \notin [0,H^l-1] 或j-n \notin[0,\hat W^l-1] \tag {15}
\end{cases}
\end{align}
$$
​         可以看出公式(15)与公式(7)完全一致。



## 结论

a) 卷积前向计算公式如下:
$$
\begin{align}
&z^l_{i,j} = \sum_{m=0}^{k_1^{l-1}-1} \sum_{n=0}^{k_2^{l-1}-1} W_{m,n}^{l-1} z_{i+m,j+n}^{l-1} + b^{l-1}  & i \in [0,H^l-1], j\in [0,\hat W^l-1]\tag 4
\end{align}
$$
b) 损失函数$L$关于第$l-1$层权重$W^{l-1}$ 的梯度，是以损失函数$L$关于第$l$层梯度 $\delta^l$ 为卷积核在$z^{l-1}$上做卷积的结果
$$
\frac {\partial L} {\partial W_{m,n}^{l-1}} =\sum_i \sum_j \delta^l_{i,j} * z^{l-1}_{i+m,j+n} \tag 5
$$
c) 损失函数$L$关于第$l-1$层偏置$b^{l-1}$ 的梯度，是$\delta^l$ 元素之和
$$
\frac {\partial L} {\partial b^{l-1}} =\sum_i \sum_j \delta^l_{i,j}  \tag 6
$$


d) 以损失函数$L$关于第$l-1$层梯度 $\delta^{l-1}$, 是以第$l-1$ 层权重的翻转$rot_{180^\circ} W^{l-1}$为卷积核在$\delta^l$ 加上零padding高宽 为$(k_1^{l-1}-1,k_2^{l-1}-1)$ 后的梯度矩阵$p\delta^{l}$上卷积
$$
\delta^{l-1}=\sum_{m=0}^{k_1^{l-1}-1} \sum_{n=0}^{k_2^{l-1}-1}rot_{180^\circ} W^{l-1}_{m,n}p\delta^{l}_{i+m,j+n} \tag {16}
$$




## 参考

1. [backpropagation-in-convolutional-neural-networks](http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/)
2. [矩阵乘法](https://baike.baidu.com/item/%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95/5446029)





